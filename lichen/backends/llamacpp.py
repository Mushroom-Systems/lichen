"""The llama.cpp backend: chat templates, next-token logits, and label probabilities.

A judgment is read from one forward pass: render the chat prompt with the
model's own template, evaluate it, take the logits at the last position, and
softmax over the first token of each answer label. The Evaluator does this for
several prompts at once, evaluating the prefix they share only once.

`method` writes the messages and reads the answer; everything here knows
llama.cpp, which is why `import llama_cpp`, and the libcuda it loads, is in
this module and nowhere else.
"""

import functools
import pathlib

import llama_cpp
import numpy
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import Llama

from ..method import ContextOverflow, Method, chat_messages, question_part, state_part
from . import embed

# Qwen3Guard's own template can only ask its fixed safety question, so it is
# given the plain Qwen3 chat format, with thinking closed as Qwen3 does it.
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _raise_exception(message: str):
    raise ValueError(message)


class _Generation(Extension):
    """`{% generation %}...{% endgeneration %}`, a tag from Hugging Face's chat
    templates (LFM2.5 uses it). transformers uses it to mark the assistant's own
    text; for rendering a prompt it outputs its body unchanged."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body).set_lineno(lineno)


@functools.cache
def _compiled(source: str):
    """A chat template compiled once. Compiling Gemma 4's template on every call
    took 45% of the time to answer a question with gemma-4-E4B."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_Generation])
    env.globals["raise_exception"] = _raise_exception
    return env.from_string(source)


def chat_prompt(model: Llama, messages: list[dict], template: str | None = None, **context) -> str:
    """A chat template over `messages`, open for the assistant's reply.

    The model's own template unless `template` is given; `context` reaches the
    template as extra variables.
    """
    compiled = _compiled(template or model.metadata["tokenizer.chat_template"])
    return compiled.render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token=model.detokenize([model.token_bos()], special=True).decode(),
        eos_token=model.detokenize([model.token_eos()], special=True).decode(),
        **context,
    )


def render(name: str, model: Llama, case: dict, method: Method,
           previous: str | None = None) -> tuple[str, list[str], list[str]]:
    """The prompt for one case, and the labels to read and the keys they stand for.

    `previous` is the label of a first answer to write back for a recheck.
    """
    if case.get("images"):
        raise ValueError("the llama.cpp backend reads no images; serve images from --vllm-endpoint")
    if "granite-guardian" in name:
        # Granite Guardian judges the user message against `custom_criteria` and
        # answers "<score> yes </score>". The question goes in the criteria, the
        # state is the message, and the prompt ends at "<score>" so the next
        # token is the label, with the leading space its tokenizer puts on it
        # where the space and the label make one token.
        body, labels, keys = question_part(case["question"])
        if case["question"]["type"] == "noul":
            labels = ["yes", "no"]
            body = body.replace("Answer Yes or No.", "Answer yes or no.")
        criteria, _, schema = body.rpartition("\n\n")
        prompt = chat_prompt(model, [{"role": "user", "content": state_part(case["state"])}],
                             guardian_config={"custom_criteria": criteria, "custom_scoring_schema": schema})
        spaced = [" " + l for l in labels]
        if all(len(model.tokenize(l.encode(), add_bos=False)) == 1 for l in spaced):
            return prompt + "<score>", spaced, keys
        return prompt + "<score> ", labels, keys  # digits: the space is its own token
    messages, labels, keys = chat_messages(case, method, previous)
    template = CHATML if "Qwen3Guard" in name else None
    prompt = chat_prompt(model, messages, template)
    # Some templates open a reasoning block for the reply and have no switch to
    # leave it out (LFM2.5 always ends in "<think>"). Closing it at once gives an
    # empty block, as the Qwen and Nemotron templates do with thinking off, so
    # the next token is the answer.
    if prompt.endswith("<think>"):
        prompt += "</think>"
    return prompt, labels, keys


def last_logits(model: Llama, prompt: str) -> numpy.ndarray:
    """Evaluate `prompt` from an empty context and return the next-token logits."""
    tokens = model.tokenize(prompt.encode(), add_bos=False, special=True)
    if len(tokens) > model.n_ctx():
        raise ContextOverflow(f"prompt of {len(tokens)} tokens exceeds the context of {model.n_ctx()}")
    model.reset()
    model.eval(tokens)
    # Without logits_all, Llama.eval leaves model.scores empty; llama.cpp still holds
    # the logits of the last position, so read them from the context. The copy is
    # load-bearing, as it is in Evaluator.logits: as_array gives a view on llama.cpp's
    # own buffer, which the next eval overwrites. Without the copy two of these read
    # the same numbers, and --runoff without --batch split every answer 50/50.
    return numpy.ctypeslib.as_array(
        llama_cpp.llama_get_logits_ith(model.ctx, -1), shape=(model.n_vocab(),)
    ).copy()


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Evaluator:
    """Next-token logits for several prompts at once, sharing their common prefix.

    Each Evaluator owns a llama.cpp context on `model`'s weights, with one
    sequence per prompt in a unified KV cache. Whatever prefix the prompts
    share is evaluated once in sequence 0 and copied to the others, at no
    compute cost in a unified cache, and the differing endings then go through
    the model in one batch. Sequence 0 keeps its prefix between calls, so a
    later call evaluates only the tokens after the part that still matches.

    A model with recurrent layers (Qwen3.5 and Qwen3.6 are hybrids) cannot cut
    a sequence part way. When a cut fails, the cache is cleared and the prefix
    is evaluated from the start.
    """

    def __init__(self, model: Llama, n_ctx: int = 8192, n_seq: int = 8, n_ubatch: int = 1024):
        params = llama_cpp.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_ctx
        params.n_ubatch = min(n_ctx, n_ubatch)
        params.n_seq_max = n_seq
        params.kv_unified = True
        params.swa_full = True
        self.model, self.n_ctx, self.n_seq = model, n_ctx, n_seq
        self.ctx = llama_cpp.llama_init_from_model(model.model, params)
        if not self.ctx:
            raise RuntimeError("llama_init_from_model failed")
        self.memory = llama_cpp.llama_get_memory(self.ctx)
        self.batch = llama_cpp.llama_batch_init(n_ctx, 0, n_seq)
        self.cached: list[int] = []  # the prefix held in sequence 0

    def close(self) -> None:
        llama_cpp.llama_batch_free(self.batch)
        llama_cpp.llama_free(self.ctx)

    def _decode(self, items: list[tuple[int, int, int, bool]]) -> None:
        """Decode (token, position, sequence, wants logits) items as one batch."""
        b = self.batch
        for i, (token, pos, seq, logits) in enumerate(items):
            b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = token, pos, 1, logits
            b.seq_id[i][0] = seq
        b.n_tokens = len(items)
        if rc := llama_cpp.llama_decode(self.ctx, b):
            raise RuntimeError(f"llama_decode returned {rc}")

    def _keep_prefix(self, prefix: list[int]) -> int:
        """Cut every sequence back to the cached tokens `prefix` still shares; return that length."""
        keep = _common_prefix(self.cached, prefix)
        ok = llama_cpp.llama_memory_seq_rm(self.memory, 0, keep, -1)
        for s in range(1, self.n_seq):
            ok = llama_cpp.llama_memory_seq_rm(self.memory, s, -1, -1) and ok
        if not ok:
            llama_cpp.llama_memory_clear(self.memory, True)
            keep = 0
        self.cached = self.cached[:keep]
        return keep

    def logits(self, prompts: list[str]) -> tuple[list[numpy.ndarray], int]:
        """The next-token logits after each prompt, and the tokens evaluated to get them."""
        if len(prompts) > self.n_seq:
            raise ValueError(f"{len(prompts)} prompts; this evaluator holds {self.n_seq} sequences")
        toks = [self.model.tokenize(p.encode(), add_bos=False, special=True) for p in prompts]
        shared = min(len(t) for t in toks) - 1  # each prompt keeps at least its last token
        for t in toks[1:]:
            shared = min(shared, _common_prefix(toks[0], t))
        prefix = toks[0][:shared]
        need = shared + sum(len(t) - shared for t in toks)
        if need > self.n_ctx:
            raise ContextOverflow(f"prompts need {need} tokens; the context holds {self.n_ctx}")

        keep = self._keep_prefix(prefix)
        if keep < shared:
            self._decode([(tok, keep + j, 0, False) for j, tok in enumerate(prefix[keep:])])
        self.cached = prefix
        for s in range(1, len(toks)):
            llama_cpp.llama_memory_seq_cp(self.memory, 0, s, -1, -1)

        items, last = [], []
        for s, t in enumerate(toks):
            tail = t[shared:]
            items += [(tok, shared + j, s, j == len(tail) - 1) for j, tok in enumerate(tail)]
            last.append(len(items) - 1)
        self._decode(items)
        n_vocab = self.model.n_vocab()
        out = [numpy.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(self.ctx, i), shape=(n_vocab,)).copy()
               for i in last]
        return out, (shared - keep) + len(items)


def label_probabilities(model: Llama, logits: numpy.ndarray, labels: list[str],
                        temperature: float = 1.0) -> numpy.ndarray:
    """Softmax over the token of each label, of the logits divided by `temperature`.

    A label of more than one token, or two labels with the same token, cannot
    be read from one next-token distribution, so either is an error rather
    than a wrong answer.
    """
    tokens = [model.tokenize(label.encode(), add_bos=False) for label in labels]
    if any(len(t) != 1 for t in tokens):
        raise ValueError(f"labels are not one token each: {labels} -> {tokens}")
    token_ids = [t[0] for t in tokens]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"labels share a token: {labels} -> {token_ids}")
    choice_logits = numpy.asarray([logits[i] for i in token_ids]) / temperature
    return numpy.exp(choice_logits - numpy.logaddexp.reduce(choice_logits))


def batched_logits(evaluator: Evaluator, prompts: list[str]) -> tuple[list[numpy.ndarray], int]:
    """Logits for every prompt, in groups that fit the evaluator's sequences and context.

    A choice may have more options than the evaluator has sequences, and many
    long prompts may not fit its context together; a group that does not fit
    is split in half. One prompt too long for the context still raises.
    """
    if len(prompts) > evaluator.n_seq:
        head, n = batched_logits(evaluator, prompts[:evaluator.n_seq])
        tail, m = batched_logits(evaluator, prompts[evaluator.n_seq:])
        return head + tail, n + m
    try:
        return evaluator.logits(prompts)
    except ContextOverflow:
        if len(prompts) == 1:
            raise
        half = len(prompts) // 2
        head, n = batched_logits(evaluator, prompts[:half])
        tail, m = batched_logits(evaluator, prompts[half:])
        return head + tail, n + m


def parse_overrides(pairs: list[str]) -> dict:
    """llama.cpp metadata overrides from KEY=VALUE, such as gemma4.expert_used_count=6.

    A value that reads as an integer or a float is passed as one; true and
    false as booleans; anything else as a string.
    """
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not key or not raw:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        if raw in ("true", "false"):
            out[key] = raw == "true"
            continue
        for kind in (int, float):
            try:
                out[key] = kind(raw)
                break
            except ValueError:
                pass
        else:
            out[key] = raw
    return out


class Gguf:
    """A GGUF model on the local GPU.

    `label_probs` reads a question's variants: with `method.batch` they go
    through the Evaluator, which evaluates the prefix they share once,
    otherwise one at a time from an empty context.
    """

    threaded = False  # a llama.cpp context is not safe to share between threads

    def __init__(self, name: str, model: Llama, n_ctx: int, evaluator: Evaluator | None = None):
        # n_ctx is the context a judgment gets. With --batch the Llama's own is
        # 256 and the Evaluator holds this one, so the startup line reports it.
        self.name, self.model, self.evaluator, self.where = name, model, evaluator, f"n_ctx {n_ctx}"

    def label_probs(self, asked: list[dict], method: Method) -> tuple[list[numpy.ndarray], int]:
        """The label probabilities of each variant, and the tokens evaluated for them."""
        if method.batch and not method.recheck:
            rendered = [render(self.name, self.model, v, method) for v in asked]
            logits, tokens = batched_logits(self.evaluator, [prompt for prompt, _, _ in rendered])
            return [label_probabilities(self.model, row, labels, method.temperature)
                    for (_, labels, _), row in zip(rendered, logits, strict=True)], tokens
        probs, tokens = [], 0
        for variant in asked:
            prompt, labels, _ = render(self.name, self.model, variant, method)
            p = label_probabilities(self.model, last_logits(self.model, prompt), labels, method.temperature)
            tokens += self.model.n_tokens
            if method.recheck:
                first = labels[int(numpy.argmax(p))].strip()
                prompt, labels, _ = render(self.name, self.model, variant, method, previous=first)
                p = label_probabilities(self.model, last_logits(self.model, prompt), labels, method.temperature)
                tokens += self.model.n_tokens
            probs.append(p)
        return probs, tokens

    def close(self) -> None:
        if self.evaluator:
            self.evaluator.close()
        self.model.close()


class Embedding:
    """An embedding model, which answers by cosine similarity rather than label logits."""

    threaded = False

    def __init__(self, name: str, model: Llama, n_ctx: int):
        self.name, self.model, self.where = name, model, f"n_ctx {n_ctx}"

    def similarities(self, case: dict) -> tuple[numpy.ndarray, list[str], int]:
        return embed.probabilities(self.model, case)

    def close(self) -> None:
        self.model.close()


def backend(gguf: str, n_ctx: int, method: Method, n_ubatch: int = 1024,
            kv: list[str] | None = None) -> Gguf | Embedding:
    """The model, with an Evaluator on it when `method.batch`.

    With --batch the Llama object only tokenizes and renders, so its own
    context is kept small and the Evaluator holds the KV cache.
    """
    name = pathlib.Path(gguf).stem
    if method.embedding:
        return Embedding(name, embed.load(gguf), n_ctx)
    model = Llama(model_path=gguf, n_ctx=256 if method.batch else n_ctx, n_gpu_layers=-1, verbose=False,
                  kv_overrides=parse_overrides(kv or []) or None)
    return Gguf(name, model, n_ctx, Evaluator(model, n_ctx, n_ubatch=n_ubatch) if method.batch else None)
