"""The model side of Lichen: chat templates, next-token logits, and label probabilities.

A judgment is read from one forward pass: render the chat prompt with the
model's own template, evaluate it, take the logits at the last position, and
softmax over the first token of each answer label. The Evaluator does this for
several prompts at once, evaluating the prefix they share only once.
"""

from __future__ import annotations

import functools

import numpy
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only -- importing llama_cpp needs libcuda,
    from llama_cpp import Llama  # which a vLLM-only install does not have


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


class ContextOverflow(ValueError):
    """The prompt has more tokens than the model's context holds."""


def last_logits(model: Llama, prompt: str) -> numpy.ndarray:
    import llama_cpp  # lazy: only this path needs the library
    """Evaluate `prompt` from an empty context and return the next-token logits."""
    tokens = model.tokenize(prompt.encode(), add_bos=False, special=True)
    if len(tokens) > model.n_ctx():
        raise ContextOverflow(f"prompt of {len(tokens)} tokens exceeds the context of {model.n_ctx()}")
    model.reset()
    model.eval(tokens)
    # Without logits_all, Llama.eval leaves model.scores empty; llama.cpp still holds
    # the logits of the last position, so read them from the context.
    logits = numpy.ctypeslib.as_array(
        llama_cpp.llama_get_logits_ith(model.ctx, -1), shape=(model.n_vocab(),)
    )
    return logits


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
        import llama_cpp  # lazy, as in last_logits
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
