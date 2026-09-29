"""Read the same typed decisions from a vLLM endpoint instead of a local GGUF.

Lichen's method is engine-agnostic: build a prompt that ends where the label goes, read the
softmax over the label tokens of one next-token distribution. Only the way the logits arrive is
llama.cpp-specific. This module gets them over HTTP from vLLM; `method` does the rest, so
Method, variants, rotations, fibers, readings, shrink, disagreement, runoff and confidence are
the same code that serves a GGUF, not a second copy of it.

Two reasons it is worth having:

  * The local server answers one request at a time (see README's "Limits"). vLLM does
    continuous batching, and with --enable-prefix-caching gives for free the shared-prefix
    reuse that --batch hand-rolls. For a sweep over thousands of decisions that serialisation
    is the whole cost.
  * vLLM cannot read GGUF at all (its V1 engine lists GGUF as removed), so this is the only way
    to put the same question to an FP8 or AWQ checkpoint.

Each request names its label tokens in `allowed_token_ids`, and the endpoint must run with
--logprobs-mode processed_logprobs. The log-probabilities then come after the mask, so they are
the softmax over the label tokens alone, and the top-N list holds every label and nothing else.
With a raw mode the top-N is taken over the whole vocabulary, and a confident model ranks its
unlikely labels below other tokens: on gemma-4-26B-A4B, 3 of 6 labels were missing from the top
64. The greedy request applies no temperature, so the processed values are the masked logits
less one constant, which cancels in the softmax over the labels -- provided no penalty moves
them, so the request pins the penalties rather than inherit the server's defaults.

The server needs --max-logprobs >= (options x fibers), at most 62; its default of 20 is too low
for a wide choice. `_probs` still refuses a reply that lacks a label, which would otherwise read
as probability zero. (62 is len(method.LABELS); a raw-mode server fails that check on most wide
choices, which is how a missing --logprobs-mode shows up.)

vLLM is not deterministic across batches: the same request gave label probabilities up to 0.09
apart on gemma-4-26B-A4B, enough to move a borderline answer. VLLM_BATCH_INVARIANT=1 in the
server's environment made two full JevBench runs agree on all 231 items, for ~23% latency. That
holds within one running server: after a restart with the same launch, probabilities moved by
up to 0.14 on gemma-4-26B-A4B NVFP4 (vLLM 0.30).
A launch that serves this backend reproducibly:

    VLLM_BATCH_INVARIANT=1 vllm serve MODEL --enable-prefix-caching \\
      --logprobs-mode processed_logprobs --max-logprobs 64 --max-model-len 32768

A case's `images` are base64 data URLs, never links, so vLLM fetches nothing. They reach it as
image_url content parts, all of them in each copy of the state,
so the endpoint must allow images x --repeat per prompt with --limit-mm-per-prompt. With a limit
of 0 vLLM does not load the vision encoder, which on gemma-4-26B-A4B is 1.08 GiB taken from the
KV cache. There one image is 262 prompt tokens, which the n_ctx check counts, since /tokenize
does. With VLLM_BATCH_INVARIANT=1 image answers repeat bit for bit when requests come one at a
time; with 8 at once, one answer in 84 or 125 moved, with the same top answer.

`usage.input_tokens` is the full prompt as vLLM counts it. The llama.cpp backend counts only the
tokens it evaluated after its cached prefix, so the two are not comparable as costs.

Not ported: --recheck, whose second round would want the endpoint's own answer written back,
and --embedding. Asked for, they raise rather than be ignored. Nor are the prompts that
llamacpp.render builds per model -- Granite Guardian's criteria, Qwen3Guard's plain ChatML, and
closing a reasoning block a template leaves open (LFM2.5) -- so those models are refused rather
than read with a prompt they were never given (`check_model`). Nothing here needs vLLM
installed: the engine is reached over HTTP, and a `pip install lichen` is the whole dependency.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from urllib import error, request

import numpy

from ..method import ContextOverflow, Method, chat_messages


class Endpoint:
    """A vLLM OpenAI-compatible endpoint, in place of a model on this machine."""

    threaded = True  # vLLM batches for itself; requests may be served concurrently

    def __init__(self, endpoint: str, model: str, method: Method, top_logprobs: int = 64,
                 workers: int = 8, served: str | None = None, thinking: bool = False,
                 timeout: float = 600.0, n_ctx: int | None = None, priority: int = 0):
        for unsupported in ("recheck", "embedding"):
            if getattr(method, unsupported):
                raise SystemExit(f"--{unsupported} is not ported to the vLLM backend")
        self.endpoint = endpoint.rstrip("/")
        self.name = model           # the name a reply carries
        self.served = served or model  # the name this endpoint knows it by
        self.where = f"vLLM {self.served} at {self.endpoint}, n_ctx {n_ctx}, priority {priority}"
        self.top_logprobs = top_logprobs
        # The endpoint's --max-model-len also bounds its chat requests, so a judgment gets its
        # own, smaller limit here, checked against the prompt as the endpoint tokenizes it.
        self.n_ctx = n_ctx
        # Lower runs sooner. Anything but 0 needs the endpoint's --scheduling-policy priority,
        # so 0 is left out of the request.
        self.priority = priority
        self.workers = workers
        self.thinking = thinking
        self.timeout = timeout
        self._tokens: dict[str, int] = {}
        self._model_checked = False

    def close(self) -> None:
        pass  # the weights are the endpoint's

    def _post(self, path: str, body: dict) -> dict:
        req = request.Request(f"{self.endpoint}{path}", data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except error.HTTPError as exc:
            raise RuntimeError(f"{path} -> HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}") from exc

    def _get(self, path: str) -> dict:
        try:
            with request.urlopen(f"{self.endpoint}{path}", timeout=self.timeout) as r:
                return json.load(r)
        except error.HTTPError as exc:
            raise RuntimeError(f"{path} -> HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}") from exc

    def check_model(self, messages: list[dict]) -> None:
        """Refuse a model whose lichen prompt only the llama.cpp backend knows how to build.

        llamacpp.render gives Granite Guardian and Qwen3Guard their own prompts, and closes a
        reasoning block the template leaves open. Here the server's template renders the plain
        messages, so those models would be read at the wrong place or with the wrong question --
        and a distribution comes back either way. `--model` is a served name, which need not say
        what it serves, so the checkpoint path the server reports is checked too; and the open
        block is found in the rendered prompt itself, whatever the model is called. Once per
        endpoint: the answer cannot change while it is up.
        """
        if self._model_checked:
            return
        served = self._get("/v1/models").get("data", [])
        root = next((m.get("root") or "" for m in served if m.get("id") == self.served), "")
        names = f"{self.name} {self.served} {root}".lower()
        for family in ("granite-guardian", "qwen3guard"):
            if family in names:
                raise RuntimeError(f"{family} needs a prompt only the llama.cpp backend builds; "
                                   f"serve it from a GGUF rather than --vllm-endpoint")
        strs = self._post("/tokenize", {
            "model": self.served, "messages": messages, "add_generation_prompt": True,
            "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": self.thinking},
        }).get("token_strs") or []
        if not self.thinking and strs and strs[-1] == "<think>":
            raise RuntimeError(
                "this model's chat template opens a reasoning block with thinking off, so the "
                "next token is reasoning, not a label. The llama.cpp backend closes it; this one "
                "does not -- serve it from a GGUF rather than --vllm-endpoint")
        self._model_checked = True

    def check_labels(self, labels: list[str]) -> None:
        """lichen's invariant: every label is exactly one token, and no two share one.

        Same check, same reason -- a label that splits cannot be read from one next-token
        distribution, so it is an error rather than a quietly wrong answer.
        """
        unknown = [l for l in labels if l not in self._tokens]
        for label in unknown:
            ids = self._post("/tokenize", {"model": self.served, "prompt": label,
                                           "add_special_tokens": False}).get("tokens", [])
            if len(ids) != 1:
                raise ValueError(f"label {label!r} is {len(ids)} tokens, not one: {ids}")
            self._tokens[label] = ids[0]
        ids = [self._tokens[l] for l in labels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"labels share a token: {labels} -> {ids}")

    def label_probs(self, asked: list[dict], method: Method) -> tuple[list[numpy.ndarray], int]:
        """The label probabilities of each variant, and the prompt tokens the endpoint charged.

        The whole point of the port: hand every rotation or fibered prompt over at once and let
        vLLM's scheduler batch them, rather than walking them one at a time.
        """
        rendered = [chat_messages(v, method) for v in asked]
        self.check_model(rendered[0][0])

        def one(r):
            return self._probs(r[0], r[1], method.temperature)

        if len(rendered) == 1:
            read = [one(rendered[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(rendered))) as pool:
                read = list(pool.map(one, rendered))
        return [p for p, _ in read], sum(tokens for _, tokens in read)

    def _probs(self, messages: list[dict], labels: list[str],
               temperature: float) -> tuple[numpy.ndarray, int]:
        """One prompt: the softmax over its label tokens, and the tokens it cost.

        The count is returned rather than added to the endpoint, which several threads and
        several requests share.
        """
        self.check_labels(labels)
        ids = [self._tokens[l] for l in labels]
        if len(ids) > self.top_logprobs:
            raise ValueError(f"{len(ids)} labels; --top-logprobs is {self.top_logprobs}")
        body = {
            "model": self.served, "messages": messages,
            "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": len(ids),
            "allowed_token_ids": ids, "return_tokens_as_token_ids": True,
            # processed_logprobs puts every logits processor before the read, and vLLM fills any
            # parameter a request omits from the model's generation_config (or the server's
            # --override-generation-config). Greedy resets top-k, top-p and min-p, but not the
            # penalties: a repetition_penalty != 1 would scale the labels the prompt already
            # contains, which is not a constant and does not cancel. Pinned, so the read is the
            # masked logits whatever the server's sampling defaults.
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            # Lichen's chat_prompt renders with enable_thinking=False. It is load-bearing: with
            # thinking on, the first token is <think> and every decision is garbage -- silently,
            # since a distribution still comes back. A chat template defaults it to true, so this
            # override per request is what makes the method work at all.
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if self.n_ctx:
            need = self._post("/tokenize", {"model": self.served, "messages": messages,
                                            "add_generation_prompt": True,
                                            "chat_template_kwargs": body["chat_template_kwargs"]})["count"]
            if need > self.n_ctx:
                raise ContextOverflow(f"prompt of {need} tokens exceeds the context of {self.n_ctx}")
        if self.priority:
            body["priority"] = self.priority
        d = self._post("/v1/chat/completions", body)
        tokens = int((d.get("usage") or {}).get("prompt_tokens") or 0)
        content = (d["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            raise RuntimeError("no logprobs in reply; is the server built with logprobs support?")
        # With return_tokens_as_token_ids each token reads "token_id:<id>".
        top = {int(e["token"].rpartition(":")[2]): e["logprob"] for e in content[0].get("top_logprobs", [])}
        missing = [l for l, i in zip(labels, ids) if i not in top]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(labels)} labels absent from the reply ({missing[:6]}). "
                f"Is the endpoint running with --logprobs-mode processed_logprobs?")
        x = numpy.asarray([top[i] for i in ids], dtype=numpy.float64) / temperature
        return numpy.exp(x - numpy.logaddexp.reduce(x)), tokens
