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
less one constant, which cancels in the softmax over the labels.

The server needs --max-logprobs >= (options x fibers), at most 62; its default of 20 is too low
for a wide choice. `_probs` still refuses a reply that lacks a label, which would otherwise read
as probability zero.

Not ported: --recheck, whose second round would want the endpoint's own answer written back,
and --embedding. Asked for, they raise rather than be ignored. Nothing here needs vLLM
installed: the engine is reached over HTTP, and a `pip install lichen` is the whole dependency.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from urllib import error, request

import numpy

from ..method import Method, chat_messages


class Endpoint:
    """A vLLM OpenAI-compatible endpoint, in place of a model on this machine."""

    threaded = True  # vLLM batches for itself; requests may be served concurrently

    def __init__(self, endpoint: str, model: str, method: Method, top_logprobs: int = 64,
                 workers: int = 8, served: str | None = None, thinking: bool = False,
                 timeout: float = 600.0):
        for unsupported in ("recheck", "embedding"):
            if getattr(method, unsupported):
                raise SystemExit(f"--{unsupported} is not ported to the vLLM backend")
        self.endpoint = endpoint.rstrip("/")
        self.name = model           # the name a reply carries
        self.served = served or model  # the name this endpoint knows it by
        self.where = f"vLLM {self.served} at {self.endpoint}"
        self.top_logprobs = top_logprobs
        self.workers = workers
        self.thinking = thinking
        self.timeout = timeout
        self._tokens: dict[str, int] = {}

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
            # Lichen's chat_prompt renders with enable_thinking=False. It is load-bearing: with
            # thinking on, the first token is <think> and every decision is garbage -- silently,
            # since a distribution still comes back. A chat template defaults it to true, so this
            # override per request is what makes the method work at all.
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
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
