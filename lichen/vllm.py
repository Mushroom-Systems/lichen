"""Serve the same typed decisions from a vLLM endpoint instead of a local GGUF.

Lichen's method is engine-agnostic: build a prompt that ends where the label goes, read the
softmax over the label tokens of one next-token distribution. Only the way the logits arrive is
llama.cpp-specific. This module gets them over HTTP from vLLM, and reuses `method` for
everything else -- Method, variants, rotations, fibers, readings, shrink, disagreement,
confidence are imported, not reimplemented.

Two reasons it is worth having:

  * The local server answers one request at a time (see README's "Limits"). vLLM does
    continuous batching, and with --enable-prefix-caching gives for free the shared-prefix
    reuse that --batch hand-rolls. For a sweep over thousands of decisions that serialisation
    is the whole cost.
  * vLLM cannot read GGUF at all (its V1 engine lists GGUF as removed), so this is the only way
    to put the same question to an FP8 or AWQ checkpoint.

vLLM returns log-probabilities where `label_probabilities` wants logits. They are interchangeable
here: log_softmax differs from the logits by a constant that is the same for every token at this
position, and dividing by the temperature divides that constant too, so it cancels in the softmax
over the label subset. What matters is that they are RAW -- vLLM's --logprobs-mode must be
raw_logprobs (the default) or raw_logits, never a `processed` mode.

The server needs --max-logprobs >= (options x fibers); its default of 20 is too low for a wide
choice, and a label missing from the top-N would otherwise read as probability zero, so
`label_probs` checks and refuses.

Not ported: --runoff and --recheck (both off by default, recheck is documented as making answers
worse, and runoff needs a second dependent round). Asked for, they raise rather than be ignored.
"""

import json
import time
from urllib import error, request
from concurrent.futures import ThreadPoolExecutor

import numpy

from . import method as method_mod_default  # noqa: F401  (documents the dependency)


class Vllm:
    """The three calls lichen needs from a model, over vLLM's OpenAI-compatible API."""

    def __init__(self, endpoint: str, model: str, top_logprobs: int = 64,
                 thinking: bool = False, timeout: float = 600.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.top_logprobs = top_logprobs
        self.thinking = thinking
        self.timeout = timeout
        self._tokens: dict[str, int] = {}
        self.prompt_tokens = 0          # server's usage accounting reads and resets this

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
            ids = self._post("/tokenize", {"model": self.model, "prompt": label,
                                           "add_special_tokens": False}).get("tokens", [])
            if len(ids) != 1:
                raise ValueError(f"label {label!r} is {len(ids)} tokens, not one: {ids}")
            self._tokens[label] = ids[0]
        ids = [self._tokens[l] for l in labels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"labels share a token: {labels} -> {ids}")

    def label_probs(self, messages: list[dict], labels: list[str], temperature: float) -> numpy.ndarray:
        """Softmax over the label tokens of the next-token distribution, at `temperature`."""
        self.check_labels(labels)
        body = {
            "model": self.model, "messages": messages,
            "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": self.top_logprobs,
            # Lichen's chat_prompt renders with enable_thinking=False. It is load-bearing: with
            # thinking on, the first token is <think> and every decision is garbage -- silently,
            # since a distribution still comes back. Our compose defaults it to true, so this
            # override per request is what makes the method work at all.
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        d = self._post("/v1/chat/completions", body)
        self.prompt_tokens += int((d.get("usage") or {}).get("prompt_tokens") or 0)
        content = (d["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            raise RuntimeError("no logprobs in reply; is the server built with logprobs support?")
        top = {e["token"]: e["logprob"] for e in content[0].get("top_logprobs", [])}
        missing = [l for l in labels if l not in top]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(labels)} labels absent from the top {self.top_logprobs} "
                f"({missing[:6]}). Raise the server's --max-logprobs; a missing label would "
                f"otherwise read as probability zero.")
        x = numpy.asarray([top[l] for l in labels], dtype=numpy.float64) / temperature
        return numpy.exp(x - numpy.logaddexp.reduce(x))


class Jev:
    """lichen's `probabilities`/`answer`, with vLLM underneath and variants asked in parallel."""

    def __init__(self, method_mod, backend: Vllm, method, workers: int = 8):
        self.method_mod, self.backend, self.method, self.workers = (
            method_mod, backend, method, workers)
        for unsupported in ("runoff", "recheck", "embedding"):
            if getattr(method, unsupported, None):
                raise SystemExit(f"--{unsupported} is not ported to the vLLM backend")

    def render(self, case: dict):
        """lichen's render(), stopping at the messages instead of templating them.

        Upstream then calls chat_prompt() to produce a string; vLLM applies the model's own
        template server-side, so we hand it the messages and skip that step entirely.
        """
        m, method = self.method_mod, self.method
        earlier = case.get("earlier_question") if method.rotate_last else None
        body, labels, keys = m.user_message(case["state"], case["question"], method.repeat,
                                            method.options_once, method.question_first,
                                            method.compact_json, earlier)
        messages = [{"role": "system", "content": method.system},
                    {"role": "user", "content": body}]
        return messages, labels, keys

    def probabilities(self, case: dict):
        """P over the answer keys, and every reading that went into it (as `method` does)."""
        m, method = self.method_mod, self.method
        _, _, keys = self.render(case)
        asked = m.variants(case, method)
        rendered = [self.render(v) for v in asked]

        # The whole point of the port: ask every rotation / fibered prompt at once and let
        # vLLM's scheduler batch them, rather than walking them one at a time.
        def one(r):
            return self.backend.label_probs(r[0], r[1], method.temperature)

        if len(rendered) == 1:
            probs = [one(rendered[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(rendered))) as pool:
                probs = list(pool.map(one, rendered))

        parts = []
        for (_, _, variant_keys), p in zip(rendered, probs, strict=True):
            parts += m.readings(p, variant_keys)

        # Verbatim from upstream: a fibered list's blocks add up, separate prompts average.
        per_prompt = len(parts) // len(asked)
        total = numpy.asarray([sum(r.get(k, 0.0) for r in parts) for k in keys]) * per_prompt / len(parts)
        lam = m.disagreement(parts)
        if method.shrink:
            total = (1 - lam) * total + lam / len(keys)
        return total, keys, parts

    def answer(self, case: dict) -> dict:
        m = self.method_mod
        question = case["question"]
        start = time.perf_counter()
        self.backend.prompt_tokens = 0
        p, keys, parts = self.probabilities(case)
        probs = {k: round(float(v), 4) for k, v in zip(keys, p, strict=True)}
        out = {"type": question["type"], "latency_ms": (time.perf_counter() - start) * 1000,
               "prompt_tokens": self.backend.prompt_tokens, "readings": parts}
        match question["type"]:
            case "choice":
                out |= {"choice": keys[int(numpy.argmax(p))], "probabilities": probs,
                        "confidence": round(m.confidence(p), 4)}
            case "noul":
                out |= {"noul": probs["yes"]}
            case "score":
                out |= {"score": float(numpy.dot(p, numpy.arange(len(p)))), "probabilities": probs,
                        "confidence": round(m.confidence(p), 4),
                        "legend": {str(i): m.text(level) for i, level in enumerate(question["criteria"])}}
        return out
