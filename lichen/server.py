"""A local stand-in for TypeSafe's System One endpoint, backed by one GGUF model.

    python -m lichen.server --model MODEL.gguf [--repeat 2 --permute --batch --fibers 2 --fiber-map
        --shrink --temperature 1.25] [--n-ctx N] [...]

`--help` lists every option.

POST /v1/systemone takes TypeSafe's request body ({state, model, questions})
and returns its response body ({model, answers, usage}), so a TypeSafe client
works against it unchanged. Each question goes to lichen.method.answer: one
forward pass, a softmax over the answer-label logits, and TypeSafe's documented
confidence. A request's `model` field is accepted and ignored, since the reply
names the model this server loaded, and no Authorization header is checked.

Differences from TypeSafe:
- A choice may have at most 62 options (A-Z, a-z, 0-9); TypeSafe allows 255.
- Questions in one request are answered one after another, each from an
  empty context, unless --batch, which keeps the shared prefix cached.
  --permute, --fibers, --repeat, --shrink and the other method options
  change how a question is put and how many passes it takes; see
  lichen/method.py.
- Requests are served one at a time, because a llama.cpp context is not
  safe to share between threads.

GET /health answers 200 with the loaded model's name once the model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only -- importing llama_cpp needs libcuda,
    from llama_cpp import Llama  # which a vLLM-only install does not have

from .method import LABELS, Method, answer, load, method_arguments, method_from, parse_overrides, positive
from .runtime import Evaluator
from .runtime import ContextOverflow

MAX_OPTIONS = len(LABELS)
MAX_LEVELS = 10
MAX_BODY = 8 << 20  # bytes; a 16k-token context holds far less


class BadRequest(ValueError):
    """The request body does not describe a System One request."""


@dataclass(frozen=True)
class Request:
    state: object
    questions: dict[str, dict]


def _instructions_ok(value) -> bool:
    return isinstance(value, (str, dict, list)) and bool(value)


def parse_question(qid: str, q) -> dict:
    if not isinstance(q, dict):
        raise BadRequest(f"questions.{qid} must be an object")
    kind = q.get("type")
    if not _instructions_ok(q.get("instructions")):
        raise BadRequest(f"questions.{qid}.instructions must be a non-empty string, object, or array")
    criteria = q.get("criteria")
    match kind:
        case "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise BadRequest(f"questions.{qid}.criteria must map at least two options to descriptions")
            if len(criteria) > MAX_OPTIONS:
                raise BadRequest(f"questions.{qid} has {len(criteria)} options; this server takes at most {MAX_OPTIONS}")
        case "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_LEVELS:
                raise BadRequest(f"questions.{qid}.criteria must be a list of 2 to {MAX_LEVELS} levels")
        case "noul":
            if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
                raise BadRequest(f"questions.{qid}.criteria may only have the keys 'true' and 'false'")
        case _:
            raise BadRequest(f"questions.{qid}.type must be 'noul', 'choice', or 'score'")
    return q


def parse_request(raw: bytes) -> Request:
    try:
        body = json.loads(raw)
    except ValueError as exc:  # JSONDecodeError, or bytes that are not UTF-8
        raise BadRequest(f"body is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise BadRequest("body must be an object")
    if "state" not in body:
        raise BadRequest("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a non-empty object")
    return Request(body["state"], {qid: parse_question(qid, q) for qid, q in questions.items()})


def trace_line(state, question: dict, a: dict) -> str:
    """One question's readings as a JSON line, keyed by a hash of its state."""
    state_sha = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    served = a.get("probabilities") or {"yes": a.get("noul")}
    return json.dumps({"state_sha256": state_sha, "instructions": question["instructions"],
                       "type": question["type"], "answer": served, "readings": a["readings"]})


def respond(answer_fn, name: str, request: Request, trace=None) -> dict:
    answers, input_tokens = {}, 0
    for qid, question in request.questions.items():
        a = answer_fn({"state": request.state, "question": question})
        if trace:
            print(trace_line(request.state, question, a), file=trace, flush=True)
        input_tokens += a.pop("prompt_tokens")
        a.pop("latency_ms")
        a.pop("readings")
        answers[qid] = a
    return {"model": name, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def handler(answer_fn, name: str, trace=None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        timeout = 30  # seconds a client may take to send its request

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"model": name})
            else:
                self._send(404, {"detail": f"no such path: {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._send(404, {"detail": f"no such path: {self.path}"})
                return
            start = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                self._send(400, {"detail": f"Content-Length must be an integer from 0 to {MAX_BODY}"})
                return
            raw = self.rfile.read(length)
            try:
                reply = respond(answer_fn, name, parse_request(raw), trace)
            except (BadRequest, ContextOverflow) as exc:
                self._send(422, {"detail": str(exc)})
                return
            except Exception as exc:
                self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})
                raise
            self._send(200, reply)
            ms = (time.perf_counter() - start) * 1000
            print(f"{len(reply['answers'])} questions {reply['usage']['input_tokens']} tokens {ms:.0f} ms",
                  file=sys.stderr, flush=True)

        def log_message(self, fmt: str, *args) -> None:
            pass  # one line per request is printed by do_POST

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a GGUF file")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--n-ctx", type=positive, default=32768)
    ap.add_argument("--no-guard", action="store_true",
                    help="leave out the sentence that tells the model the state is data")
    ap.add_argument("--vllm-endpoint", metavar="URL",
                    help="serve from a vLLM OpenAI-compatible endpoint instead of a local GGUF. "
                         "--model is then only a name unless --vllm-model is given.")
    ap.add_argument("--vllm-model", metavar="NAME", help="served-model-name on that vLLM")
    ap.add_argument("--vllm-workers", type=positive, default=8,
                    help="variants asked concurrently with --vllm-endpoint")
    ap.add_argument("--top-logprobs", type=positive, default=64,
                    help="logprobs requested per position; must be >= options x fibers and "
                         "<= the endpoint's --max-logprobs")
    ap.add_argument("--trace", metavar="PATH",
                    help="append each question's separate readings (rotations or fiber blocks) to PATH as JSON lines")
    method_arguments(ap)
    args = ap.parse_args()
    # With --vllm-endpoint, --model is a served-model-name rather than a path, and .stem would
    # mangle it -- Path("qwen3.8-27b-fp8").stem is "qwen3", which the endpoint then rejects.
    name = args.model if args.vllm_endpoint else pathlib.Path(args.model).stem
    method = method_from(args, guard=not args.no_guard)
    if args.vllm_endpoint:
        # The same method, logits fetched over HTTP. See lichen/vllm.py for why log-probabilities
        # stand in for logits here, and for the --max-logprobs the endpoint needs.
        from . import method as method_module
        from .vllm import Jev, Vllm
        backend = Vllm(args.vllm_endpoint, args.vllm_model or args.model, args.top_logprobs)
        answer_fn = Jev(method_module, backend, method, args.vllm_workers).answer
        where = f"vLLM {args.vllm_model or name} at {args.vllm_endpoint}"
        server = ThreadingHTTPServer      # vLLM batches; serve requests concurrently
    else:
        model, evaluator = load(args.model, args.n_ctx, method, args.n_ubatch,
                                parse_overrides(args.kv))
        answer_fn = lambda case: answer(model, case, method, name, evaluator)  # noqa: E731
        where = f"n_ctx {args.n_ctx}"
        server = HTTPServer
    print(f"serving {name} on {args.host}:{args.port}, {where}, {method}",
          file=sys.stderr, flush=True)
    trace = open(args.trace, "a", encoding="utf-8") if args.trace else None
    server((args.host, args.port), handler(answer_fn, name, trace)).serve_forever()


if __name__ == "__main__":
    main()
