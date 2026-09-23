"""A local stand-in for TypeSafe's System One endpoint, backed by one GGUF model.

    python -m lichen.server --model MODEL.gguf [--repeat 2 --permute --batch] [--n-ctx N] [...]

`--help` lists every option.

POST /v1/systemone takes TypeSafe's request body ({state, model, questions})
and returns its response body ({model, answers, usage}), so a TypeSafe client
works against it unchanged. Each question goes to lichen.method.answer: one
forward pass, a softmax over the answer-label logits, and TypeSafe's documented
confidence. A request's `model` field is accepted and ignored, since the reply
names the model this server loaded, and no Authorization header is checked.

Differences from TypeSafe:
- A choice may have at most 26 options (one letter each); TypeSafe allows 255.
- Questions in one request are answered one after another, each from an
  empty context, unless --batch, which keeps the shared prefix cached.
  --permute, --repeat, --options-once and --recheck change how a question is
  put and how many passes it takes; see lichen/method.py.
- Requests are served one at a time, because a llama.cpp context is not
  safe to share between threads.

GET /health answers 200 with the loaded model's name once the model is loaded.
"""

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from llama_cpp import Llama

from .method import Method, answer, load, method_arguments, method_from, parse_overrides, positive
from .runtime import Evaluator
from .runtime import ContextOverflow

MAX_OPTIONS = 26
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


def respond(model: Llama, name: str, method: Method, evaluator: Evaluator | None, request: Request) -> dict:
    answers, input_tokens = {}, 0
    for qid, question in request.questions.items():
        a = answer(model, {"state": request.state, "question": question}, method, name, evaluator)
        input_tokens += a.pop("prompt_tokens")
        a.pop("latency_ms")
        answers[qid] = a
    return {"model": name, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def handler(model: Llama, name: str, method: Method, evaluator: Evaluator | None) -> type[BaseHTTPRequestHandler]:
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
                reply = respond(model, name, method, evaluator, parse_request(raw))
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
    method_arguments(ap)
    args = ap.parse_args()
    name = pathlib.Path(args.model).stem
    method = method_from(args, guard=not args.no_guard)
    model, evaluator = load(args.model, args.n_ctx, method, args.n_ubatch, parse_overrides(args.kv))
    print(f"serving {name} on {args.host}:{args.port}, n_ctx {args.n_ctx}, {method}", file=sys.stderr, flush=True)
    HTTPServer((args.host, args.port), handler(model, name, method, evaluator)).serve_forever()


if __name__ == "__main__":
    main()
