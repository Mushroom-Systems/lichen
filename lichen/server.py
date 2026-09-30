"""A local stand-in for TypeSafe's System One endpoint, backed by one model.

    python -m lichen.server --model MODEL.gguf [--repeat 2 --permute --batch --fibers 2 --fiber-map
        --shrink --temperature 1.25] [--n-ctx N] [...]

A GGUF on this machine by default, or a vLLM endpoint with --vllm-endpoint; the
method is the same either way (lichen/backends/).

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
  safe to share between threads. With --vllm-endpoint the engine batches for
  itself, so they are served concurrently.

A request may add `images`, a list of images sent inline as base64 data URLs
(data:image/png;base64,...), which belong to the state: each copy of the state
in the prompt shows them. Links are refused. The server takes at most
--max-images of them, 0 by default, and only with --vllm-endpoint on a model
that reads images. TypeSafe's request has no such field.

GET /health answers 200 with the loaded model's name once the model is loaded.
"""

import argparse
import base64
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

from .backends import backend_from
from .method import ContextOverflow, LABELS, Method, answer, method_arguments, method_from, positive

MAX_OPTIONS = len(LABELS)
MAX_LEVELS = 10
MAX_BODY = 32 << 20  # bytes; the text a 16k-token context holds is far less, and images are data URLs
IMAGE_TYPES = ("png", "jpeg", "webp", "gif")


class BadRequest(ValueError):
    """The request body does not describe a System One request."""


@dataclass(frozen=True)
class Request:
    state: object
    questions: dict[str, dict]
    images: tuple[str, ...] = ()


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


def parse_image(i: int, value) -> str:
    """One image as a data URL with its bytes in base64. A link is refused: the engine would
    fetch whatever it names, from wherever it runs."""
    head, sep, data = value.partition(",") if isinstance(value, str) else ("", "", "")
    kind = head.removeprefix("data:image/").removesuffix(";base64")
    if not sep or not head.startswith("data:image/") or not head.endswith(";base64") or kind not in IMAGE_TYPES:
        raise BadRequest(f"images[{i}] must be a data URL, data:image/<{'|'.join(IMAGE_TYPES)}>;base64,<data>")
    try:
        base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise BadRequest(f"images[{i}] is not valid base64: {exc}") from exc
    return value


def parse_images(value, max_images: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise BadRequest("images must be a list of data URLs")
    if len(value) > max_images:
        raise BadRequest(f"{len(value)} images; this server takes at most {max_images} (--max-images)")
    return tuple(parse_image(i, v) for i, v in enumerate(value))


def parse_request(raw: bytes, max_images: int = 0) -> Request:
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
    return Request(body["state"], {qid: parse_question(qid, q) for qid, q in questions.items()},
                   parse_images(body.get("images"), max_images))


def trace_line(state, question: dict, a: dict, images: tuple[str, ...] = ()) -> str:
    """One question's readings as a JSON line, keyed by a hash of its state (and its images)."""
    state_sha = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    served = a.get("probabilities") or {"yes": a.get("noul")}
    line = {"state_sha256": state_sha, "instructions": question["instructions"],
            "type": question["type"], "answer": served, "readings": a["readings"]}
    if images:
        line["images_sha256"] = [hashlib.sha256(u.encode()).hexdigest() for u in images]
    return json.dumps(line)


def respond(backend, method: Method, request: Request, trace=None) -> dict:
    answers, input_tokens = {}, 0
    for qid, question in request.questions.items():
        a = answer(backend, {"state": request.state, "question": question, "images": request.images}, method)
        if trace:
            print(trace_line(request.state, question, a, request.images), file=trace, flush=True)
        input_tokens += a.pop("prompt_tokens")
        a.pop("latency_ms")
        a.pop("readings")
        answers[qid] = a
    return {"model": backend.name, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def handler(backend, method: Method, trace=None, max_images: int = 0) -> type[BaseHTTPRequestHandler]:
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
                self._send(200, {"model": backend.name})
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
                reply = respond(backend, method, parse_request(raw, max_images), trace)
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
    ap.add_argument("--model", required=True,
                    help="path to a GGUF file, or with --vllm-endpoint the name vLLM serves the model under")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--n-ctx", type=positive, default=32768,
                    help="most prompt tokens one judgment may use; a longer prompt is refused with 422. "
                         "On a GGUF it is also the context llama.cpp allocates.")
    ap.add_argument("--no-guard", action="store_true",
                    help="leave out the sentence that tells the model the state is data")
    ap.add_argument("--vllm-endpoint", metavar="URL",
                    help="serve from a vLLM OpenAI-compatible endpoint instead of a local GGUF. "
                         "--model is then only a name unless --vllm-model is given.")
    ap.add_argument("--vllm-model", metavar="NAME", help="served-model-name on that vLLM")
    ap.add_argument("--vllm-workers", type=positive, default=8,
                    help="variants asked concurrently with --vllm-endpoint")
    ap.add_argument("--vllm-priority", type=int, default=0,
                    help="vLLM scheduling priority of each judgment, lower is sooner; anything but 0 "
                         "needs the endpoint's --scheduling-policy priority")
    ap.add_argument("--max-images", type=int, default=0, metavar="N",
                    help="most images one request may carry (0: none). Needs --vllm-endpoint and a model "
                         "that reads images; the endpoint's --limit-mm-per-prompt must allow N x --repeat")
    ap.add_argument("--top-logprobs", type=positive, default=64,
                    help="the most labels one prompt may be read over (vLLM); must be >= options x "
                         "fibers and <= the endpoint's --max-logprobs")
    ap.add_argument("--trace", metavar="PATH",
                    help="append each question's separate readings (rotations or fiber blocks) to PATH as JSON lines")
    method_arguments(ap)
    args = ap.parse_args()
    method = method_from(args, guard=not args.no_guard)
    if args.max_images and not args.vllm_endpoint:
        raise SystemExit("--max-images needs --vllm-endpoint; the llama.cpp backend reads no images")
    backend = backend_from(args, method)
    print(f"serving {backend.name} on {args.host}:{args.port}, {backend.where}, {method}",
          file=sys.stderr, flush=True)
    trace = open(args.trace, "a", encoding="utf-8") if args.trace else None
    server = ThreadingHTTPServer if backend.threaded else HTTPServer
    server((args.host, args.port), handler(backend, method, trace, args.max_images)).serve_forever()


if __name__ == "__main__":
    main()
