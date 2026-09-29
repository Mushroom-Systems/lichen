"""A stub vLLM server, just enough of its API for the vLLM backend.

It answers the four routes the backend uses, and does what vLLM does with a masked read: it
returns the log-softmax over `allowed_token_ids` only, each token as "token_id:<id>". Every
request body is recorded, so a test can check what was sent as well as what came back.

These tests check Lichen's side of the conversation. They cannot show that a real vLLM
behaves the way the stub does; the masked read was checked against one separately.
"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class Stub:
    def __init__(self):
        # A single character is one token, id = its code point; anything longer is split
        # into characters, so a multi-character label is several tokens, as it would be.
        self.words = {"Yes": 1001, "No": 1002}
        self.logits: dict[int, float] = {}   # id -> logit; unlisted ids get 0
        self.root = "/models/some-model"
        self.rendered_tail = ["<|im_start|>", "assistant", "\n"]
        self.prompt_tokens = 100
        self.requests: list[tuple[str, dict]] = []

    def tokenize(self, text: str) -> list[int]:
        return [self.words[text]] if text in self.words else [ord(c) for c in text]

    def reply(self, path: str, body: dict) -> dict:
        self.requests.append((path, body))
        if path == "/v1/models":
            return {"data": [{"id": "stub", "root": self.root}]}
        if path == "/tokenize":
            if "messages" in body:
                return {"tokens": list(range(len(self.rendered_tail))),
                        "token_strs": self.rendered_tail, "count": self.prompt_tokens}
            return {"tokens": self.tokenize(body["prompt"])}
        if path == "/v1/chat/completions":
            ids = body["allowed_token_ids"]
            z = math.log(sum(math.exp(self.logits.get(i, 0.0)) for i in ids))
            top = [{"token": f"token_id:{i}", "logprob": self.logits.get(i, 0.0) - z} for i in ids]
            return {"usage": {"prompt_tokens": self.prompt_tokens},
                    "choices": [{"logprobs": {"content": [{"top_logprobs": top[:body["top_logprobs"]]}]}}]}
        raise KeyError(path)


@pytest.fixture
def stub():
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self, body):
            try:
                data, code = json.dumps(state.reply(self.path, body)).encode(), 200
            except KeyError:
                data, code = b"{}", 404
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._answer({})

        def do_POST(self):
            self._answer(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
