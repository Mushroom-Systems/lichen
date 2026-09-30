"""The vLLM backend against a stub server: what it sends, and what it makes of the reply."""
import math

import numpy
import pytest

from lichen.backends.vllm import Endpoint
from lichen.method import ContextOverflow, Method, user_message
from lichen.server import BadRequest, parse_request, respond

MESSAGES = [{"role": "user", "content": "state and question"}]


def endpoint(stub, method=Method(), **kw):
    return Endpoint(stub.url, "stub", method, **kw)


def reads(stub):
    return [body for path, body in stub.requests if path == "/v1/chat/completions"]


def test_the_read_is_masked_to_the_labels_with_the_penalties_pinned(stub):
    endpoint(stub)._probs(MESSAGES, ["A", "B", "C"], 1.0)
    body, = reads(stub)
    assert body["allowed_token_ids"] == [ord("A"), ord("B"), ord("C")]
    assert body["top_logprobs"] == 3
    assert body["return_tokens_as_token_ids"] is True
    assert (body["repetition_penalty"], body["presence_penalty"], body["frequency_penalty"]) == (1.0, 0.0, 0.0)
    assert body["temperature"] == 0.0 and body["max_tokens"] == 1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_probabilities_are_the_tempered_softmax_over_the_labels(stub):
    stub.logits = {ord("A"): 2.0, ord("B"): 1.0, ord("C"): -1.0}
    p, tokens = endpoint(stub)._probs(MESSAGES, ["A", "B", "C"], 2.0)
    e = [math.exp(x / 2.0) for x in (2.0, 1.0, -1.0)]
    assert numpy.allclose(p, [v / sum(e) for v in e])
    assert tokens == stub.prompt_tokens


def test_labels_are_matched_by_token_id_not_by_text(stub):
    # "Yes" and "No" are one token each but not their characters' code points; a match by
    # decoded string would find nothing in replies that carry only ids.
    stub.logits = {1001: 3.0}
    p, _ = endpoint(stub)._probs(MESSAGES, ["Yes", "No"], 1.0)
    assert p[0] > 0.9


def test_a_label_missing_from_the_reply_is_an_error_not_a_zero(stub):
    ep = endpoint(stub)
    real = stub.reply
    def drop_last(path, body):
        out = real(path, body)
        if path == "/v1/chat/completions":
            out["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        return out
    stub.reply = drop_last
    with pytest.raises(RuntimeError, match="processed_logprobs"):
        ep._probs(MESSAGES, ["A", "B", "C"], 1.0)


def test_more_labels_than_top_logprobs_is_refused(stub):
    with pytest.raises(ValueError, match="--top-logprobs"):
        endpoint(stub, top_logprobs=2)._probs(MESSAGES, ["A", "B", "C"], 1.0)


def test_a_label_of_several_tokens_is_refused(stub):
    with pytest.raises(ValueError, match="not one"):
        endpoint(stub).check_labels(["A", "BC"])


@pytest.mark.parametrize("root", ["/models/granite-guardian-3.3-8b", "/models/Qwen3Guard-Gen-8B"])
def test_a_guard_model_is_refused_by_its_checkpoint_path(stub, root):
    stub.root = root   # served under a name that does not say what it is
    with pytest.raises(RuntimeError, match="llama.cpp backend"):
        endpoint(stub).check_model(MESSAGES)


def test_a_template_left_open_at_think_is_refused(stub):
    stub.rendered_tail = ["assistant", "\n", "<think>"]
    with pytest.raises(RuntimeError, match="reasoning block"):
        endpoint(stub).check_model(MESSAGES)


def test_the_model_is_checked_once(stub):
    ep = endpoint(stub)
    ep.check_model(MESSAGES)
    ep.check_model(MESSAGES)
    assert sum(path == "/v1/models" for path, _ in stub.requests) == 1


@pytest.mark.parametrize("flag", ["recheck", "embedding"])
def test_unported_options_are_refused(stub, flag):
    with pytest.raises(SystemExit):
        endpoint(stub, Method(**{flag: True}))


def test_a_request_is_answered_end_to_end(stub):
    stub.logits = {ord("A"): 4.0, 1001: 2.0}
    request = parse_request(b'''{"state": "Payouts failing for 3 days.", "model": "lichen",
        "questions": {"team": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": "b", "technical": "t", "sales": "s"}},
                      "urgent": {"type": "noul", "instructions": "Urgent?"}}}''')
    method = Method(permute=True)
    reply = respond(endpoint(stub, method), method, request)
    team = reply["answers"]["team"]
    assert set(team["probabilities"]) == {"billing", "technical", "sales"}
    assert math.isclose(sum(team["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert reply["answers"]["urgent"]["noul"] > 0.8
    # three rotations for the choice, one read for the noul, each charged in full
    assert len(reads(stub)) == 4
    assert reply["usage"]["input_tokens"] == 4 * stub.prompt_tokens


def test_a_prompt_over_n_ctx_is_refused_before_it_is_read(stub):
    stub.prompt_tokens = 200
    with pytest.raises(ContextOverflow, match="200 tokens"):
        endpoint(stub, n_ctx=199)._probs(MESSAGES, ["A", "B"], 1.0)
    assert reads(stub) == []
    endpoint(stub, n_ctx=200)._probs(MESSAGES, ["A", "B"], 1.0)
    assert len(reads(stub)) == 1


def test_priority_is_sent_only_when_set(stub):
    endpoint(stub, priority=-1)._probs(MESSAGES, ["A", "B"], 1.0)
    endpoint(stub)._probs(MESSAGES, ["A", "B"], 1.0)
    first, second = reads(stub)
    assert first["priority"] == -1 and "priority" not in second


PNG = "data:image/png;base64,iVBORw0KGgo="
JPEG = "data:image/jpeg;base64,/9j/4AAQ"
IMAGE_REQUEST = b'''{"state": "Sent with a screenshot.", "images": ["%s", "%s"],
    "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}}}''' % (PNG.encode(), JPEG.encode())


def test_images_are_refused_unless_the_server_takes_them():
    with pytest.raises(BadRequest, match="at most 0"):
        parse_request(IMAGE_REQUEST)
    with pytest.raises(BadRequest, match="at most 1"):
        parse_request(IMAGE_REQUEST, max_images=1)
    assert parse_request(IMAGE_REQUEST, max_images=2).images == (PNG, JPEG)


@pytest.mark.parametrize("images, error", [
    ('"a.png"', "list of data URLs"),
    ('["https://example.com/a.png"]', "must be a data URL"),   # a link: the engine would fetch it
    ('["http://169.254.169.254/latest/meta-data/"]', "must be a data URL"),
    ('["file:///etc/passwd"]', "must be a data URL"),
    ('["data:image/svg+xml;base64,PHN2Zz4="]', "must be a data URL"),
    ('["data:image/png,iVBORw0KGgo="]', "must be a data URL"),   # not base64
    ('["data:image/png;base64,not base64!"]', "not valid base64"),
    ('["data:image/png;base64,"]', "has no data"),
    ('[1]', "must be a data URL"),
])
def test_images_must_be_base64_data_urls(images, error):
    body = b'{"state": "s", "images": %s, "questions": {"q": {"type": "noul", "instructions": "Q?"}}}'
    with pytest.raises(BadRequest, match=error):
        parse_request(body % images.encode(), max_images=4)


def test_each_copy_of_the_state_shows_every_image_after_its_label():
    body, _, _ = user_message("the text", {"type": "noul", "instructions": "Urgent?"}, repeat=2,
                              question_first=True, images=("u1", "u2"))
    kinds = [p["type"] if p["type"] == "text" else p["image_url"]["url"] for p in body]
    assert kinds == ["text", "u1", "u2", "text", "u1", "u2", "text"]
    assert body[0]["text"].endswith("State:\n") and body[3]["text"].startswith("the text")
    text_only, _, _ = user_message("the text", {"type": "noul", "instructions": "Urgent?"}, repeat=2,
                                   question_first=True)
    assert "".join(p["text"] for p in body if p["type"] == "text") == text_only


def test_a_request_with_images_reaches_vllm_as_content_parts(stub):
    method = Method(repeat=2)
    reply = respond(endpoint(stub, method), method, parse_request(IMAGE_REQUEST, max_images=2))
    assert "noul" in reply["answers"]["urgent"]
    body, = reads(stub)
    urls = [p["image_url"]["url"] for p in body["messages"][-1]["content"] if p["type"] == "image_url"]
    assert urls == [PNG, JPEG] * 2
