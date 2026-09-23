"""Typed answers from an embedding model: cosine similarity between the question and each answer.

EmbeddingGemma embeds the state and question as a query, under its query
prompt, and each answer as a document titled with the answer's key. A softmax
over the cosine similarities, each divided by `TEMPERATURE`, gives the
probabilities; changing the temperature moves the calibration and never the
winning answer.

- choice: one document per option, "title: <key> | text: <description>".
- score:  one document per level, "title: level <i> | text: <level>".
- noul:   two documents, yes and no, from the question's criteria, or
          "Yes: <question>" and "No: <question>" when it gives none.

Queries longer than EmbeddingGemma's 2048 tokens are cut at the end.
"""

import numpy
from llama_cpp import Llama

from .method import state_part, text

TEMPERATURE = 0.05
QUERY = "task: classification | query: "
MAX_TOKENS = 2048


def load(gguf: str) -> Llama:
    return Llama(model_path=gguf, embedding=True, n_ctx=MAX_TOKENS, n_gpu_layers=-1, verbose=False)


def _embed(model: Llama, texts: list[str]) -> numpy.ndarray:
    """Unit vectors, one row per text; a text over the context is cut at the end."""
    rows = numpy.asarray(model.embed(texts, truncate=True), dtype=numpy.float64)
    return rows / numpy.linalg.norm(rows, axis=1, keepdims=True)


def documents(question: dict) -> tuple[list[str], list[str]]:
    """The answer keys and their documents."""
    criteria = question.get("criteria")
    match question["type"]:
        case "choice":
            keys = list(criteria)
            return keys, [f"title: {k} | text: {text(criteria[k]) if criteria[k] else k}" for k in keys]
        case "score":
            keys = [str(i) for i in range(len(criteria))]
            return keys, [f"title: level {i} | text: {text(level)}" for i, level in enumerate(criteria)]
        case "noul":
            q = text(question["instructions"])
            yes = text(criteria["true"]) if criteria and criteria.get("true") else f"Yes: {q}"
            no = text(criteria["false"]) if criteria and criteria.get("false") else f"No: {q}"
            return ["yes", "no"], [f"title: yes | text: {yes}", f"title: no | text: {no}"]
    raise ValueError(question["type"])


def probabilities(model: Llama, case: dict) -> tuple[numpy.ndarray, list[str], int]:
    question = case["question"]
    query = f"{QUERY}{state_part(case['state'])}\n\nQuestion: {text(question['instructions'])}"
    keys, docs = documents(question)
    vectors = _embed(model, [query] + docs)
    cos = vectors[1:] @ vectors[0]
    z = cos / TEMPERATURE
    p = numpy.exp(z - z.max())
    tokens = sum(min(len(model.tokenize(t.encode(), add_bos=True)), model.n_ctx()) for t in [query] + docs)
    return p / p.sum(), keys, tokens
