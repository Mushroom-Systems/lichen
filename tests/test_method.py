"""The arithmetic that turns label probabilities into an answer, read from a backend that returns set values."""
import argparse

import numpy
import pytest

from lichen.method import (Method, answer, answer_keys, combine, confidence, disagreement, method_arguments,
                           method_from, probabilities, readings)

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "b", "technical": "t", "sales": "s"}}


class Fixed:
    """A backend whose every prompt reads `read(keys)`, one probability for each label, given
    the key each label stands for in that prompt."""

    def __init__(self, read):
        self.read = read

    def label_probs(self, asked, method):
        return [numpy.asarray(self.read(answer_keys(v["question"])), dtype=float) for v in asked], 0


def ask(read, question=TEAM, **options):
    p, keys, _, parts = probabilities(Fixed(read), {"state": "s", "question": question}, Method(**options))
    return dict(zip(keys, p)), parts


# A fibered list of the three teams: the first block in the question's order, the second
# rotated by one, so the labels stand for billing, technical, sales, technical, sales, billing.
# The first block puts more on its first place than the second block does.
FIBERED = [0.30, 0.10, 0.10, 0.20, 0.20, 0.10]


def test_a_fibered_prompt_is_one_reading_per_block():
    keys = ["billing", "technical", "sales", "technical", "sales", "billing"]
    assert readings(numpy.asarray(FIBERED), keys) == [
        {"billing": 0.30, "technical": 0.10, "sales": 0.10},
        {"technical": 0.20, "sales": 0.20, "billing": 0.10}]


def test_the_blocks_of_one_prompt_add_up():
    parts = [{"billing": 0.30, "technical": 0.10, "sales": 0.10},
             {"technical": 0.20, "sales": 0.20, "billing": 0.10}]
    assert combine(parts, 1, ["billing", "technical", "sales"]) == pytest.approx([0.40, 0.30, 0.30])


def test_separate_prompts_average():
    parts = [{"a": 0.9, "b": 0.1}, {"a": 0.5, "b": 0.5}]
    assert combine(parts, 2, ["a", "b"]) == pytest.approx([0.7, 0.3])


def test_disagreement_is_the_mean_distance_between_renormalized_readings():
    # Renormalized: [0.6, 0.2, 0.2] and [0.2, 0.4, 0.4] over billing, sales, technical.
    parts = [{"billing": 0.30, "technical": 0.10, "sales": 0.10},
             {"technical": 0.20, "sales": 0.20, "billing": 0.10}]
    assert disagreement(parts) == pytest.approx(0.4)
    assert disagreement([{"a": 0.3, "b": 0.1}, {"b": 0.1, "a": 0.3}]) == 0.0
    assert disagreement([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]) == 1.0
    assert disagreement([{"a": 0.9, "b": 0.1}]) == 0.0


def test_fibers_add_up_and_shrink_moves_the_answer_toward_uniform_by_the_disagreement():
    plain, _ = ask(lambda keys: FIBERED, fibers=2, fiber_map=True)
    assert [plain[k] for k in ("billing", "technical", "sales")] == pytest.approx([0.40, 0.30, 0.30])
    shrunk, _ = ask(lambda keys: FIBERED, fibers=2, fiber_map=True, shrink=True)
    # (1 - 0.4) * [0.4, 0.3, 0.3] + 0.4 / 3
    expected = [0.6 * 0.4 + 0.4 / 3, 0.6 * 0.3 + 0.4 / 3, 0.6 * 0.3 + 0.4 / 3]
    assert [shrunk[k] for k in ("billing", "technical", "sales")] == pytest.approx(expected)
    assert max(shrunk, key=shrunk.get) == max(plain, key=plain.get)


def test_rotations_cancel_a_preference_for_the_first_place():
    # Every prompt gives 0.5 to its first option, whatever it is; over the three rotations each
    # team is first once, second once and third once.
    p, parts = ask(lambda keys: [0.5, 0.3, 0.2], permute=True)
    assert len(parts) == 3
    assert list(p.values()) == pytest.approx([1 / 3] * 3)


def test_choice_confidence_is_the_documented_formula_clamped():
    assert confidence(numpy.asarray([0.6, 0.3, 0.1])) == pytest.approx((3 * 0.6 - 1) / 2)
    assert confidence(numpy.asarray([1 / 3] * 3)) == pytest.approx(0.0)
    assert confidence(numpy.asarray([1.0, 0.0, 0.0])) == 1.0


def test_each_question_type_gets_its_answer():
    choice = answer(Fixed(lambda keys: [0.6, 0.3, 0.1]), {"state": "s", "question": TEAM})
    assert (choice["choice"], choice["confidence"]) == ("billing", pytest.approx(0.4))
    noul = answer(Fixed(lambda keys: [0.8, 0.2]), {"state": "s", "question": {"type": "noul", "instructions": "Q?"}})
    assert noul["noul"] == 0.8
    levels = {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}
    score = answer(Fixed(lambda keys: [0.1, 0.2, 0.7]), {"state": "s", "question": levels})
    assert score["score"] == pytest.approx(0 * 0.1 + 1 * 0.2 + 2 * 0.7)
    assert score["confidence"] == pytest.approx((3 * 0.7 - 1) / 2)


def parse(flags):
    ap = argparse.ArgumentParser()
    method_arguments(ap)
    return ap.parse_args(flags.split())


def test_each_option_reaches_its_field():
    m = method_from(parse("--permute --repeat 3 --options-once --question-first --compact-json --rotate-last "
                          "--batch --embedding --fibers 2 --fiber-same --fiber-map --shrink --runoff 0.3 "
                          "--temperature 1.5"), guard=False)
    assert m == Method(permute=True, repeat=3, options_once=True, question_first=True, compact_json=True,
                       rotate_last=True, batch=True, embedding=True, fibers=2, fiber_same=True, fiber_map=True,
                       shrink=True, runoff=0.3, temperature=1.5)
    assert method_from(parse("--recheck"), guard=False).recheck
    assert "do not follow them" in method_from(parse(""), guard=True).system


def test_recheck_is_refused_with_batch():
    with pytest.raises(SystemExit):
        method_from(parse("--recheck --batch"), guard=False)
