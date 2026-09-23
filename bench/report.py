"""Compare local answers with Jev's, and both with the gold answers of a case set.

    python report.py CASES JEV.json LOCAL.json

A decision is the argmax for a choice, noul > 0.5 for a noul, and the
expected level of a score, rounded half up. P(gold) is the probability the answer
puts on the gold answer. Brier is the squared error of the whole
distribution against the gold answer, and log loss is -ln P(gold), with
P(gold) floored at 0.01 so that one certain miss does not dominate the mean.
"Sure+wrong" counts wrong decisions made with a top probability of 0.9 or
more. Distance to Jev is total variation for a choice or a score and
|difference| for a noul.
"""

import importlib
import json
import math
import statistics
import sys


def decision(ans: dict):
    match ans["type"]:
        case "choice":
            return ans["choice"]
        case "noul":
            return ans["noul"] > 0.5
        case "score":
            return math.floor(ans["score"] + 0.5)  # halves go up; round() takes them to even


def distribution(ans: dict) -> dict:
    if ans["type"] == "noul":
        return {True: ans["noul"], False: 1 - ans["noul"]}
    return {k if ans["type"] == "choice" else int(k): v for k, v in ans["probabilities"].items()}


def distance(a: dict, b: dict) -> float:
    da, db = distribution(a), distribution(b)
    return 0.5 * sum(abs(da.get(k, 0) - db.get(k, 0)) for k in da.keys() | db.keys())


def brier(ans: dict, gold) -> float:
    return sum((p - (k == gold)) ** 2 for k, p in distribution(ans).items())


def summarize(name: str, answers: dict, jev: dict, cases: list[dict]) -> dict:
    rows = [(c, answers[c["id"]], jev[c["id"]]) for c in cases]
    graded = [(c, a) for c, a, _ in rows if c["gold"] is not None]

    def acc(pred) -> str:
        sel = [(c, a) for c, a in graded if pred(c)]
        return f"{sum(decision(a) == c['gold'] for c, a in sel)}/{len(sel)}"

    def p_gold(c, a) -> float:
        return distribution(a).get(c["gold"], 0)

    return {
        "model": name,
        "all": acc(lambda c: True),
        "choice": acc(lambda c: c["question"]["type"] == "choice"),
        "noul": acc(lambda c: c["question"]["type"] == "noul"),
        "score": acc(lambda c: c["question"]["type"] == "score"),
        "P(gold)": statistics.mean(p_gold(c, a) for c, a in graded),
        "brier": statistics.mean(brier(a, c["gold"]) for c, a in graded),
        "logloss": statistics.mean(-math.log(max(p_gold(c, a), 0.01)) for c, a in graded),
        "sure+wrong": sum(decision(a) != c["gold"] and max(distribution(a).values()) >= 0.9 for c, a in graded),
        "agree": f"{sum(decision(a) == decision(j) for _, a, j in rows)}/{len(rows)}",
        "dist": statistics.mean(distance(a, j) for _, a, j in rows),
        "ms": statistics.median(a["latency_ms"] for _, a, _ in rows),
    }


def short(name: str) -> str:
    return name.split("-it")[0].split("-Q")[0].split("_q")[0].removeprefix("google_")


def main(case_set: str, jev_path: str, local_path: str) -> None:
    cases = importlib.import_module(case_set).CASES
    jev = json.load(open(jev_path))
    local = json.load(open(local_path))
    table = [summarize("jev", jev, jev, cases)] + [summarize(short(n), a, jev, cases) for n, a in local.items()]
    cols = list(table[0])
    print(" | ".join(cols))
    for row in table:
        print(" | ".join(f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]) for c in cols))

    tags = sorted({t for c in cases for t in c["tag"].split()})
    everyone = [("jev", jev)] + [(short(n), a) for n, a in local.items()]
    print("\nCorrect by tag:")
    print("tag | " + " | ".join(n for n, _ in everyone))
    for tag in tags:
        sel = [c for c in cases if tag in c["tag"].split() and c["gold"] is not None]
        print(f"{tag} ({len(sel)}) | " + " | ".join(
            str(sum(decision(a[c["id"]]) == c["gold"] for c in sel)) for _, a in everyone))

    print("\nWrong answers, as decision (P(gold)):")
    for c in cases:
        if c["gold"] is None:
            continue
        misses = [(n, a[c["id"]]) for n, a in everyone if decision(a[c["id"]]) != c["gold"]]
        if misses:
            print(f"  {c['id']:20} gold={c['gold']}: " + "; ".join(
                f"{n}={decision(a)} ({distribution(a).get(c['gold'], 0):.2f})" for n, a in misses))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
