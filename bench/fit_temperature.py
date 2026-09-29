"""Fit --temperature for a model: the value that gives the gold answers of the case sets the
lowest negative log-likelihood.

    python bench/fit_temperature.py --vllm-endpoint URL --vllm-model NAME [METHOD OPTIONS]
    python bench/fit_temperature.py --model MODEL.gguf [METHOD OPTIONS]

Every case of bench/cases.py and bench/cases_hard.py (98, all with a gold answer) is answered
through lichen.method at each temperature of --grid, with the other method options as given,
so fibers, the letter map and shrink act as they do when served. The likelihood of a case is
the probability its answer gives the gold one: the gold option of a choice or a score, P(Yes)
or 1 - P(Yes) for a noul. A temperature changes no answer, only how sure each one is.

Prints mean NLL, Brier score and accuracy per temperature, and the best temperature.
"""

import argparse
import math
import pathlib
import sys
from dataclasses import replace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from lichen.backends import backend_from  # noqa: E402
from lichen.method import method_arguments, method_from, probabilities  # noqa: E402

import cases  # noqa: E402
import cases_hard  # noqa: E402


def gold_key(case: dict) -> str:
    """The answer key of the gold answer: an option, "yes" or "no", or a level's digit."""
    gold, kind = case["gold"], case["question"]["type"]
    return ("yes" if gold else "no") if kind == "noul" else str(gold) if kind == "score" else gold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model", help="a GGUF, or with --vllm-endpoint only a name")
    ap.add_argument("--vllm-endpoint", metavar="URL")
    ap.add_argument("--vllm-model", metavar="NAME")
    ap.add_argument("--n-ctx", type=int, default=16384)
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[1.0, 1.1, 1.25, 1.4, 1.5, 1.6, 1.75, 2.0, 2.25, 2.5, 3.0])
    method_arguments(ap)
    args = ap.parse_args()
    args.top_logprobs, args.vllm_workers, args.vllm_priority = 64, 8, 0
    method = method_from(args, guard=True)
    backend = backend_from(args, method)
    every = [c for c in cases.CASES + cases_hard.CASES if c.get("gold") is not None]

    print(f"{len(every)} cases, {backend.name}")
    print("temperature  NLL     Brier   right")
    best = None
    for t in args.grid:
        m = replace(method, temperature=t)
        nll = brier = right = 0.0
        for case in every:
            # The unrounded distribution: a reply rounds to 4 places, and a confident wrong
            # answer would read as probability 0.
            p, keys, _, _ = probabilities(backend, {"state": case["state"], "question": case["question"]}, m)
            dist, gold = dict(zip(keys, map(float, p))), gold_key(case)
            nll -= math.log(dist[gold])
            brier += sum((v - (k == gold)) ** 2 for k, v in dist.items())
            right += max(dist, key=dist.get) == gold
        n = len(every)
        print(f"{t:11.2f}  {nll / n:.4f}  {brier / n:.4f}  {int(right)}/{n}", flush=True)
        if best is None or nll < best[1]:
            best = (t, nll)
    print(f"lowest NLL at temperature {best[0]}")
    backend.close()


if __name__ == "__main__":
    main()
