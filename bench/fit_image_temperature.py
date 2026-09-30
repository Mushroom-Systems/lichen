"""Mean NLL, Brier score and accuracy of Lichen's image answers per set, over a grid of temperatures.

    python bench/fit_image_temperature.py OUT.jsonl --vllm-endpoint URL --vllm-model NAME [METHOD OPTIONS]

Asks every image of run_images.py's sets commons, drawn and signs (the six-option sign
question) through lichen.method at each temperature of --grid, with the method options as
given, and writes one JSON line per set and temperature. The likelihood of an answer is
the probability it gives the right one, as in fit_temperature.py.

Prints the best temperature pooled over all images, and the result of fitting on one group
of sets and testing on the other: commons and drawn, the images chosen or made to be
clear, against the signs drawn at random.
"""

import argparse
import json
import math
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from lichen.backends.vllm import Endpoint  # noqa: E402
from lichen.method import method_arguments, method_from, probabilities  # noqa: E402

import run_images  # noqa: E402

CLEAR, RANDOM = ("commons", "drawn"), ("signs",)


def gold(item: dict) -> str:
    truth, kind = item["truth"], item["question"]["type"]
    return ("yes" if truth else "no") if kind == "noul" else str(truth)


def score(backend, method, item: dict) -> tuple[float, float, bool]:
    case = {"state": "The attached image.", "question": item["question"], "images": (item["url"],)}
    p, keys, _, _ = probabilities(backend, case, method)
    dist, right = dict(zip(keys, map(float, p))), gold(item)
    return (-math.log(max(dist[right], 1e-12)), sum((v - (k == right)) ** 2 for k, v in dist.items()),
            max(dist, key=dist.get) == right)


def pooled(rows: list[dict], t: float, sets: tuple) -> float:
    chosen = [r for r in rows if r["t"] == t and r["set"] in sets]
    return sum(r["nll"] * r["n"] for r in chosen) / sum(r["n"] for r in chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--vllm-endpoint", required=True, metavar="URL")
    ap.add_argument("--vllm-model", metavar="NAME")
    ap.add_argument("--n-ctx", type=int, default=16384)
    ap.add_argument("--grid", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.75, 3.5, 5.0])
    method_arguments(ap)
    args = ap.parse_args()
    method = method_from(args, guard=True)
    backend = Endpoint(args.vllm_endpoint, "model", method, served=args.vllm_model, n_ctx=args.n_ctx)
    sets = {"commons": run_images.commons(), "drawn": run_images.drawn(), "signs": run_images.signs(False)}

    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        for t in args.grid:
            m = replace(method, temperature=t)
            for name, items in sets.items():
                with ThreadPoolExecutor(8) as pool:
                    got = list(pool.map(lambda i: score(backend, m, i), items))
                n = len(got)
                row = {"t": t, "set": name, "n": n, "nll": sum(g[0] for g in got) / n,
                       "brier": sum(g[1] for g in got) / n, "right": sum(g[2] for g in got)}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                print(f"T {t:4}  {name:8} NLL {row['nll']:.4f}  Brier {row['brier']:.4f}  {row['right']}/{n}", flush=True)

    every = CLEAR + RANDOM
    best = min(args.grid, key=lambda t: pooled(rows, t, every))
    print(f"all images: lowest NLL at temperature {best} ({pooled(rows, best, every):.4f})")
    for fit, test in ((CLEAR, RANDOM), (RANDOM, CLEAR)):
        t = min(args.grid, key=lambda t: pooled(rows, t, fit))
        print(f"fit on {'+'.join(fit)}: temperature {t}; NLL on {'+'.join(test)} {pooled(rows, t, test):.4f}, "
              + ", ".join(f"at {g} {pooled(rows, g, test):.4f}" for g in args.grid if g != t))
    backend.close()


if __name__ == "__main__":
    main()
