"""Compare Lichen JevBench runs with the published systems on the 231 public items.

    python jevbench_compare.py PER_TASK.json RUNS_DIR NAME [NAME ...]

PER_TASK.json is JevBench's results/v1.2/jevbench-v1.2-per-task.json, which
holds every published system's outcome and latency on each public item.
RUNS_DIR holds this project's runs as NAME.easy.jsonl, NAME.original.jsonl
and NAME.hard.jsonl, written by `jevbench run --adapter typesafe` against
lichen.server. An item counts when the harness scored it (`correct` not None).
"""

import json
import os
import statistics
import sys

PICK = ["jev-1.13.0", "semif-qwen3.5-4b", "djev", "simplejev-qwen3.6-35b-a3b", "open-jev-zefan-9b",
        "openjev-sglang", "smalljev"]
MATCH = ("winnow", "reflex", "laya", "von", "jevk5", "hopper")


def ours(runs: str, name: str) -> dict:
    out = {}
    for tier in ("easy", "original", "hard"):
        path = os.path.join(runs, f"{name}.{tier}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            if r.get("correct") is not None:
                out[r["task_id"]] = ("c" if r["correct"] else "w", r.get("latency_s"))
    return out


def main(per_task: str, runs: str, names: list[str]) -> None:
    pub = json.load(open(per_task))
    tier_of = {t["id"]: t["tier"] for t in pub["tasks"]}
    rows = [(f"Lichen {n}", ours(runs, n)) for n in names]
    systems = pub["systems"]
    keys = [k for k in PICK if k in systems]
    keys += [k for k, v in systems.items() if k not in keys and any(w in v["display"].lower() for w in MATCH)]
    rows += [(systems[k]["display"], {t: tuple(v) for t, v in systems[k]["public_tasks"].items()}) for k in keys]
    print(f"{'system':52} {'easy':>7} {'standard':>9} {'hard':>7} {'all':>8}  median s")
    for label, res in rows:
        cells = []
        for tier in ("easy", "standard", "hard"):
            ids = [t for t, tr in tier_of.items() if tr == tier and t in res and res[t][0] in "cwf"]
            cells.append(f"{sum(res[t][0] == 'c' for t in ids)}/{len(ids)}" if ids else "-")
        scored = [t for t in tier_of if t in res and res[t][0] in "cwf"]
        acc = sum(res[t][0] == "c" for t in scored) / len(scored) if scored else 0.0
        lat = [res[t][1] for t in scored if res[t][1] is not None]
        med = statistics.median(lat) if lat else float("nan")
        print(f"{label[:52]:52} {cells[0]:>7} {cells[1]:>9} {cells[2]:>7} {acc:8.3f}  {med:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
