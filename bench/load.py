"""Throughput and latency of a Lichen server with several requests at once.

    python load.py JEVBENCH_DIR [--url URL] [--concurrency 1 4 8 16]

Sends the 231 public JevBench items (datasets/public/{easy,original,hard}.jsonl in a JevBench
checkout), one question per request, at each concurrency in turn, and prints requests per
second and the latency of a request at that load. The answers are not scored; run_images.py
and JevBench's own harness do that. A GGUF server answers one request at a time, so only a
server with --vllm-endpoint gains from concurrency.
"""

import argparse
import json
import pathlib
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url: str, body: dict) -> float:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jevbench", type=pathlib.Path)
    ap.add_argument("--url", default="http://localhost:8765/v1/systemone")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    args = ap.parse_args()
    bodies = [{"state": t["state"], "model": "lichen", "questions": {"q": t["question"]}}
              for tier in ("easy", "original", "hard")
              for t in map(json.loads, open(args.jevbench / "datasets" / "public" / f"{tier}.jsonl"))]
    print("at once  requests/s  p50 ms  p95 ms  max ms")
    for n in args.concurrency:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            seconds = sorted(pool.map(lambda b: post(args.url, b), bodies))
        wall = time.perf_counter() - start
        ms = [s * 1000 for s in seconds]
        print(f"{n:7}  {len(bodies) / wall:10.1f}  {statistics.median(ms):6.0f}  "
              f"{ms[int(0.95 * (len(ms) - 1))]:6.0f}  {ms[-1]:6.0f}")


if __name__ == "__main__":
    main()
