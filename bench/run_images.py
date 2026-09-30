"""Ask a Lichen server questions about images, and score the answers.

    python run_images.py SET OUT.jsonl [--url URL] [--workers N] [--same-as EARLIER.jsonl]

SET is one of:

  commons       125 photos, paintings, signs and charts from Wikimedia Commons, checked by eye,
                listed with their licenses and authors in images_commons.jsonl
  drawn         the 84 images of images_drawn.py
  signs         the road signs of images_signs.jsonl, drawn at random from Commons (see
                sample_signs.py), asked with six options: five signs, or "another sign, or no
                road sign". Removed images and those labeled unclear are left out.
  signs-forced  the random draw of images_signs.jsonl, asked with the five signs only

The server must run with --vllm-endpoint and --max-images 1 or more.

Each image is one request: the state "The attached image.", the image, and its task's
question. The Commons images are downloaded once into ~/.cache/lichen-images; a file whose
SHA-256 differs from the one listed is used, and its rows say so.

Writes one JSON line per image and prints accuracy and latency per task. --workers sends that
many requests at once. --same-as counts the answers identical to an earlier run of the same set.
"""

import argparse
import base64
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import web

HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path.home() / ".cache" / "lichen-images"
USER_AGENT = "lichen-image-bench/0.1 (https://github.com/Mushroom-Systems/lichen)"


def choice(instructions: str, options: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


QUESTIONS = {
    "animal": choice("Which animal is in the image?",
                     {"cat": "", "dog": "", "horse": "", "cow": "", "bird": ""}),
    "vehicle": choice("Which vehicle is in the image?",
                      {"car": "", "bicycle": "", "train": "", "boat": "", "airplane": ""}),
    "food": choice("Which dish is in the image?", {"pizza": "", "salad": "", "cake": "", "soup": ""}),
    "snow": {"type": "noul", "instructions": "Is there snow in the image?"},
    "time": choice("Was this picture taken by day or at night?", {"day": "", "night": ""}),
    "sign": choice("Which road sign is in the image?",
                   {"stop": "Stop", "one_way": "One way", "speed_limit": "Speed limit",
                    "no_entry": "No entry, or do not enter"}),
    "chart": choice("What kind of chart is this?",
                    {"bar": "Bar chart", "pie": "Pie chart", "line": "Line chart"}),
}
SIGNS = {"stop": "Stop", "yield": "Yield, or give way", "one_way": "One way", "speed_limit": "Speed limit",
         "no_entry": "No entry, or do not enter"}
QUESTIONS["sign6"] = choice("Which road sign is in the image?", SIGNS | {"other": "Another sign, or no road sign"})
QUESTIONS["sign5"] = choice("Which road sign is in the image?", SIGNS)


def data_url(data: bytes) -> str:
    kind = "png" if data.startswith(b"\x89PNG") else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


def fetch(row: dict) -> tuple[bytes, bool]:
    """The image's bytes, and whether they are the ones listed."""
    path = CACHE / f"{row['sha256']}.img"
    if not path.exists():
        data = web.read(urllib.request.Request(row["url"], headers={"User-Agent": USER_AGENT}), timeout=60)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest() == row["sha256"]


def commons() -> list[dict]:
    items = []
    for line in open(HERE / "images_commons.jsonl", encoding="utf-8"):
        row = json.loads(line)
        data, listed = fetch(row)
        truth = row["label"] == "yes" if row["task"] == "snow" else row["label"]
        items.append({"task": row["task"], "id": row["title"], "truth": truth, "url": data_url(data),
                      "question": QUESTIONS[row["task"]], "listed_bytes": listed})
    return items


def signs(forced: bool) -> list[dict]:
    items = []
    for line in open(HERE / "images_signs.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["removed"] or row["label"] == "unclear" or (forced and row["set"] != "random"):
            continue
        data, listed = fetch(row)
        task = "sign5" if forced else "sign6"
        items.append({"task": task, "id": row["title"], "truth": row["label"], "url": data_url(data),
                      "question": QUESTIONS[task], "listed_bytes": listed, "set": row["set"]})
    return items


def drawn() -> list[dict]:
    import io
    sys.path.insert(0, str(HERE))
    from images_drawn import cases
    items = []
    for n, (task, img, question, truth) in enumerate(cases(random.Random(7))):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        items.append({"task": task, "id": n, "truth": truth, "url": data_url(buf.getvalue()),
                      "question": question})
    return items


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.loads(web.read(req))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}") from exc


def judge(url: str, item: dict) -> dict:
    start = time.perf_counter()
    reply = post(url, {"state": "The attached image.", "images": [item["url"]], "questions": {"q": item["question"]}})
    ms = (time.perf_counter() - start) * 1000
    a = reply["answers"]["q"]
    if a["type"] == "noul":
        got, right = a["noul"], (a["noul"] > 0.5) == item["truth"]
    elif a["type"] == "score":
        got = a["score"]
        right = max(a["probabilities"], key=a["probabilities"].get) == str(item["truth"])
    else:
        got, right = a["choice"], a["choice"] == item["truth"]
    row = {k: v for k, v in item.items() if k not in ("url", "question")}
    return row | {"got": got, "right": right, "latency_ms": round(ms, 1),
                  "input_tokens": reply["usage"]["input_tokens"], "answer": a}


def top(answer: dict) -> float:
    """The probability of the answer given: the top option, or the side of a noul it falls on."""
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())


def spread(answer: dict) -> dict:
    return answer.get("probabilities") or {"yes": answer["noul"]}


def same_as(rows: list[dict], path: str) -> str:
    """How many answers match an earlier run exactly, and the largest difference of the rest."""
    earlier = {json.dumps([r["task"], r["id"]]): r["answer"] for r in map(json.loads, open(path, encoding="utf-8"))}
    pairs = [(spread(r["answer"]), spread(earlier[json.dumps([r["task"], r["id"]])])) for r in rows]
    diffs = [max(abs(a[k] - b[k]) for k in a) for a, b in pairs if a != b]
    return (f"same as {path}: {len(pairs) - len(diffs)}/{len(pairs)} answers identical"
            + (f"; largest probability difference {max(diffs):.4f}" if diffs else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=["commons", "drawn", "signs", "signs-forced"])
    ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8765/v1/systemone")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--same-as", metavar="EARLIER")
    args = ap.parse_args()

    items = {"commons": commons, "drawn": drawn, "signs": lambda: signs(False),
             "signs-forced": lambda: signs(True)}[args.set]()
    changed = sum(not i.get("listed_bytes", True) for i in items)
    if changed:
        print(f"{changed} downloaded images differ from the listed SHA-256", file=sys.stderr)
    start = time.perf_counter()
    rows = []
    # Each answer is written as it comes, so a run that fails part way keeps what it has.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(args.out, "w", encoding="utf-8") as f:
        for r in pool.map(lambda i: judge(args.url, i), items):
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    wall = time.perf_counter() - start

    ms = sorted(r["latency_ms"] for r in rows)
    print(f"{args.set}: {sum(r['right'] for r in rows)}/{len(rows)} right; latency p50 {statistics.median(ms):.0f} ms, "
          f"p95 {ms[int(0.95 * (len(ms) - 1))]:.0f} ms, max {ms[-1]:.0f} ms; {len(rows) / wall:.1f} requests/s "
          f"with {args.workers} at once; input tokens p50 {statistics.median(r['input_tokens'] for r in rows):.0f}")
    confident = {ok: [top(r["answer"]) for r in rows if r["right"] == ok] for ok in (True, False)}
    print("  median probability of the top answer: " + ", ".join(
        f"{'right' if ok else 'wrong'} {statistics.median(v):.3f}" for ok, v in confident.items() if v))
    # A sign question is one task; its answers are grouped by draw and label instead.
    group = (lambda r: f"{r['set']}/{r['truth']}") if args.set.startswith("signs") else (lambda r: r["task"])
    for task in dict.fromkeys(map(group, rows)):
        t = [r for r in rows if group(r) == task]
        wrong = [f"{r['truth']}->{round(r['got'], 2) if isinstance(r['got'], float) else r['got']}"
                 for r in t if not r["right"]]
        print(f"  {task:18} {sum(r['right'] for r in t):3}/{len(t):3}" + (f"  wrong: {', '.join(wrong)}" if wrong else ""))
    if args.same_as:
        print(same_as(rows, args.same_as))

if __name__ == "__main__":
    main()
