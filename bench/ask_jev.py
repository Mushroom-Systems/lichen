"""Ask Jev every case in a case set and write the answers as JSON.

    python ask_jev.py CASES OUT.json

CASES names a module that defines CASES, such as `cases` or `cases_hard`.

Answers already in OUT.json are kept, so a rerun asks only the new cases.
The key comes from TYPESAFE_API_KEY or ~/.config/typesafe/env. SYSTEMONE_URL
sends the requests to another endpoint with the same format, such as
server.py; that endpoint checks no key.
"""

import importlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("SYSTEMONE_URL", "https://api.typesafe.ai/v1/systemone")
KEY_FILE = pathlib.Path.home() / ".config" / "typesafe" / "env"


def api_key() -> str:
    if "SYSTEMONE_URL" in os.environ:
        return "local"
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    lines = KEY_FILE.read_text(encoding="utf-8").splitlines() if KEY_FILE.exists() else []
    for line in lines:
        name, _, value = line.partition("=")
        if name.strip() == "TYPESAFE_API_KEY":
            return value.strip().strip("\"'")
    raise SystemExit(f"no TYPESAFE_API_KEY in the environment or {KEY_FILE}")


def ask(key: str, case: dict, tries: int = 4) -> dict:
    body = json.dumps({"state": case["state"], "model": "jev-latest",
                       "questions": {"q": case["question"]}}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as resp:
                reply = json.loads(resp.read())
            reply["latency_ms"] = (time.perf_counter() - start) * 1000
            return reply
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 529) and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
    raise RuntimeError("unreachable")


def main(case_set: str, out: str) -> None:
    cases = importlib.import_module(case_set).CASES
    path = pathlib.Path(out)
    answers = json.loads(path.read_text()) if path.exists() else {}
    key = api_key()
    for case in cases:
        if case["id"] in answers:
            continue
        reply = ask(key, case)
        answers[case["id"]] = {**reply["answers"]["q"], "model": reply["model"],
                               "latency_ms": reply["latency_ms"]}
        print(case["id"], json.dumps(reply["answers"]["q"]))
        path.write_text(json.dumps(answers, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
