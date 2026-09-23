"""Answer every case in a case set with local GGUF models, and write the answers as JSON.

    python bench/run_cases.py [--guard] [--permute] [--repeat N] [--batch] ... CASES OUT.json MODEL.gguf [...]

CASES names a module in this folder that defines CASES (`cases` or
`cases_hard`). The options are the method's (see lichen/method.py). Answers
already in OUT.json are kept; each model's answers go under its file name.
"""

import argparse
import importlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lichen.method import Method, answer, load, method_arguments, method_from, parse_overrides  # noqa: E402


def main(case_set: str, out: str, paths: list[str], method: Method, n_ubatch: int = 1024,
         overrides: dict | None = None) -> None:
    cases = importlib.import_module(case_set).CASES
    path = pathlib.Path(out)
    results = json.loads(path.read_text()) if path.exists() else {}
    for gguf in paths:
        name = pathlib.Path(gguf).stem
        model, evaluator = load(gguf, 32768, method, n_ubatch, overrides)
        answer(model, cases[0], method, name, evaluator)  # warm-up
        results[name] = {case["id"]: answer(model, case, method, name, evaluator) for case in cases}
        print(name, "done", flush=True)
        path.write_text(json.dumps(results, indent=1))
        if evaluator:
            evaluator.close()
        model.close()
        del model, evaluator


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guard", action="store_true", help="tell the model the state is data")
    method_arguments(ap)
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    main(args.cases, args.out, args.models, method_from(args, args.guard), args.n_ubatch, parse_overrides(args.kv))
