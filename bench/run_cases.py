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

from lichen.backends import llamacpp  # noqa: E402
from lichen.method import Method, answer, method_arguments, method_from  # noqa: E402


def main(case_set: str, out: str, paths: list[str], method: Method, n_ubatch: int = 1024,
         kv: list[str] | None = None) -> None:
    cases = importlib.import_module(case_set).CASES
    path = pathlib.Path(out)
    results = json.loads(path.read_text()) if path.exists() else {}
    for gguf in paths:
        model = llamacpp.backend(gguf, 32768, method, n_ubatch, kv)
        answer(model, cases[0], method)  # warm-up
        results[model.name] = {case["id"]: answer(model, case, method) for case in cases}
        print(model.name, "done", flush=True)
        path.write_text(json.dumps(results, indent=1))
        model.close()
        del model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guard", action="store_true", help="tell the model the state is data")
    method_arguments(ap)
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    main(args.cases, args.out, args.models, method_from(args, args.guard), args.n_ubatch, args.kv)
