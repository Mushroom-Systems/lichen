"""llama_cpp stays inside the llama.cpp backend.

Importing llama_cpp loads libcuda, which a vLLM-only install has not got, so only the two
backends that run on llama.cpp (llamacpp.py, and embed.py, which it loads) may import it, and
nothing else may import them except through backends.backend_from.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_only_the_llamacpp_backends_import_llama_cpp():
    importing = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "lichen").rglob("*.py")
                       if re.search(r"^\s*(import|from) llama_cpp\b", p.read_text(), re.M))
    assert importing == ["lichen/backends/embed.py", "lichen/backends/llamacpp.py"]


def test_the_server_and_the_vllm_backend_import_without_llama_cpp():
    # A fresh interpreter in which `import llama_cpp` fails, as it does without the extra.
    code = ("import sys; sys.modules['llama_cpp'] = None; "
            "import lichen.server, lichen.method, lichen.backends.vllm")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
