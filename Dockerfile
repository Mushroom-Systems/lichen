# Lichen: a drop-in, API-compatible replacement for Jev, TypeSafe's System One model.
#
# Two images, one for each backend. Both listen on 0.0.0.0:8765 and answer POST
# /v1/systemone. Arguments after the image name are added to the entrypoint's
# configuration, and a repeated option replaces the default.
#
# The vLLM front (target "vllm") reads the model from a vLLM server. It has no CUDA and
# no llama.cpp; compose.yaml starts it together with vLLM:
#
#   docker build -t lichen:vllm --target vllm .
#   docker run --rm -p 8765:8765 lichen:vllm --vllm-endpoint http://HOST:8000 \
#       --vllm-model gemma4 --model gemma-4-26b-a4b-nvfp4
#
# Its entrypoint turns on the configuration measured on gemma-4-26B-A4B NVFP4: repeat 2,
# each choice option listed twice with a letter map, rotation for a list past 62
# letters, confidence shrunk by disagreement, the question before the state, compact
# JSON, label temperature 2.25 and a 16k limit on a judgment's prompt.
#
# The llama.cpp image (the default target) loads a GGUF itself:
#
#   docker build -t lichen .
#   docker run --rm --gpus all -p 8765:8765 -v /path/to/models:/models:ro \
#       lichen --model /models/gemma-4-26B_q4_0-it.gguf
#
# Its entrypoint turns on the configuration measured on gemma-4-26B-A4B QAT (repeat 2,
# each choice option listed twice with a letter map, rotation for a list past 62
# letters, batching, confidence shrunk by disagreement, label temperature 1.25, a 16k
# context); see `docker run --rm --gpus all lichen --help` (the CUDA runtime needs the
# GPU even to print the help).

FROM python:3.12-slim AS vllm

RUN pip install --no-cache-dir numpy jinja2

WORKDIR /app
COPY LICENSE /app/LICENSE
COPY lichen /app/lichen

RUN useradd --system --no-create-home lichen
USER lichen

EXPOSE 8765
ENTRYPOINT ["python3", "-m", "lichen.server", "--repeat", "2", "--permute", \
            "--fibers", "2", "--fiber-map", "--shrink", "--temperature", "2.25", \
            "--question-first", "--compact-json", "--n-ctx", "16384"]
CMD ["--help"]

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS llamacpp

# CUDA architectures to build for. The default covers Ampere (80, 86),
# Ada (89), Hopper (90) and Blackwell (100, 120). One architecture, such as
# "120" for an RTX 50-series card, builds much faster.
ARG CUDA_ARCHITECTURES="80;86;89;90;100;120"
# The llama-cpp-python to build. The measured results used the 0.3.35 release;
# "git+https://github.com/abetlen/llama-cpp-python.git@main" builds a newer
# llama.cpp.
ARG LLAMA_CPP_PYTHON="llama-cpp-python==0.3.35"
# Extra CMake flags, for example -DGGML_CUDA_FORCE_CUBLAS=ON.
ARG EXTRA_CMAKE=""

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends python3 python3-venv python3-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# The build links against libcuda.so; the stub stands in for the driver, which the
# container gets from the host at run time.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
       CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES $EXTRA_CMAKE" \
       pip install --no-cache-dir "$LLAMA_CPP_PYTHON" numpy jinja2 \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

WORKDIR /app
COPY LICENSE /app/LICENSE
COPY lichen /app/lichen

RUN useradd --system --no-create-home lichen
USER lichen

EXPOSE 8765
ENTRYPOINT ["python3", "-m", "lichen.server", "--repeat", "2", "--permute", "--batch", \
            "--fibers", "2", "--fiber-map", "--shrink", "--temperature", "1.25", "--n-ctx", "16384"]
CMD ["--help"]
