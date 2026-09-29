"""The engines Lichen can read a next-token distribution from.

`method` writes the prompts and reads the answer out of the label probabilities; a
backend is the part in between. What `method.answer` and the server ask of one:

    name                      the model name a reply carries
    where                     a phrase about the engine, for the startup line
    threaded                  whether requests may be served concurrently
    close()                   give back whatever the engine holds
    label_probs(asked,        one array of label probabilities per variant, in
                method)       order, and the prompt tokens they cost

An embedding model reads an answer another way, and offers `similarities(case)`
instead; `method.embedding` says which to expect.

`backend_from` is the one place a backend module is named, and it imports only the
one asked for. That is not tidiness: `import llama_cpp` loads libcuda, which a
vLLM-only install has not got, and the vLLM backend needs no Python package at all,
so neither install can afford to import the other's module.
"""


def backend_from(args, method):
    """The backend the command line asks for, ready to answer."""
    if args.vllm_endpoint:
        from .vllm import Endpoint
        return Endpoint(args.vllm_endpoint, args.model, method, args.top_logprobs,
                        args.vllm_workers, args.vllm_model, n_ctx=args.n_ctx, priority=args.vllm_priority)
    try:
        from .llamacpp import backend
    except ModuleNotFoundError as exc:  # llama-cpp-python is the `llamacpp` extra
        raise SystemExit(f"{exc}: a GGUF needs llama-cpp-python (pip install 'lichen[llamacpp]')")
    return backend(args.model, args.n_ctx, method, args.n_ubatch, args.kv)
