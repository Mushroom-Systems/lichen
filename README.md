# Lichen

Lichen is a drop-in, API-compatible replacement for Jev, TypeSafe's System One
model, that runs open-weight models on local hardware. It serves the same
`/v1/systemone` endpoint and returns the same typed answers (choice, noul and
score) with a probability for every option, so existing Jev clients work
against it without changes.

On the 231 public decisions of [JevBench](https://github.com/fstandhartinger/jevbench),
Lichen with Google's gemma-4-26B-A4B answers 89.6% correctly, against 86.6% for
Jev 1.13, at a median latency of 147 ms per decision on a single laptop GPU;
Jev's hosted API takes 665 ms.

## How it works

Lichen never generates text. Each question becomes one chat prompt that lists
the possible answers as labels (letters for a choice, Yes and No for a yes/no
question, digits for a score), and the answer is the probability the model
gives each label as its next token. One forward pass yields the model's own
distribution over the answers, with nothing to parse.

```mermaid
flowchart LR
    R["request<br/>state + questions"] --> P["chat prompt<br/>state, question, labelled answers<br/>written twice"]
    P --> V["one prompt per rotation<br/>of the option order"]
    V --> E["shared prefix evaluated once,<br/>the rest in one batch"]
    E --> L["next-token probabilities<br/>of the answer labels"]
    L --> A["averaged over rotations"]
    A --> O["reply<br/>choice, yes/no or score<br/>+ confidence"]
```

Two changes to the prompt make the answers more accurate, and a third change,
to how the prompts are evaluated, keeps them fast.

A causal model reads the prompt left to right, so it takes in the state before
it knows what will be asked about it. Lichen writes the state and the question
twice; the second copy is read with the question already in view. This is
prompt repetition, from Leviathan et al., "Prompt Repetition Improves
Non-Reasoning LLMs" (arXiv 2512.14982).

Small models also favour an answer for its place in the list, so Lichen asks a
choice once for each rotation of the option order and averages the results.

The rotations share most of their prompt, so Lichen evaluates the shared part
once, runs the differing endings together in one batch, and keeps the prefix
in the cache for the next request.

A line in the system prompt also tells the model that the state is data, and
that it must not follow instructions written inside it.

## Results

Public items of JevBench v1.4, scored by JevBench's own harness:

| System | Easy | Standard | Hard | Accuracy | Median latency |
|---|---|---|---|---|---|
| Lichen, gemma-4-26B-A4B QAT | 48/48 | 71/72 | 88/111 | 0.896 | 147 ms |
| Lichen, Qwen3.6-35B-A3B | 48/48 | 71/72 | 83/111 | 0.874 | 468 ms |
| Lichen, Qwen3.5-9B | 48/48 | 71/72 | 82/111 | 0.870 | 195 ms |
| Jev 1.13.0 (TypeSafe, hosted) | 48/48 | 71/72 | 81/111 | 0.866 | 665 ms |
| Lichen, gemma-4-E4B QAT | 48/48 | 69/72 | 62/111 | 0.775 | 69 ms |

JevBench's official score also uses held-out items and weighs calibration,
speed and cost, so this table is the public part only. [docs/RESULTS.md](docs/RESULTS.md)
has the published systems for comparison, the other models tried, a Doom
benchmark, the speed measurements and the method in detail; the runs are in
`results/jevbench/`.

## Run it

This needs Docker with the NVIDIA container toolkit. The results above came from
an RTX 5090 Laptop GPU (24 GB).

Download the default model, Google's QAT build of gemma-4-26B-A4B (Apache 2.0,
about 14 GB). It is a mixture-of-experts model, with about 4B parameters active
per token:

```
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf --local-dir models
```

Build the image. The default targets every architecture from Ampere through
Blackwell; setting `CUDA_ARCHITECTURES` to the target GPU alone builds much
faster (`120` for RTX 50-series, `89` for RTX 40-series, `86` for RTX 30-series):

```
docker build -t lichen --build-arg CUDA_ARCHITECTURES=120 .
```

Start the server. The image enables repetition, rotation and batching, with a
16k context:

```
docker run --rm --gpus all -p 8765:8765 -v "$PWD/models:/models:ro" \
    lichen --model /models/gemma-4-26B_q4_0-it.gguf
```

Send a request:

```
curl -s localhost:8765/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "lichen",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments, invoicing, refunds",
                          "technical": "Bugs, outages, integrations",
                          "sales": "Pricing, upgrades, new accounts"}},
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
  }
}'
```

```json
{
  "model": "gemma-4-26B_q4_0-it",
  "answers": {
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.9854, "technical": 0.0145, "sales": 0.0002},
             "confidence": 0.9781},
    "urgent": {"type": "noul", "noul": 1.0}
  },
  "usage": {"input_tokens": 462, "output_tokens": 0}
}
```

Existing TypeSafe clients work once their endpoint is set to
`http://localhost:8765`. The server does not check API keys and is meant for a
trusted network. `GET /health`
returns the model's name once it has loaded.

## Other models

Any instruction-tuned GGUF that llama.cpp loads can be served. gemma-4-E4B
(`google/gemma-4-E4B-it-qat-q4_0-gguf`, 5.2 GB) is the fast choice at 69 ms,
with `--compact-json --rotate-last` added. Qwen3.5-9B scores 0.870. Qwen3.6-35B-A3B
(`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) crashes llama.cpp when its rotations are
batched, so it needs the server without `--batch`:

```
docker run --rm --gpus all -p 8765:8765 -v "$PWD/models:/models:ro" \
    --entrypoint python3 lichen -m lichen.server \
    --model /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --repeat 2 --permute --n-ctx 16384
```

## Options

Arguments after `lichen` in `docker run` are added to the image's defaults.
Run outside the image, `python -m lichen.server` or `lichen-server` starts
with one copy, no rotation and no batching; pass
`--repeat 2 --permute --batch --n-ctx 16384` for the measured configuration.
`docker run --rm --gpus all lichen --help` lists all of them.

| Option | Effect |
|---|---|
| `--model PATH` | The GGUF file to serve. Required. |
| `--n-ctx N` | Context in tokens. The image uses 16384. |
| `--n-ubatch N` | Tokens per GPU pass, 1024 by default. 2048 makes long prompts 7-9% faster and cost 2 of the 111 hard JevBench items. |
| `--repeat N` | Copies of the state and question in the prompt. The image uses 2. |
| `--compact-json`, `--rotate-last` | JSON without spaces, and rotating only the last copy's options. Together they made gemma-4-E4B about 36% faster. |
| `--kv KEY=VALUE` | Override model metadata at load, such as `gemma4.expert_used_count=6`. |
| `--no-guard` | Leave out the system-prompt line about instructions inside the state. |
| `--embedding` | Serve an embedding model, answering by cosine similarity. |
| `--port N`, `--host H` | Where to listen; 8765 on 0.0.0.0 by default. |

## Reproducing the JevBench numbers

With the server running, from a checkout of JevBench:

```
for tier in easy original hard; do
  python3 -m jevbench.cli run --tasks datasets/public/$tier.jsonl --adapter typesafe \
    --endpoint http://localhost:8765 --key-env "" --price-in-per-m 0 --price-out-per-m 0 \
    --results lichen.$tier.jsonl --ledger lichen.ledger.jsonl --raw-dir lichen-raw
done
```

`bench/jevbench_compare.py` puts such runs next to the published systems, using
JevBench's `results/v1.2/jevbench-v1.2-per-task.json`.

## Limits

The server answers one request at a time, and the questions in a request one
after another. A choice takes 2 to 26 options, one letter each, and a score 2
to 10 levels, one digit each. Jev does not publish how it computes a score's
confidence, so Lichen uses the choice formula for both. Every measurement comes
from one GPU model.

## License

MIT; see [LICENSE](LICENSE). The models carry their own licenses; the Gemma 4
models used here are Apache 2.0.
