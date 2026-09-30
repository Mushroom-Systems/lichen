# Lichen

Lichen is a drop-in, API-compatible replacement for Jev, TypeSafe's System One
model, that runs open-weight models on local hardware. It serves the same
`/v1/systemone` endpoint and returns the same typed answers (choice, noul and
score) with a probability for every option, so existing Jev clients work
against it without changes. A request can also carry images as part of the
state, and the questions are then answered about the pictures.

With Google's gemma-4-26B-A4B on one 24 GB laptop GPU, Lichen answers 204 to
207 of the 231 public decisions of [JevBench](https://github.com/fstandhartinger/jevbench),
against 200 for Jev 1.13. Served from vLLM, a decision takes 56 ms at the
median. It answered all 125 questions of a hand-checked set of Wikimedia Commons
photos, and named the right sign in 92 of 98 road-sign photos drawn at random
from Commons. Asked the same with "none of these" as an option, it missed a
quarter of those signs, and was sure of most of the misses (see
[Results](#results)). The same vLLM server holds a 128k-token chat, with
thinking and images, while it judges.

## What it does

A request gives the state and asks one or more questions about it:

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "lichen",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments, invoicing, refunds",
                          "technical": "Bugs, outages, integrations",
                          "sales": "Pricing, upgrades, new accounts"}},
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
  }
}
```

Each answer comes back with a probability for every option:

```json
{
  "model": "gemma-4-26b-a4b-nvfp4",
  "answers": {
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.9269, "technical": 0.038, "sales": 0.0351},
             "confidence": 0.8904},
    "urgent": {"type": "noul", "noul": 0.9998}
  },
  "usage": {"input_tokens": 434, "output_tokens": 0}
}
```

With images, the state is the picture together with a line of text. Here are
six of the test pictures, with what was asked and what came back: four it read
well, and two it got wrong with confidence. The chart went to the chat endpoint
of the same vLLM server, with thinking on; the others are Lichen judgments.

| Picture | Request | Answer |
|---|---|---|
| <img src="docs/images/stop-sign.jpg" width="240" alt="A person in a blue dress standing next to a stop sign"> | **State:** the picture, and the text "A photo from a street survey."<br><br>**Choice:** Which road sign is in the image? (stop, one way, speed limit, no entry)<br><br>**Yes/no:** Is there a person in the image? | **stop**, 0.978<br><br><br>**yes**, 0.9999 |
| <img src="docs/images/painted-sign.jpg" width="240" alt="A no-entry sign with a stick figure painted on it, sawing the white bar"> | **State:** the picture, and the text "A photo from a street survey."<br><br>**Choice:** Which road sign is in the image? (stop, one way, speed limit, no entry) | **no entry**, 0.700; about 0.10 for each of the others. In the benchmark, with the text "The attached image.", this was the least certain of the 125 Commons answers, at 0.41. |
| <img src="docs/images/snowy-balcony.jpg" width="240" alt="A snow-covered balcony over a town in falling snow"> | **State:** the picture, and the text "A photo sent in with a weather report."<br><br>**Choice:** Was this picture taken by day or at night?<br><br>**Yes/no:** Is there snow in the image? | **day**, 0.929<br><br><br>**yes**, 0.9999 |
| <img src="docs/images/pie-chart.jpg" width="240" alt="A pie chart of South Australian passenger car builders, 1880 to 1921"> | **Chat:** the picture, and "Which builder made the largest share of these cars, and about how many cars is that out of the total? Answer in two sentences." | After 934 tokens of thinking, in 10.0 s: "SAR Islington Works made the largest share of these cars, accounting for 51% of the total. This represents approximately 86 cars out of the 168 built." The chart gives 51% of 168. |
| <img src="docs/images/one-way-overpass.jpg" width="240" alt="A road passing under a highway overpass, with a small one-way sign on a pole at the left"> | **State:** the picture, and the text "The attached image."<br><br>**Choice:** Which road sign is in the image? (stop, yield, one way, speed limit, no entry, or another sign or no road sign) | **other**, 0.974. Wrong: the small sign on the pole at the left, under the overpass, is a one-way sign. Without the "other" option it answered stop, at 0.32. |
| <img src="docs/images/freedom-sign.jpg" width="240" alt="A round red no-entry sign with the word FREEDOM painted over its white bar"> | **State:** the picture, and the text "The attached image."<br><br>**Choice:** the same six options | **other**, 0.955. Wrong: this is a no-entry sign with its bar painted over. 6 of the 7 altered no-entry signs in the test read as "other". |

Pictures, resized to 480 pixels: [Trougnouf (Benoit Brummer)](https://commons.wikimedia.org/wiki/File:A_person_in_blue_standing_next_to_a_stop_sign_in_Anseremme,_Belgium_(DSCF7416).jpg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0);
[Project-128](https://commons.wikimedia.org/wiki/File:Saw_street_sign_(14167641036).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0);
[Katherine Bowman](https://commons.wikimedia.org/wiki/File:Crystal_City_Snow_-_Daytime_Balcony_View_(4199057962).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0);
[SCHolar44](https://commons.wikimedia.org/wiki/File:Pie_chart_--_South_Australian_end-loading_passenger_car_builders.png), CC0;
[Michael Rivera](https://commons.wikimedia.org/wiki/File:WB_CR318_at_I75_overpass.jpg), CC0;
[Matt Brown](https://commons.wikimedia.org/wiki/File:Freedom_no_entry.jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0).

## How it works

Lichen never generates text. Each question becomes one chat prompt that lists
the possible answers as labels (letters for a choice, Yes and No for a yes/no
question, digits for a score), and the answer is the probability the model
gives each label as its next token. One forward pass yields the model's own
distribution over the answers, with nothing to parse. The idea comes from
Duarte O. Carmo's post [Jev in 25 lines of Python](https://www.nobodywho.ai/posts/jev-in-25-lines/)
for NobodyWho. Lichen adds the prompt changes and the confidence correction
described below, and a server that speaks the TypeSafe API.

```mermaid
flowchart LR
    R["request<br/>state + questions"] --> P["chat prompt<br/>state and question written twice;<br/>each option listed twice,<br/>then a letter map"]
    P --> E["one forward pass,<br/>shared prefix cached"]
    E --> L["next-token probabilities<br/>of the letters"]
    L --> A["each option's letters<br/>added up, at the<br/>model's temperature"]
    A --> S["moved toward uniform<br/>by how far the two<br/>copies disagree"]
    S --> O["reply<br/>choice, yes/no or score<br/>+ confidence"]
```

Two changes to the prompt aim at more accurate answers, and two changes to how
the answer is read make its confidence honest. On gemma-4-26B-A4B the prompt
changes add a few hard items; on smaller models they add more.

A causal model reads the prompt left to right, so it takes in the state before
it knows what will be asked about it. Lichen writes the state and the question
twice; the second copy is read with the question already in view. This is
prompt repetition, from Leviathan et al., "Prompt Repetition Improves
Non-Reasoning LLMs" (arXiv 2512.14982).

Small models also favour an answer for its place in the list. Lichen lists
every option twice, the second time in a rotated order, and then maps letters
to options (A -> billing, ..., F -> billing). Each option's answer is the sum
over its letters. This one prompt matches the accuracy of asking once per
rotation of the options and averaging, with half the input tokens.

The two copies of the list are two readings of the same question. On
JevBench's 139 public choices, the readings of a wrong answer disagreed about
six times as much as those of a right one. Lichen moves each answer toward
uniform in proportion to that disagreement, which lowers the confidence of
answers the model is unsure of and never changes which answer comes first.
This rule was picked over one alternative by their calibration on JevBench's
public items.

Raw label probabilities are also too sharp: the model states 99% where it is
right far less often. Lichen divides the label logits by a temperature before
the softmax, 1.25 for gemma-4-26B-A4B QAT and 2.25 for its NVFP4 build. Each
was fitted with `bench/fit_temperature.py` on the project's own 98 test
questions in `bench/`, not on JevBench, and like the correction above it
changes no answer. A temperature suits one model and one kind of question, so
another model needs its own fit.

Images belong to the state. Each copy of the state shows every image, after
`State:` and before the state's text, so the second copy's images are also read
with the question in view.

A line in the system prompt also tells the model that the state is data, and
that it must not follow instructions written inside it.

## Results

Public items of JevBench v1.4, scored by JevBench's own harness:

| System | Easy | Standard | Hard | Accuracy | Median latency |
|---|---|---|---|---|---|
| Lichen, gemma-4-26B-A4B QAT on llama.cpp | 48/48 | 71/72 | 88/111 | 0.896 | 93 ms |
| Lichen, gemma-4-26B-A4B NVFP4 on vLLM† | 48/48 | 71/72 | 85/111 | 0.883 | 56 ms |
| gemma-4-26B-A4B QAT alone (no prompt techniques) | 48/48 | 71/72 | 83/111 | 0.874 | 85 ms |
| Lichen, Qwen3.6-35B-A3B\* | 48/48 | 71/72 | 83/111 | 0.874 | 468 ms |
| Lichen, Qwen3.5-9B\* | 48/48 | 71/72 | 82/111 | 0.870 | 195 ms |
| Jev 1.13.0 (TypeSafe, hosted) | 48/48 | 71/72 | 81/111 | 0.866 | 665 ms |
| Lichen, gemma-4-E4B QAT | 48/48 | 69/72 | 64/111 | 0.784 | 59 ms |

\* Measured with the earlier configuration, which asked each choice once per
rotation of its options instead of listing them twice.

† unsloth's NVFP4 build on vLLM 0.30, served as in [Run it](#run-it), which
adds `--question-first --compact-json`. Paired with the QAT row, 9 items are
right only there and 6 only here (p = 0.61), and four vLLM configurations
scored 204 to 206 of 231. It runs at temperature 2.25, fitted for it the same
way as the QAT model's 1.25; at 1.25 its hard-tier calibration error was 0.177,
and at 2.25 it is 0.119, with the same answer on every item.

Lichen on llama.cpp and Jev disagree on 13 items, 10 of them in Lichen's
favour: likely a real edge, though not statistically significant on this many
items (p = 0.09). The model alone, without Lichen's prompt techniques, answers
202. The techniques probably add a few points on the hard tier, and they reach
that accuracy with half the input tokens of asking once per rotation of the
options.

The model alone was run through the same server with the guard line and none
of the other options. Lichen's latencies are local, one request at a time on an
otherwise idle RTX 5090 Laptop GPU (24 GB); Jev's are its hosted API, including
the network round trip, so the column compares deployments rather than models.
Items decided by a small margin can come out differently on another GPU, so a
difference of a few hard items between two rows is within that variation.

On the hard tier, gemma-4-26B-A4B QAT also has a calibration error (ECE) of
0.120 and a JevBench calibration score of 77.5; Jev's on the published board is
76.3. JevBench's official score also uses held-out items and weighs calibration,
speed and cost, so this table is the public part only.

For images, `bench/run_images.py` asks one question about each image, one
request per image, on the vLLM configuration:

| Set | Right | Median top probability, right / wrong | p50 |
|---|---|---|---|
| 125 photos, paintings, road signs and charts from Wikimedia Commons, checked by eye (`bench/images_commons.jsonl`) | 125/125 | 0.963 / none | 182 ms |
| 84 drawn images with exact answers (`bench/images_drawn.py`) | 82/84 | 0.907 / 0.443 | 167 ms |
| 98 road-sign photos drawn at random from Commons (`bench/images_signs.jsonl`); five signs to choose from | 92/98 | 0.947 / 0.631 | 189 ms |
| The same 98, with "another sign, or no road sign" as a sixth option | 73/98 | 0.905 / 0.833 | 197 ms |
| 71 photos of damaged and obscured signs, with the same six options | 65/71 | 0.805 / 0.739 | 190 ms |

All the images are CC0, public domain or CC BY; the files list each one's page,
author and license, and the script downloads them. The first set was checked by
eye and the unclear pictures left out, so it shows the method working, not
where it fails. The road signs were drawn in random order from Commons' sign
categories and labeled by their category, with only a painting and a photo
without its sign removed. The damaged and obscured signs were drawn the same
way and labeled by hand before any run (`bench/sample_signs.py` has the rules).

Made to choose, it names the right sign in 92 of 98 photos, and is unsure of
the six it misses. Given a way out, it takes it too often: 24 of its 25 misses
on the random draw are "another sign", 14 of them at 0.8 or more. They are
small or distant signs in street scenes, a sign buried in snow, no-entry signs
altered with stickers or paint, as in the last two pictures above, and three
pedestrian or no-vehicle bans that Commons files under no entry, where "another
sign" is a fair answer.

The confidence is therefore not calibrated on pictures, and it errs both ways.
On the clear pictures it is too low: every Commons answer is right, but the
median is 0.963, and plain photos of cars read 0.64. On the random signs with a
way out, it is too high. The temperature, 2.25, was fitted on text; over all
378 labeled images it is still the best single value, and fitting one on either
group makes the other worse (details in [docs/RESULTS.md](docs/RESULTS.md)).
Use an image answer's probability to compare answers; it does not give the
chance that the answer is right. The two drawn misses are dot counts, four read
as five and five as six.

[docs/RESULTS.md](docs/RESULTS.md) has the published systems for comparison,
the other models tried, a Doom benchmark, the speed measurements and the method
in detail; the runs are in `results/`.

## Run it

Both ways need Docker with the NVIDIA container toolkit, and a GPU with 24 GB
for gemma-4-26B-A4B. vLLM is faster, serves several requests at once, reads
images and can serve chat from the same GPU. llama.cpp runs Google's QAT build
of the model from a GGUF, in one container.

### With vLLM

From this directory:

```
docker compose up -d --build
```

This starts vLLM with unsloth's NVFP4 build of gemma-4-26B-A4B
(`unsloth/gemma-4-26B-A4B-it-NVFP4`), which it downloads into
`~/.cache/huggingface` on the first run, and Lichen in front of it. Lichen
starts once vLLM reports healthy, which takes a few minutes. Lichen answers on
port 8765, and vLLM answers chat on port 8000. `compose.yaml` lists the
environment variables that change the model, the ports, the context and the
temperature; a model other than the default needs its own temperature (see
[How it works](#how-it-works)).

Send the request above:

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

A request adds images in `images`, a list of up to 4, each sent inline as a
base64 data URL (`data:image/<png|jpeg|webp|gif>;base64,...`), the way a vision
model's chat API takes them. Links are refused, so vLLM downloads nothing. The
model judges the images and the `state` text together, as one state. With
[this photo](https://commons.wikimedia.org/wiki/File:A_person_in_blue_standing_next_to_a_stop_sign_in_Anseremme,_Belgium_(DSCF7416).jpg)
(Trougnouf, CC BY 4.0), its 960-pixel preview saved as `stop.jpg`:

```
curl -s localhost:8765/v1/systemone -H 'Content-Type: application/json' -d @- <<EOF
{
  "state": "A photo from a street survey.",
  "images": ["data:image/jpeg;base64,$(base64 -w0 stop.jpg)"],
  "questions": {
    "sign": {"type": "choice", "instructions": "Which road sign is in the image?",
             "criteria": {"stop": "Stop", "one_way": "One way", "speed_limit": "Speed limit",
                          "no_entry": "No entry, or do not enter"}},
    "person": {"type": "noul", "instructions": "Is there a person in the image?"}
  }
}
EOF
```

```json
{
  "model": "gemma-4-26b-a4b-nvfp4",
  "answers": {
    "sign": {"type": "choice", "choice": "stop",
             "probabilities": {"stop": 0.9776, "one_way": 0.0076, "speed_limit": 0.0074, "no_entry": 0.0073},
             "confidence": 0.9702},
    "person": {"type": "noul", "noul": 0.9999}
  },
  "usage": {"input_tokens": 1576, "output_tokens": 0}
}
```

Chat clients talk to vLLM on port 8000 in OpenAI's format, as model `gemma4`;
[Chat](#chat) has the details.

Without Docker, the same two servers start with these commands. Lichen's side
needs no GPU or llama.cpp, and `pip install .` is enough:

```
VLLM_BATCH_INVARIANT=1 vllm serve unsloth/gemma-4-26B-A4B-it-NVFP4 --served-model-name gemma4 \
    --max-model-len 131072 --gpu-memory-utilization 0.94 --enable-prefix-caching \
    --logprobs-mode processed_logprobs --max-logprobs 64 \
    --scheduling-policy priority --max-num-batched-tokens 4096 --long-prefill-token-threshold 2048 \
    --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking": true}' \
    --limit-mm-per-prompt '{"image":8,"audio":0,"video":0}'

python -m lichen.server --vllm-endpoint http://localhost:8000 --vllm-model gemma4 \
    --model gemma-4-26b-a4b-nvfp4 --n-ctx 16384 --vllm-priority -1 --max-images 4 \
    --repeat 2 --permute --fibers 2 --fiber-map --shrink --temperature 2.25 \
    --question-first --compact-json
```

### With llama.cpp

Download Google's QAT build of gemma-4-26B-A4B (Apache 2.0, about 14 GB). It is
a mixture-of-experts model, with about 4B parameters active per token:

```
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf --local-dir models
```

Build the image. The default targets every architecture from Ampere through
Blackwell; setting `CUDA_ARCHITECTURES` to the target GPU alone builds much
faster (`120` for RTX 50-series, `89` for RTX 40-series, `86` for RTX 30-series):

```
docker build -t lichen --build-arg CUDA_ARCHITECTURES=120 .
```

If `apt-get` fails with a 404 during the build, the Ubuntu mirror's index is
ahead of its packages; building again later usually works. A build that must
succeed now can list only `archive.ubuntu.com` (suites `noble noble-updates
noble-backports`) in `/etc/apt/sources.list.d/ubuntu.sources` before
`apt-get update`, since Ubuntu also publishes security fixes to `-updates`.

Start the server. The image turns on the measured configuration, with a 16k
context:

```
docker run --rm --gpus all -p 8765:8765 -v "$PWD/models:/models:ro" \
    lichen --model /models/gemma-4-26B_q4_0-it.gguf
```

The curl above works against it unchanged, apart from images, and gives:

```json
{
  "model": "gemma-4-26B_q4_0-it",
  "answers": {
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.9509, "technical": 0.0264, "sales": 0.0227},
             "confidence": 0.9264},
    "urgent": {"type": "noul", "noul": 0.9999}
  },
  "usage": {"input_tokens": 356, "output_tokens": 0}
}
```

Any instruction-tuned GGUF that llama.cpp loads can be served. gemma-4-E4B
(`google/gemma-4-E4B-it-qat-q4_0-gguf`, 5.2 GB) is the fast choice at 59 ms,
with `--compact-json --temperature 1.5` added; 1.5 is its temperature fitted on
the same test questions. Qwen3.5-9B scores 0.870 with the earlier
configuration. Qwen3.6-35B-A3B (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) crashes
llama.cpp when rotations are batched, so it was measured with the earlier
configuration and without `--batch`:

```
docker run --rm --gpus all -p 8765:8765 -v "$PWD/models:/models:ro" \
    --entrypoint python3 lichen -m lichen.server \
    --model /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --repeat 2 --permute --n-ctx 16384
```

### Either way

Existing TypeSafe clients work once their endpoint is set to
`http://localhost:8765`. The server does not check API keys and is meant for a
trusted network. `GET /health` returns the model's name once it has loaded.

## Serving from vLLM

`--vllm-endpoint URL` reads the same method from a vLLM server instead of a
local GGUF. That opens up checkpoints only vLLM serves (FP8, AWQ, NVFP4), and
the one vLLM server can also answer chat requests, with thinking, and read
images. On the 24 GB laptop GPU, the configuration of [With vLLM](#with-vllm)
serves Lichen's judgments with prompts of up to 16k tokens, and chat with up to
128k.

| Flags | Why |
|---|---|
| `VLLM_BATCH_INVARIANT=1` | Without it the same request can give probabilities up to 0.09 apart, enough to change a close answer. With it, repeated requests to one running vLLM agree bit for bit, for about 23% more latency. A restart of vLLM does not keep them: see [Limits](#limits). Some models cannot run with it (on vLLM 0.29, Qwen3.8's linear-attention layers); `--max-num-seqs 1` removes most of the variation instead. |
| `--logprobs-mode processed_logprobs`, `--max-logprobs 64` | Lichen restricts each read to the label tokens with `allowed_token_ids`, and needs the log-probabilities after that restriction. Without the mode, most choices fail with an error that names it. `--max-logprobs` must be at least options × fibers. |
| `--max-model-len 131072`, Lichen's `--n-ctx 16384` | Chat gets 128k tokens. Lichen checks each judgment's prompt as vLLM tokenizes it and refuses one over 16k with 422. |
| `--scheduling-policy priority`, Lichen's `--vllm-priority -1` | Judgments go ahead of chat requests waiting in the queue. |
| `--max-num-batched-tokens 4096 --long-prefill-token-threshold 2048` | A long chat prompt is read 2048 tokens at a time, so judgments do not wait for all of it. |
| `--reasoning-parser gemma4`, `--default-chat-template-kwargs` | Chat thinks by default, and the thinking comes back apart from the answer. Lichen's requests turn thinking off. |
| Lichen's `--temperature 2.25` | Fitted for this model with `bench/fit_temperature.py`. The QAT GGUF's 1.25 leaves the NVFP4 model overconfident. |
| `--limit-mm-per-prompt`, Lichen's `--max-images 4` | Up to 4 images a judgment. Each is shown in both copies of the state, so vLLM must allow 8. |

### Speed

On JevBench's 231 public items, one request at a time, against the QAT GGUF on
llama.cpp on the same GPU:

| Prompt tokens | Items | llama.cpp p50 / p95 | vLLM p50 / p95 |
|---|---|---|---|
| under 1,024 | 163 | 81 / 134 ms | 50 / 77 ms |
| 1,024 to 2,047 | 21 | 221 / 286 ms | 102 / 136 ms |
| 2,048 to 4,095 | 10 | 371 / 430 ms | 184 / 210 ms |
| 4,096 and more | 37 | 902 / 1,294 ms | 405 / 583 ms |
| all | 231 | 93 / 928 ms | 56 / 431 ms |

vLLM serves several requests at once, but that adds little here: the GPU is
already busy with one. `bench/load.py` sent the same 231 requests at 1, 4, 8 and
16 at a time and got 8.3, 10.3, 10.9 and 10.9 requests a second, against 4.1
for llama.cpp, which takes one at a time.

A judgment sent while vLLM reads a long chat prompt waits for its share of the
GPU. During the 27.7 s prefill of a 120,000-token chat, 24 judgments took
1.1 s at the median and 2.3 s at most. Without the two prefill flags above they
waited up to 25 s.

### Chat

Clients talk to vLLM directly (`/v1/chat/completions`, OpenAI's format). The
thinking comes back in the message's `reasoning` field (`delta.reasoning` when
streaming), not `reasoning_content`, and the answer in `content`. A request
turns thinking off with `"chat_template_kwargs": {"enable_thinking": false}`.
Thinking spends tokens before the answer, so a small `max_tokens` can end a
reply before its answer starts. Chat takes images too, sent inline as base64 in
an `image_url` content part, as the pie chart above was.

With images allowed, vLLM loads the model's vision encoder, 1.08 GiB, and the
KV cache holds 209,497 tokens, which is 1.6 chats of the full 128k at once.
Without images it held 252,174.

### Images

An image is 262 prompt tokens on this model, and counts toward `--n-ctx`. More
images than `--max-images` are refused with 422, and a GGUF server does not
start with `--max-images`.

### Other differences

- `--model` is the name vLLM serves the model under. `--vllm-model NAME` sends
  that name to vLLM instead, and `--model` is then only the name replies carry.
- `--batch` does nothing here, since vLLM caches the shared prefix itself.
  `--recheck` and `--embedding` are not available, and neither are the models
  whose prompts only the llama.cpp backend builds (Granite Guardian, Qwen3Guard,
  and chat templates that leave a reasoning block open); asking for them is an
  error.
- `usage.input_tokens` counts the whole prompt. With a GGUF it counts only the
  tokens after the cached prefix, so the two are not comparable.

## Options

Arguments after the image name, or in the `command` of the `lichen` service in
`compose.yaml`, are added to the image's defaults, and a repeated option
replaces its default. Each image starts with the configuration measured for its
backend: the llama.cpp image with `--repeat 2 --permute --batch --fibers 2
--fiber-map --shrink --temperature 1.25 --n-ctx 16384`, the vLLM image with
`--repeat 2 --permute --fibers 2 --fiber-map --shrink --temperature 2.25
--question-first --compact-json --n-ctx 16384`. Run outside an image,
`python -m lichen.server` or `lichen-server` starts with all of these off; it
needs `pip install .` for vLLM, and `pip install '.[llamacpp]'` with a CUDA
build of llama-cpp-python for a GGUF.
`--help` lists every option.

| Option | Effect |
|---|---|
| `--model PATH` | The GGUF file to serve, or with `--vllm-endpoint` the model's name. Required. |
| `--n-ctx N` | Most prompt tokens a judgment may use, refused with 422 above it; on a GGUF also the context llama.cpp allocates. 16384 in both images. |
| `--n-ubatch N` | Tokens per GPU pass on a GGUF, 1024 by default. 2048 makes long prompts 7-9% faster and cost 2 of the 111 hard JevBench items. |
| `--repeat N` | Copies of the state and question in the prompt; 2 in both images. 3 added one hard JevBench item and improved calibration, for 50% more input tokens. |
| `--options-once` | With `--repeat`, list the options once, at the end, in place of once per copy. |
| `--fibers M`, `--fiber-map` | List each option M times in one prompt, in rotated blocks, and read the answer as the sum over each option's letters; with `--fiber-map`, list the options first and then map letters to them. Both images use 2 and the map. |
| `--fiber-same` | With `--fibers`, repeat the blocks in the same order in place of rotating them. |
| `--permute` | Ask a choice once per rotation of its options and average. With `--fibers`, only a list that would pass 62 letters falls back to this; without `--permute`, such a list is asked once. |
| `--shrink` | Move each answer toward uniform by how far its readings disagree. On in both images. |
| `--temperature T` | Divide the label logits by T before the softmax: 1.25 in the llama.cpp image, fitted for gemma-4-26B-A4B QAT, and 2.25 in the vLLM image, fitted for its NVFP4 build. `bench/fit_temperature.py` fits it for another model. |
| `--runoff D` | When a choice's readings disagree by more than D, ask its two leading options alone and split their mass by that answer. Off by default: it added 3 hard JevBench items for gemma-4-E4B and cost 4 for gemma-4-26B-A4B. |
| `--question-first` | Put the question before the state in each copy. On in the vLLM image; on the QAT GGUF it cost 6 hard JevBench items. |
| `--compact-json`, `--rotate-last` | JSON without spaces, and, when rotating, rotating only the last copy's options. `--compact-json` saves little on JevBench. |
| `--batch` | On a GGUF, read a question's shared prefix once and evaluate its rotations or blocks together. On in the llama.cpp image. vLLM caches the prefix itself, so with `--vllm-endpoint` it does nothing. |
| `--recheck` | Ask a second turn that shows the first answer, and read that answer. Not with `--batch` or `--vllm-endpoint`. |
| `--trace PATH` | Append each question's separate readings to PATH as JSON lines. |
| `--kv KEY=VALUE` | Override model metadata at load, such as `gemma4.expert_used_count=6`. |
| `--no-guard` | Leave out the system-prompt line about instructions inside the state. |
| `--embedding` | Serve an embedding model, answering by cosine similarity. |
| `--vllm-endpoint URL`, `--vllm-model NAME` | Read the model from a vLLM server instead of a GGUF (see [Serving from vLLM](#serving-from-vllm)). |
| `--vllm-workers N` | With `--vllm-endpoint`, the prompts of one question sent at once; 8 by default. |
| `--top-logprobs N` | With `--vllm-endpoint`, the most labels one prompt may be read over; 64 by default. It must be at least options × fibers, and at most vLLM's `--max-logprobs`. |
| `--vllm-priority P` | vLLM scheduling priority of each judgment, lower first; needs vLLM's `--scheduling-policy priority`. |
| `--max-images N` | Most images a request may carry; 0 by default. Needs `--vllm-endpoint`, and vLLM's `--limit-mm-per-prompt` must allow N × `--repeat`. |
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

With a GGUF the server answers one request at a time; with `--vllm-endpoint`,
concurrently. Either way the questions in a request are answered one after
another. A choice takes 2 to 62 options, one label each (A-Z, a-z, 0-9), and a
score 2 to 10 levels, one digit each. Jev does not publish how it computes a
score's confidence, so Lichen uses the choice formula for both. Every
measurement comes from one GPU model.

On pictures, the probabilities are not calibrated: too low on clear pictures,
too high on small or altered signs when "none of these" is an option (see
[Results](#results)).

Each start of vLLM can give different probabilities, even with the same launch
and `VLLM_BATCH_INVARIANT=1`: over three starts, one JevBench item's top answer
read 0.85, 0.91 and 0.99. The top answers moved little: vLLM runs after separate
starts scored 204 to 206 of 231. The cause is not found. Within one running
vLLM, image answers repeat bit for bit when requests come one at a time. Judged
8 at a time, one answer in each image set moved (a dog photo from 0.88 to 0.93,
and a drawn purple shape from 0.51 to 0.70), with the same top answer: batch
invariance is not exact once images are read together, and shrink enlarges the
difference.

## License

MIT; see [LICENSE](LICENSE). The models carry their own licenses; the Gemma 4
models used here are Apache 2.0.
