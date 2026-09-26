# openjev

One-pass option scoring with a local Gemma 3 4B, on Apple silicon via MLX or on
NVIDIA/CPU via PyTorch (`--backend torch`).
Design notes: [docs/design/one-pass-option-scoring.md](docs/design/one-pass-option-scoring.md);
per-task training: [docs/design/per-task-finetuning-with-gemma.md](docs/design/per-task-finetuning-with-gemma.md).

Given a context and a list of pre-written options, the model prefills the
context once, expands that KV cache across the option batch, and scores every
option in a single padded forward pass. No decoding. The score is the
log-probability of the option tokens given the context; a softmax over the
option scores gives a probability per option, like `jevlike-predict`.

## Setup

```sh
make setup       # uv sync (arm64 Python 3.12 venv) + download google/gemma-3-4b-it into models/ (gated; needs HF login)
make serve       # start the HTTP server on :8000
make health      # curl /health
make request     # example curl against /score
make systemone   # TypeSafe-style request against /v1/systemone
make check       # verify cached batched scoring against naive re-encoding
make bench       # latency benchmark
make eval DATA=data/synthetic/test.jsonl
```

`make setup` picks the backend extra for the platform: `mlx` (plus `torch`, for the jevlike comparison
and HF cross-checks) on macOS, `torch` alone elsewhere. Installing by hand:

```sh
uv sync --extra mlx --extra torch   # Apple silicon
uv sync --extra torch               # NVIDIA or CPU
```

Both backends are optional extras, so a bare `uv sync` installs neither. `mlx` in particular must not
be installed off Apple silicon: it resolves on Linux without its `libmlx.so`, and `transformers`
finds the metadata, tries to import it, and dies -- taking the torch backend down with it.

`make` on macOS needs the Xcode licence accepted (`sudo xcodebuild -license accept`) or Homebrew's `gmake`.

## Usage

```sh
# Rank options for one context (prints probability, score, raw sum, token count)
.venv/bin/openjev score --context "The capital of France is" \
    --option " Paris" --option " Berlin" --option " Lyon"

# Chat template (context + the option list as a user turn, each option scored as the reply) + PMI normalisation
.venv/bin/openjev score --chat --norm pmi --context "..." --option "..." --option "..."

# Predefined options: one per line in a text file, reused for every context
.venv/bin/openjev score --options-file options.txt --context "..."
.venv/bin/openjev eval contexts.jsonl --fixed-options options.txt   # rows need only {"context": ...}; add "label" for accuracy

# Top-1 / top-3 accuracy on jevlike-style JSONL: {"context": ..., "options": [...], "label": 0}
.venv/bin/openjev eval data.jsonl --norm mean

# Verify the cached batched path against naive re-encoding, and benchmark it
.venv/bin/openjev check
.venv/bin/openjev bench --context-tokens 200 --options 8 --option-tokens 30
```

## Server

Loads the model once and answers scoring requests in about 90 ms each. Runs natively on macOS with
Metal, or on Linux/NVIDIA with `--backend torch` (see below); the MLX path has no container story,
because Linux containers cannot reach the Apple GPU.

```sh
make serve                                     # = .venv/bin/openjev serve --port 8000
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "context": "Customer: my order arrived broken. Agent:",
  "options": [" I am sorry, I will send a replacement.", " Please read our returns policy."],
  "norm": "mean", "chat": false, "sep": ""
}'
```

The response has `best`, `best_index`, per-option `probability` / `logprob_sum` / `n_tokens`, and `timing`.

### With LoRA adapters

`--lora NAME=DIR` (torch backend, repeatable) loads one 4-bit base model plus every adapter under its
name; float32 adapters and int8 ones from `openjev lora-quantize` both work. Training them:
[LoRA fine-tuning](docs/lora.md).

```sh
openjev serve --backend torch --quantize 4bit --model google/gemma-4-E4B-it --port 8000 \
    --lora claims=runs/lora-claims_attrib_keys-1500 \
    --lora movies=runs/lora-movies_general_v2-int8
```

`POST /v1/lora/score` picks the adapter per request; `"adapter": null` is the base model zero-shot.
`chat` must match the format the adapter was trained in: `true` (the default) puts the context and
the option list in a user turn and scores each option as the reply, `false` scores `context + sep +
option` as plain text (`sep` defaults to `"\nChoice: "`).

```sh
curl -s localhost:8000/v1/lora/score -H 'content-type: application/json' -d '{
  "context": "Which study found that bone density shrank by 11% after a flu vaccine booster?",
  "options": ["Mancini et al. (2024)", "Delacroix et al. (2022)", "Varga et al. (2016)", "Walsh et al. (2023)",
              "Iyer et al. (2009)", "Rossi et al. (2017)", "Ferreira et al. (2018)", "Petrov et al. (2008)"],
  "adapter": "claims"
}'
# -> {"best": "Mancini et al. (2024)", "best_index": 0, "adapter": "claims", "chat": true,
#     "probabilities": [1.0, ...], "logprob_sums": [...], "timing": {...}}
#    with "adapter": null the base model picks Delacroix et al. (2022), p=0.98

curl -s localhost:8000/v1/lora/score -H 'content-type: application/json' -d '{
  "context": "Tonight I want to watch something about two gunslingers settling a grudge. Pick one film.",
  "options": ["Stone River Pact", "Velvet Orbit", "Iron Tide", "Midnight Market"],
  "adapter": "movies", "chat": false
}'
```

A server started with `--lora` also answers `/score` and `/v1/systemone` in the chat format: a
question of type `choice`, `score` or `noul` uses the adapter loaded under that same name, and the
zero-shot base model when there is none. So `--lora choice=DIR` makes every choice question of
`/v1/systemone` go through that adapter.

In Python, without a server:

```python
from openjev.lora_serve import LoraEngine

engine = LoraEngine("google/gemma-4-E4B-it", {"claims": "runs/lora-claims_attrib_keys-1500"})
sums, probs = engine.score(context, options, adapter="claims")           # chat format
sums, probs = engine.score(context, options, adapter=None)               # zero-shot
```

`openjev lora-eval DIR DATA.jsonl --model ... [--chat]` scores a saved adapter on a labelled set, and
`compare.py` (below) puts several adapters, the zero-shot model and Jev side by side on the same rows.

### Released adapters (v0.2.0)

The [v0.2.0 release](https://github.com/pakkio/openjev/releases/tag/v0.2.0) ships four int8 adapters
for `google/gemma-4-E4B-it` (revision `ee0ef60`), each a ~37 MB zip with the weights, its
`openjev_adapter.json` (task, prompt format, test results) and the Gemma notice. The base model is not
included: it downloads from Hugging Face on first use, after accepting the Gemma license there. Top-1
on the 300-row test split, all three engines on the same rows:

| adapter | task | format | Jev | openjev zero-shot | openjev + adapter | size |
|---|---|---|---|---|---|---|
| `claims-keys` | claim -> which of 8 invented papers supports it, options as 'Surname et al. (year)' keys | chat | 0.117 | 0.140 | **1.000** | 37 MB |
| `movies` | viewer request -> 1 of 10 made-up film titles | plain | 0.820 | 0.670 | **1.000** | 37 MB |
| `genres` | film description -> 1 of 10 genres | chat | 0.813 | 0.810 | **0.900** | 37 MB |
| `claims-refs` | claim -> which of 8 full references (2 same-topic) supports it | chat | 0.820 | 0.790 | **0.867** | 37 MB |

```sh
gh release download v0.2.0 -R pakkio/openjev -p '*.zip' -p SHA256SUMS && sha256sum -c SHA256SUMS
for z in openjev-lora-*-int8.zip; do unzip -q "$z"; done
openjev serve --backend torch --quantize 4bit --model google/gemma-4-E4B-it \
    --lora claims-keys=./claims-keys --lora claims-refs=./claims-refs \
    --lora genres=./genres --lora movies=./movies      # movies is a plain-format adapter: call it with "chat": false
```

They were trained on synthetic data only (the generators are in this repo), so they know the made-up
papers and films of those sets, not real ones; they are examples of what a per-task adapter learns,
and templates for training your own. The adapters are Gemma model derivatives and fall under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## TypeSafe System One contract

`POST /v1/systemone` implements the request/response shape documented at
[docs.typesafe.ai](https://docs.typesafe.ai): a `state` (string, object or array) plus a map of typed
`questions`, answered against that state.

```sh
make systemone     # sends examples/systemone-quickstart.json; or:
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I ordered size 10 shoes but received size 8. Please send the right size.",
  "model": "jev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which department handles this?",
                    "criteria": {"returns": "Returns and exchanges", "shipping": "Delivery issues", "billing": "Charges and refunds"}},
    "severity":    {"type": "score",  "instructions": "How severe is the problem?",
                    "criteria": ["minor", "moderate: wrong item", "major: safety or financial loss"]},
    "wants_refund":{"type": "noul",   "instructions": "Is the customer asking for a refund to their card?"}
  }
}'
```

| question `type` | request `criteria` | answer fields |
|---|---|---|
| `choice` | map of option name to description (string, object, array or null); up to 255 options | `choice`, `probabilities` (sum to 1), `confidence` |
| `score` | ordered array of level descriptions | `score` (probability-weighted mean of level index), `probabilities` keyed `"0".."n-1"`, `legend`, `confidence` |
| `noul` | optional `{"true": ..., "false": ...}` | `noul` = probability of yes |

Response: `{"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens"}}`.
Set `OPENJEV_API_KEY` before `make serve` to require `Authorization: Bearer <key>` on this route.

How it maps onto the scorer: each question is rendered to a plain-text prompt (state, instructions,
options or levels, then `Answer:` and a newline) and the option names, level numbers, or `yes`/`no`
are scored as continuations in one prefix-shared batched pass per question. `confidence` is 1 minus
the normalised entropy of the distribution; TypeSafe does not publish its formula, so treat it as an
approximation. `usage.output_tokens` counts the candidate-label tokens that were scored.

Comparison on the docs' quick-start request (`examples/systemone-quickstart.json`), Gemma 3 4B
zero-shot vs the numbers TypeSafe publishes for Jev:

| answer | Jev (docs) | openjev / Gemma 3 4B |
|---|---|---|
| `department.choice` | technical, p=0.84, confidence 0.60 | technical, p=1.00, confidence 1.00 |
| `frustration.score` | 1.04 (annoyed but polite) | 2.00 (furious) |
| `is_urgent.noul` | 0.999 | 0.005 |

Same routing decision, but Gemma is over-confident and disagrees on the two judgement calls. Zero-shot
probabilities are softmaxed next-token likelihoods, not calibrated judgements. Closing the gap means
labelled data and a trained head (below).

### openjev vs Jev on the synthetic test sets

`jev_eval.py` sends each row of an openjev JSONL set to TypeSafe's hosted Jev as one System One
request: the row's context is the `state`, its options are the criteria of a single `choice`
question. It reports top-1, top-3 and ECE like `openjev lora-eval`, and caches every response in
`runs/jev/<set>.jsonl` so a rerun only sends new rows. The key comes from `TYPESAFE_API_KEY` or
`../.env`.

`compare.py` runs every engine on the same rows in one go: Jev, openjev zero-shot (chat format with
the options listed), and openjev with any number of LoRA adapters on one 4-bit base model. Each
adapter is scored in the format it was trained in (from its `openjev_lora.json`, or a `:chat` /
`:plain` suffix).

```sh
.venv/bin/python jev_eval.py data/synthetic/claims_verify/test.jsonl   # Jev only, one set
.venv/bin/python compare.py data/synthetic/movies_general/test.jsonl \
    movies=runs/lora-movies_general_v2:plain movies_int8=runs/lora-movies_general_v2-int8:plain
```

| engine (movies_general, 300 rows) | format | top-1 | top-3 | ECE | same pick as Jev | median s/row |
|---|---|---|---|---|---|---|
| jev | System One choice | 0.820 | 0.977 | 0.117 | 100% | 0.27 |
| openjev | chat | 0.670 | 0.950 | 0.190 | 68% | 0.24 |
| openjev | plain | 0.290 | 0.697 | 0.612 | 32% | 0.21 |
| openjev + movies LoRA | plain | 1.000 | 1.000 | 0.001 | 82% | 0.22 |
| openjev + movies LoRA, int8 | plain | 1.000 | 1.000 | 0.001 | 82% | 0.22 |

Top-1 on the 300-row test splits (phrasings held out of training), 2026-09-26. Jev is `jev-1.13.0`;
openjev is Gemma 4 E4B, 4-bit, on an RTX 4060 laptop GPU, zero-shot or with a LoRA adapter
([LoRA fine-tuning](docs/lora.md)):

| test set | Jev | openjev zero-shot | openjev + LoRA |
|---|---|---|---|
| news: headline -> 8 sections | **1.000** | 0.997 (chat) | 0.893 (plain format) |
| claims_verify: abstract + claim -> supported / refuted / not enough info | **0.987** | 0.90 (chat, 100 rows) | not trained yet |
| noul: does the review ask for a refund? | **0.993** | ~1.00 (chat, 100 rows) | 0.79 (plain format) |
| claims_attrib: claim -> 1 of 8 full references | 0.820 | 0.79 (chat) | **0.867** (1000 rows) |
| categorize: description -> 10 genres | 0.813 | 0.81 (chat) | **0.900** |
| movies: request -> 1 of 10 made-up films | 0.820 | 0.67 (chat) | **1.000** |
| claims_attrib_keys: claim -> 1 of 8 "Surname et al. (year)" keys | 0.117 | 0.14 | **1.000** |

- Zero-shot, openjev in the chat format is close to Jev on the general tasks (news, noul,
  categorize, attribution by reference); Jev is clearly ahead on claim verification.
- LoRA wins where the answer depends on facts only the training data holds: invented papers behind
  bare citation keys (1.000 vs 0.117, chance is 0.125), made-up film titles, genre rules. Jev cannot
  be fine-tuned, so these stay near what it can infer from the text.
- The sets were built to test exactly that, so they favour fine-tuning by construction; on a real
  task the gap is as large as the private knowledge it needs.
- Speed and cost: per question the two are close, Jev 0.27 s median over the network and openjev
  0.2-0.7 s on the laptop GPU once the model is loaded. Jev serves many requests in parallel, so it
  wins on volume (100 films, two questions each: 3.5 s vs 73 s one after another). The whole Jev run
  (2,430 requests, ~850k input tokens) cost about $0.04; openjev costs nothing per call, keeps the
  data local, and needs 0.5-3 h of training per task.

Caveats: movies, categorize, claims_attrib_keys and claims_attrib were rerun with `compare.py` on
exactly Jev's rows (the int8 adapters score the same as the float32 ones); news, noul and
claims_verify come from the training runs, some on 100-row samples, and the news and noul adapters
use the plain rather than the chat format. The two systems also see the options differently
(inside a chat turn vs as `choice` criteria).

### openjev vs Jev on real data

Both engines zero-shot, 2026-09-26:

| task | Jev | openjev (Gemma 4 E4B) |
|---|---|---|
| 287 BBC News/Sport stories -> 8 sections, scored against the BBC's own section feeds (`bbc_catalogue.py`) | **0.840** | 0.829 |
| 100 IMDb films, `score` question on the IMDb rating band: Spearman vs the live rating (`imdb_rate.py`) | **0.952** | 0.488 |
| same films, `noul` "rated 7.0 or higher?": accuracy | **0.90** | 0.64 |

- On news the two agree on 91% of the stories; most "misses" of both are BBC section choices that
  do not map onto the 8 sections (floods under Science & Environment, AI companies under Business).
  The synthetic-news LoRA adapter drops openjev to 0.791 here: it learned a narrow "science".
- On films, the question needs world knowledge about how each film was received, where the 4B
  model is far behind. Worded as "Do IMDb users rate this film...", Gemma answers "no, I have no
  access" for every film (Spearman 0.009); the numbers above use "Using what you know about this
  film..." for both engines (Jev scores the same either way).
- IMDb data comes from IMDb's non-commercial datasets and BBC stories from its RSS feeds; neither
  is committed (see `.gitignore`).

## Training a head (per-task, on frozen Gemma features)

```sh
make features    # one prefix-shared pass per example, cached to runs/feats/{train,validation,test}.npz
make train       # jevlike's cross-attention head in MLX, listwise cross-entropy -> runs/head.safetensors
make eval-head   # top-1/top-3, ECE, and the shuffled-context control on the test split
```

- `openjev features DATA --out F.npz` stops Gemma before the LM head, keeps every context token's
  final hidden state and the masked-mean of each option's tokens (float16). Options are encoded on
  their own, as in jevlike, so the head has to do the matching. `--contextual` encodes them as
  continuations of the context instead: stronger features, but the match leaks into the option
  vectors and the shuffled-context control below stops meaning anything.
- `openjev train` = AdamW 5e-4 (2e-3 diverges on Gemma features, whose norms are ~115), weight decay 1e-4, grad clip 1.0, 8 epochs, batch 64, best validation
  epoch kept. The checkpoint is `head.safetensors` plus `head.json` (rank, hidden size, training config).
- `openjev eval-head` reports what `jevlike-eval` reports: top-1, top-3, 10-bin expected calibration
  error, and the same metrics with every example paired with another example's context. If the
  shuffled number does not collapse, the head is reading option priors rather than the state.

Result on the synthetic split (2000 train / 400 validation / 400 test, 2026-09-17): features at
77 ms per example, head training 8 epochs in about 15 s.

| | top-1 | top-3 | ECE |
|---|---|---|---|
| head on frozen Gemma features, test | 0.970 | 1.000 | 0.027 |
| same head, shuffled-context control | 0.258 | 0.698 | 0.724 |
| jevlike tiny byte encoder, test | 0.998 | 1.000 | 0.008 |
| Gemma zero-shot, `--norm sum`, test | 1.000 | 1.000 | n/a |

The control sits at chance (about 4.5 options per row), so the head really matches options to the
state, and its ECE of 0.03 is what a calibrated `confidence` looks like.

Python:

```python
from openjev import OptionScorer
s = OptionScorer("models/gemma-3-4b-it", batch_size=8)
for r in s.score("The capital of France is", [" Paris", " Berlin"], norm="mean"):
    print(r.option, r.probability, r.logprob_sum, r.n_tokens)
print(s.last_timing)
```

### Normalisation (`--norm`)

| norm | score | use when |
|---|---|---|
| `mean` (default) | sum of option-token log-probs divided by token count | options are the same length in tokens |
| `sum` | total log-prob | options differ in length, or you want raw likelihood |
| `pmi` | sum minus the option's unconditional log-prob (BOS-only context) | options differ in base-rate plausibility; costs one extra batched pass |

Check the `n_tokens` column before trusting the ranking: when it differs between options, the choice
of norm, not the model, is deciding. Asking Qwen 2.5 1.5B for the capital of Italy, `" Torino"` is
two tokens and `" Roma"` is one, so `mean` divides Torino's much worse total (-7.85, against Roma's
-4.04) by two and puts it first at 49.9%. `sum` gives Roma 87.3%. `pmi` is worse again here at 85.1%
for Torino, because it divides out the unconditional probability and "Torino" is the rarer word --
the base rate was the signal, not the nuisance. Tokenisation makes this model-specific: Gemma 3 has
all four city names as single tokens, so `mean` and `sum` agree and the trap never appears.

## PyTorch backend (NVIDIA / CPU)

Every command takes `--backend torch`, which swaps `mlx`/`mlx-lm` for `transformers`+`torch` and
keeps the same design: prefill the context once, replicate its KV cache across the option batch,
score in one padded forward pass. `openjev check` verifies that path against naive per-option
re-encoding on either backend. The server also reads `OPENJEV_BACKEND`.

```sh
openjev score --backend torch --norm sum --model Qwen/Qwen2.5-1.5B-Instruct \
    --context "The capital of France is" --option " Paris" --option " Berlin"

openjev check  --backend torch --model Qwen/Qwen2.5-1.5B-Instruct   # cached vs naive
openjev serve  --backend torch --model google/gemma-3-4b-it --quantize 4bit
```

Gemma 4 E-series models (`google/gemma-4-E4B-it`) work on an 8 GB card with `--quantize 4bit`.
They carry a ~2.8B-parameter per-layer embedding table that bitsandbytes cannot quantise; `score`,
`eval`, `check`, `features` and `serve` keep it (and the unused vision/audio towers) on the CPU and
move only the looked-up rows, as LoRA training does, so the model takes ~3.2 GB of GPU memory. Their
sliding-window layers keep the last 512 tokens; the cached path copies the model's own cache, so
longer prompts score correctly.

```sh
openjev eval data/synthetic/claims_verify/test.jsonl --backend torch --quantize 4bit \
    --model google/gemma-4-E4B-it --chat --norm sum     # ~0.2 s per row on an RTX 4060 laptop GPU
```

### Quantisation (`--quantize`)

`none` (default) is bf16; `8bit` and `4bit` go through bitsandbytes, `4bit` as NF4 with double
quantisation and a bf16 compute dtype. The server also reads `OPENJEV_QUANTIZE`. Quantised weights
are pinned to GPU 0, because bitsandbytes cannot run a module that accelerate parked on the CPU and
`device_map="auto"` offloads on a small card even when the model would fit.

Gemma 3 4B it on an RTX A2000 Laptop (4 GB), 160 examples, bf16 as the reference:

| `--quantize` | VRAM | median latency | agreement with bf16 | same, where bf16 is confident |
|---|---|---|---|---|
| `none` (bf16) | 2.6 GB + **2.9B params offloaded to CPU** | 2114 ms | -- | -- |
| `8bit` | 4.8 GB (**over the 4 GB card**) | 963 ms | 0.825 | 0.858 |
| `4bit` | 3.1 GB | **211 ms** | 0.825 | 0.875 |

bf16 does not fit: two thirds of the weights end up on the CPU and every forward pass streams them
back, which is where the 2.1 s comes from. `8bit` is the worst of the three here -- it overflows the
card *and* is no more faithful than `4bit`. Note what the last column means: even at its best, 4-bit
flips about one confident decision in eight. Fine for ranking, not free if you depend on the
probabilities being calibrated. Smaller models degrade more, not less (Qwen 0.5B agrees with its own
bf16 only 0.675 of the time), so quantise the big model rather than shrinking to a small one.

Same workload as the MLX table below (202-token context, 8 options, 242 option tokens), Gemma 3 4B
in 4-bit on the same 4 GB card:

| path | median latency |
|---|---|
| context cached once, options batched | 0.63 s |
| context re-encoded per option, no cache | 4.25 s |

`openjev check` compares log-probs with an absolute tolerance (`--tol`, default 0.5) that does not
scale with context length, so quantised runs over long contexts need `--tol 1.0` to pass on what is
ordinary NF4 noise (0.4% relative).

## Measured on M5 Pro, 64 GB (MLX, bf16)

Workload: 202-token context, 8 options, 242 option tokens total.

| path | median latency |
|---|---|
| context cached once, options batched | 0.17 s |
| context re-encoded per option, no cache | 0.68 s |

Model load is about 1 s from a warm disk. First forward pass adds under a second of warm-up.

## Validating against jevlike

`jevlike` is installed into the same venv (`uv pip install -e ../../vinnylarouge/jevlike`), so both
CLIs read the same JSONL. Generate its synthetic menu set, then score it both ways:

```sh
.venv/bin/jevlike-data synthetic --output data/synthetic
.venv/bin/openjev eval data/synthetic/test.jsonl --norm sum --sep $'\nChoice: '
.venv/bin/jevlike-train data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --output runs/synthetic-tiny.pt --device mps
.venv/bin/jevlike-eval runs/synthetic-tiny.pt data/synthetic/test.jsonl --device mps
```

Results on the 400-row synthetic test set (2026-09-16):

| scorer | training | top-1 | top-3 | median latency / example |
|---|---|---|---|---|
| openjev, Gemma 3 4B zero-shot, `--norm sum` | none | 1.000 | 1.000 | 0.086 s |
| openjev, `--norm mean` | none | 0.988 | 1.000 | 0.086 s |
| openjev, `--norm pmi` | none | 0.988 | 1.000 | 0.153 s |
| jevlike tiny byte encoder + head | 2000 rows, 8 epochs | 0.998 | 1.000 | well under 10 ms |

The synthetic task is easy for both. The real validation is your own labelled rows: run
`openjev eval` on them zero-shot and compare against a `jevlike-train`/`jevlike-eval` run on the
same split. If Gemma zero-shot is close to the trained head, Route B is enough; if not, train a head
(Route A) with `make features && make train && make eval-head`.

## Demo: Doom in the terminal

`demo/doom/` runs ViZDoom headless, describes each frame in a line of text, and
lets the server rank the action menu with one `/score` call (or one System One
`choice` question). Start `make serve`, then `make doom`. Details and keys in
[`demo/doom/README.md`](demo/doom/README.md).

![openjev playing Doom in the terminal: the model ranks the action menu each step](docs/media/doom-recording.gif)

Full-resolution recording: [docs/media/doom-recording.mov](docs/media/doom-recording.mov).

## Layout

- `openjev/scorer.py`: `OptionScorer` (prefill, cache expansion, batched scoring, naive reference).
- `openjev/systemone.py`: System One request/answer models, prompt renderers, zero-shot answers.
- `openjev/server.py`: FastAPI app, `/health`, `/score`, `/v1/systemone`.
- `openjev/features.py`: frozen-Gemma feature extraction and the `.npz` feature cache.
- `openjev/head.py`: jevlike's cross-attention head in `mlx.nn`, save/load.
- `openjev/train.py`: head training loop and evaluation (top-k, ECE, shuffled-context control).
- `openjev/cli.py`: `openjev score | eval | bench | check | serve | features | train | eval-head`.
- `openjev/{scorer,features,head,train}_torch.py`: the `--backend torch` counterparts of the four
  modules above (`transformers` + `torch`, bitsandbytes quantisation). `systemone.py`, `server.py`
  and `cli.py` are backend-agnostic and import a backend only when one is selected.
- `jev_eval.py`: scores a JSONL set with TypeSafe's hosted Jev for the openjev-vs-Jev comparison.
- `compare.py`: Jev vs openjev zero-shot vs openjev + LoRA adapters on the same rows of one set.
- `demo/doom/`: Doom in the terminal, the server picks every action (`make doom`).
- `models/`: downloaded weights (git-ignored).
