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

## Demo: Doom in the terminal

`demo/doom/` runs ViZDoom headless, describes each frame in a line of text, and
lets the server rank the action menu with one `/score` call (or one System One
`choice` question). Start `make serve`, then `make doom`. Details and keys in
[`demo/doom/README.md`](demo/doom/README.md).

![openjev playing Doom in the terminal: the model ranks the action menu each step](docs/media/doom-recording.gif)

Full-resolution recording: [docs/media/doom-recording.mov](docs/media/doom-recording.mov).

## Setup

### Windows and Linux

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository in PowerShell or a Linux shell:

```sh
uv sync
uv run hf auth login
uv run hf download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it
uv run openjev serve --backend torch --device auto --port 8000
```

Accept the Gemma license on Hugging Face before downloading. You can also pass
`--model google/gemma-3-4b-it` to load directly from the Hugging Face cache.
No Make, Xcode, shell activation, or `.venv/bin` paths are needed here.

`--backend auto` (the default) selects MLX on Apple silicon and PyTorch elsewhere.
For PyTorch, `--device auto` selects CUDA when available, then MPS, then CPU.
Use `--device cuda` or `--device cuda:1` to require a specific GPU (fails clearly
if CUDA is unavailable), or `--device cpu` to force CPU inference.

For NVIDIA acceleration, install the GPU driver and a matching PyTorch build using
the [official PyTorch installer](https://pytorch.org/get-started/locally/).
Run its pip command in this repository's virtual environment (use `uv pip install`
in place of `pip3 install`). After a custom PyTorch install, use `uv run --no-sync`
so uv does not replace your selected build. On supported Linux AMD systems,
PyTorch ROCm builds also use the `cuda` device name.

```sh
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --no-sync openjev serve --backend torch --device cuda --batch-size 2
uv run --no-sync openjev check --backend torch --device cuda
uv run --no-sync openjev score --backend torch --device cuda --context "The capital of France is" --option " Paris" --option " Berlin"
```

The base 4B weights need roughly 8 GB just for GPU model weights at 16-bit precision,
plus memory for activations and the option KV caches. Reduce `--batch-size` and
context length if you run out of GPU memory. CPU runs use float32 and need more RAM.
Use original Hugging Face weights with PyTorch; MLX quantized weights and MLX LoRA
adapters are not supported by this backend.

Scoring, evaluation, benchmarks, correctness checks, and both HTTP scoring endpoints
support PyTorch. Frozen-feature extraction, head training/evaluation, and the chess
LoRA training workflow still require MLX on Apple silicon. The Doom terminal UI
also has its own platform dependencies; cross-platform support here covers the
scoring CLI and HTTP server.

### Apple silicon / existing Make workflow

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

# Chat template (context as user turn, options scored as the reply) + PMI normalisation
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

## Measured on Colab T4, 16 GB (PyTorch, bf16, fine-tuned LoRA vs Jev)

`gemma-2-2b-it` + `adapters/news-lora-2b/checkpoint-300`, scoring 4-way news category
classification (World/Sports/Business/Sci-Tech) on a 40-article held-out set
(`bbc_test.jsonl`), `--norm sum`, compared against TypeSafe's hosted Jev on the same data:

| setup | accuracy | latency / article |
|---|---|---|
| Jev (cloud) | 87.5% (35/40) | 0.29 s |
| openjev + LoRA, full GPU, shared-prefix cached scoring | 85.0% (34/40) | 0.47 s |
| openjev + LoRA, full GPU, naive per-option re-encoding | 85.0% (34/40) | 1.43 s |
| openjev + LoRA, CPU-offloaded (4 GB laptop GPU) | 85.0% (34/40) | 7.40 s |

Accuracy is stable across every configuration — it was never an accuracy problem, purely
latency/placement. Two things worth knowing if you're on a small card:

- **Under ~6-7 GB VRAM, `device_map="auto"` will split the model across GPU and CPU per layer.**
  On a 4 GB card this thrashed at 0% GPU utilization and made scoring ~16x slower than the same
  model fully resident on a 16 GB T4, despite identical accuracy. If you see near-zero
  `nvidia-smi` GPU utilization while a forward pass runs, this is almost certainly why — either
  quantize (`--quantize 4bit`/`8bit`) to fit on the small card, or use a bigger one.
- `torch.compile` was tried on the T4 and **did not help**: Turing-generation cards (T4, and
  anything pre-Ampere) have no native bf16 tensor cores, so Inductor silently skips compiling the
  bf16 matmuls (`UserWarning: Tesla T4 does not support bfloat16 compilation natively, skipping`).
  Switching to fp16 to work around it risks Gemma's known fp16 overflow issues, and combining
  `dynamic=True` with PEFT's wrapped `forward` plus per-article varying sequence lengths caused a
  run to hang rather than compile. Not worth chasing on this class of GPU; the shared-prefix KV
  cache (already the default `score()` path, see above) is the optimization that actually pays off.

### Why shared-prefix scoring is 3x faster than naive re-encoding

The naive approach re-encodes `context + option` from scratch for every option: with 4 categories,
that's 4 full forward passes over the ~300-token article, each redoing the same context work three
extra times. The shared-prefix version (this repo's actual `OptionScorer.score()`, reproduced below
from `openjev/scorer_torch.py`) prefills the context **once**, then reuses that cached KV state for
all 4 short option continuations in a single batched pass — the article is only ever encoded once,
no matter how many options you're scoring:

```python
def score_options(context: str, options: list[str]) -> list[float]:
    # 1. Tokenize context once, and each option separately (options are short:
    #    a word or two, e.g. " World"). The leading space matters -- it has to
    #    tokenize the way training data's completions were formatted.
    ctx_ids = tok.encode(context, add_special_tokens=True)
    opts = [tok.encode(o, add_special_tokens=False) for o in options]

    # 2. PREFILL: run the model over the context exactly once, with
    #    use_cache=True so it returns the KV cache (attention keys/values for
    #    every layer, for every context token) alongside the logits. This is
    #    the expensive part -- one full attention pass over ~300 tokens -- and
    #    it now happens a single time regardless of how many options follow.
    input_ids = torch.tensor([ctx_ids], device=model.device)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    kv = _cache_tensors(out.past_key_values)       # snapshot as plain tensors
    last_logits = out.logits[0, -1].float()         # logits predicting the *next* token after context

    # 3. Pad all options to the same length so they fit one batched tensor,
    #    and build a mask so padding doesn't pollute the log-prob sum.
    n = len(opts)
    L = max(len(o) for o in opts)
    arr = torch.full((n, L), pad_id, dtype=torch.long, device=model.device)
    mask = torch.zeros((n, L), dtype=torch.float32, device=model.device)
    for i, o in enumerate(opts):
        arr[i, :len(o)] = torch.tensor(o, dtype=torch.long)
        mask[i, :len(o)] = 1.0

    # 4. Replicate the ONE cached context KV across the option batch dimension
    #    (n copies, one per option) instead of recomputing it n times. This is
    #    cheap -- copying already-computed tensors -- versus n more attention
    #    passes over the full context.
    expanded = _repeat_kv(kv, n)

    # 5. Score all n options in ONE forward pass: the model only has to attend
    #    over each option's few tokens against the (shared, reused) context KV,
    #    not re-derive the context representation from scratch.
    with torch.no_grad():
        out2 = model(input_ids=arr, past_key_values=expanded, use_cache=False)
    logits = out2.logits.float()

    # 6. The log-prob of an option's first token is predicted by the context's
    #    last logits (from step 2); each subsequent token is predicted by the
    #    previous option token. Stitch these together, then read off
    #    log p(option_token | everything before it) for every option token.
    first = last_logits.unsqueeze(0).unsqueeze(0).expand(n, 1, -1)
    pred = torch.cat([first, logits[:, :-1]], dim=1)
    log_probs = F.log_softmax(pred, dim=-1)
    target = arr.unsqueeze(-1)
    tgt_lp = log_probs.gather(-1, target).squeeze(-1)

    # 7. Sum log-probs over each option's real tokens (mask zeroes out padding),
    #    then softmax across options for a calibrated probability distribution.
    sums = (tgt_lp * mask).sum(-1).cpu().tolist()
    return _softmax(sums)
```

The saving scales with option count: 4 options here means ~4x less context-encoding work than
naive re-encoding, which roughly matches the measured 1.43 s → 0.47 s speedup (the remaining time
is the now-unavoidable single prefill pass plus the small batched option pass, not redundant work).

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

## LoRA Fine-Tuning

See `finetune/README.md` for complete pipelines demonstrating how to fine-tune Gemma locally on both Apple Silicon (MLX) and NVIDIA GPUs (PyTorch) using `OptionScorer` to predict chess moves and categorize real-world news articles.

## Layout

- `openjev/torch_backend.py`: PyTorch CPU/GPU inference and shared-prefix KV cache scoring.
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
- `demo/doom/`: Doom in the terminal, the server picks every action (`make doom`).
- `models/`: downloaded weights (git-ignored).

## Development checks

```sh
uv sync --extra torch --group test
uv run python -m unittest discover -s tests -v
```

The offline tests use a tiny randomly initialized Gemma model to compare cached,
padded scoring with full re-encoding, including sliding-window attention, multiple
batch sizes, normalization, and repeated requests. CI runs them on Windows, Linux,
and macOS. Actual GPU availability and throughput must be checked on your hardware.
