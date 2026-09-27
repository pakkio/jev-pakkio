# Fine-tuning Gemma 3 4B to pick the next chess move

A small LoRA fine-tune that teaches the local `models/gemma-3-4b-it` to
predict the best next move from a board state. Data comes from
[Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles)
(CC0), training runs on Apple silicon with `mlx_lm.lora`, and evaluation uses
openjev's own `OptionScorer` to rank every legal move in one pass.

```sh
uv sync --extra finetune   # python-chess + pyarrow
make chess-data            # ~3000 puzzles -> data/chess/{train,valid,test}.jsonl
make chess-train           # LoRA adapter -> adapters/chess-lora (~15 min on an M-series Mac)
make chess-eval            # top-1 / top-3 accuracy, base vs adapter
```

## Data

`prepare_chess_data.py` reads one row group of the dataset's first parquet
shard over HTTP range requests (no full download), keeps puzzles with rating
<= 1600 and popularity >= 70, samples 3000 of them, and splits by puzzle
(80/10/10) so no position leaks across splits. Each puzzle contributes up to
two examples (the side-to-move's first two solution moves).

Each example is a prompt/completion pair:

```
You are playing chess as black. Pick the best move.

Board (uppercase = white, lowercase = black):
8 . . . . . r . k
7 . p . . . . p .
...
  a b c d e f g h

FEN: 5r1k/1p4p1/pbp3q1/4P2P/1P1PN3/P4p2/4QB1P/R4R1K b - - 0 28
Legal moves: g6g2 g6h5 g6g4 ...
Best move:
```

completion: ` g6g2`

The legal-move list is included on purpose: it turns the task into the same
"pick one of these options" shape openjev scores, and it lets the model
output a legal move string rather than inventing one.

## Training

`mlx_lm.lora` with LoRA on the last 16 layers, batch 4, 1000 iterations
(about one epoch over ~4200 examples), lr 1e-4, `--mask-prompt` so the loss
only covers the move tokens. Peak memory is about 15 GB.

## Using the adapter

`OptionScorer(model_path, adapter_path="adapters/chess-lora")` loads the
adapter on top of the base weights. `eval_chess.py --mode generate` shows
greedy decoding of the move instead of ranking.

## Results (2026-09-18, 522 held-out test positions, M5 Pro)

Ranking every legal move with `OptionScorer` (norm=sum):

| model            | top-1 | top-3 | mean rank of gold move |
|------------------|-------|-------|------------------------|
| chance           | 5.9%  |       |                        |
| gemma-3-4b-it    | 7.5%  | 19.3% | 12.0                   |
| + chess-lora     | 24.7% | 48.9% | 5.9                    |

Greedy generation on the first 100 test positions:

| model            | legal move | correct move |
|------------------|------------|--------------|
| gemma-3-4b-it    | 81%        | 8%           |
| + chess-lora     | 100%       | 26%          |

Training: 1000 iters, ~15 min, val loss 15.5 -> 0.37. The task is hard for
a text model (puzzles are tactics, ~17 legal moves on average), so 25% top-1
after ~4k examples is a real signal rather than a solved problem. More data
(`--puzzles 20000`) and more iterations are the obvious next knobs.

## Playing against it

```sh
make chess-play             # you are white
make chess-play PLAY=black
.venv/bin/python finetune/play_chess.py --no-adapter    # base model, for comparison
.venv/bin/python finetune/play_chess.py --fen "<fen>"   # start from a position
```

Type moves in UCI (`e2e4`) or SAN (`Nf3`, `O-O`). `hint` shows the model's
top choices for your side, `moves` lists legal moves, `undo` takes back a
full move, `fen` prints the position, `quit` exits. The model plays the
top-ranked legal move each turn and prints its runner-up candidates.

## PyTorch LoRA Fine-Tuning (Local & Fallback)

For systems with NVIDIA GPUs, PyTorch-based training scripts have been added:
- `train_chess_torch.py`: Replicates the MLX Lichess training pipeline using `trl` and `bitsandbytes` (4-bit quantization).
- `train_news_torch.py`: Fine-tunes the model on the `ag_news` categorization dataset.

These scripts load the model in 4-bit precision and can comfortably fit LoRA fine-tuning for a 2B/3B parameter model on a 4GB VRAM GPU (e.g. RTX A2000).

### AG News Categorization Benchmark
A pipeline compares the zero-shot Native TypeSafe SystemOne API (`jev-latest`) against a local PyTorch LoRA (`google/gemma-2-2b-it`, `train_news_torch.py`, 10,000 labeled `ag_news` examples) and against the un-tuned base model, so the LoRA's advantage over a zero-shot model and its advantage from labeled data are visible separately. `finetune/eval_news.py` runs the local side (rank mode: score each of the 4 categories in one pass, `norm=sum`); `finetune/eval_news_native.py` runs Jev and caches choice + confidence per item. All three models see the same 500-example `ag_news` test slice and the same 40-article BBC set (`bbc_test.jsonl`), so the numbers below are the first apples-to-apples run of this comparison — earlier drafts of this table quoted "96% vs 85%" from a LoRA run over only 50 examples (no interval reported) against a 500-example Jev run, evaluated with a script (importing `mlx_lm`, defaulting to `gemma-3-4b-it`) that doesn't match the PyTorch/`gemma-2-2b-it` pipeline that actually trained the adapter — nobody could reproduce that number from this repo. ECE is 10-bin expected calibration error on each model's own top-1 confidence.

**`ag_news` test split, n=500 (95% Wilson CI)**

| Model | Top-1 | 95% CI | Top-3 | ECE |
|-------|------:|--------|------:|-----:|
| `gemma-2-2b-it`, zero-shot | 77.4% | [73.5%, 80.8%] | 93.2% | 0.181 |
| `jev-latest`, zero-shot | 85.4% | [82.0%, 88.2%] | — | 0.094 |
| `gemma-2-2b-it` + LoRA (10k labels, step 300) | 91.0% | [88.2%, 93.2%] | 99.4% | 0.049 |

**`bbc.com` RSS articles, n=40, out-of-distribution (95% Wilson CI)**

| Model | Top-1 | 95% CI | Top-3 | ECE |
|-------|------:|--------|------:|-----:|
| `gemma-2-2b-it`, zero-shot | 70.0% | [54.6%, 81.9%] | 92.5% | 0.241 |
| `jev-latest`, zero-shot | 87.5% | [73.9%, 94.5%] | — | 0.181 |
| `gemma-2-2b-it` + LoRA (10k labels, step 300) | 85.0% | [70.9%, 92.9%] | 100% | 0.124 |

Two things worth separating out:

- **Zero-shot vs. zero-shot, the fair comparison the old table skipped**: Jev clearly beats the untuned local 2B model on both sets (85.4% vs 77.4% in-distribution, 87.5% vs 70.0% out-of-distribution), and is better calibrated (lower ECE) on both. That's the baseline the LoRA has to clear, not zero-shot Jev directly.
- **The LoRA's win is a 10,000-label win, not a model-size win.** It only overtakes Jev in-distribution, where its CI [88.2%, 93.2%] and Jev's [82.0%, 88.2%] barely separate. On the BBC set — the one actually meant to test generalization — **Jev's zero-shot 87.5% beats the LoRA's 85.0%**, and the two CIs overlap almost entirely at n=40, so neither number should be read as a decisive win. The LoRA's best result there is calibration (ECE 0.124 vs 0.181), not accuracy.

What this doesn't cover: an LoRA-vs-Jev-plus-same-labels comparison (e.g. a regression on Jev's per-class probabilities, the way the phishing benchmark in the main README does it) would be a fairer test of "does the LoRA add anything beyond what Jev's probabilities already contain," and hasn't been run. n=40 on BBC is also small enough that a few more articles would meaningfully tighten those intervals.

The LoRA evaluated at roughly **1.3s per article** on this 4GB GPU (RTX A2000) using the `OptionScorer` PyTorch backend, measured solo (without another job contending for the same card).

### Why Local LoRA vs Cloud APIs?
Local inference here (~1.3s/article on a 4GB laptop GPU) is slower than a Jev API call over the network. What local buys instead:
1. **Privacy.** Sensitive internal documents, private emails, or proprietary code never leave the machine — this is the actual reason to prefer local, not cost.
2. **Marginal cost at volume, if you already paid the fixed costs.** Once the 10,000 labels are collected and the ~15-minute LoRA run is paid for, further inference is "free" in the sense of not metering per token. But that upfront cost is real: TypeSafe's disclosed pricing ($0.042 per million input tokens) puts the 500-article Jev run above at roughly **$0.002** — a fraction of a cent — which the "Zero Compute Tax" framing in an earlier draft of this section ignored entirely, along with the labeling and training time the LoRA needed to get there.
3. **Task specialization helps, but the win is conditional.** A LoRA trained on enough in-distribution labels can beat a zero-shot generalist model on that distribution (91.0% vs 85.4% here) — the results above show that's real, but modest, and it doesn't hold out-of-distribution on this benchmark.

Raw output for every row above is saved under `finetune/results/`. Reproduce with:
```sh
.venv/bin/python finetune/eval_news.py --limit 500                                              # gemma base, ag_news
.venv/bin/python finetune/eval_news.py --limit 500 --adapter adapters/news-lora-2b/checkpoint-300
.venv/bin/python finetune/eval_news.py --data bbc_test.jsonl                                     # gemma base, bbc
.venv/bin/python finetune/eval_news.py --data bbc_test.jsonl --adapter adapters/news-lora-2b/checkpoint-300
.venv/bin/python finetune/eval_news_native.py --limit 500                                        # jev, ag_news
.venv/bin/python finetune/eval_news_native.py --data bbc_test.jsonl --cache data/news/bbc_jev_cache.json
```
Running the two local-model evals concurrently on a 4GB card works but is much slower than sequential (observed ~5x slowdown from GPU contention) — run them one at a time.
