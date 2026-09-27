# LoRA fine-tuning

The head ([Training a head](training.md)) learns on top of frozen Gemma
features. LoRA goes one step further (Route C in the
[design note](design/per-task-finetuning-with-gemma.md)): it adapts Gemma
itself, so that its own option scores rank the labelled option first. The
tuned model is still a plain option scorer, and the zero-shot baseline is the
same model with the adapter switched off.

Torch backend only (`peft`, `bitsandbytes`, CUDA).

## Try the chat format first

Before training anything, score your data zero-shot with `--chat`. It puts the
context and the option list in a user turn of Gemma's chat template and scores
each option as the reply. On an instruction-tuned model this is far stronger
than the plain `context + sep` format:

| task (unseen phrasings), Gemma 4 E4B 4-bit | plain zero-shot | chat zero-shot |
|---|---|---|
| noul: does the review ask for a refund? (yes/no) | 0.51 | 1.00 |
| news: headline -> 8 categories | 0.23 | 0.99 |
| categorize: description -> 10 genres | n/a | 0.86 |
| movies: request -> 10 made-up film titles | 0.29 | 0.66 |

(100-row samples, except categorize: 300 validation rows.)

If chat zero-shot is already at target, stop there. LoRA earns its place when
the model has to learn something it cannot know, like which made-up title
matches which genre.

## Train

```sh
.venv/bin/openjev lora DATA/train.jsonl --validation DATA/validation.jsonl --test DATA/test.jsonl \
    --model google/gemma-4-E4B-it --chat --eval-every 500 --max-rows 1000 --out runs/lora-TASK
```

Or `make lora` with `LORA_DATA`, `LORA_MODEL` and `LORA` set.

- Each row scores every option as the sum of its token log-probabilities; the
  loss is listwise cross-entropy over those scores, like the head.
- Defaults: LoRA r=16, alpha=32, dropout 0.05 on the attention and MLP
  projections of every text layer (34.9M trainable parameters on E4B); AdamW at
  2e-4, 8 rows per optimiser step, 20-step warmup, linear decay.
- The best validation checkpoint is kept, so checkpoint often
  (`--eval-every`): a run stopped between checkpoints loses that progress.
- `--resume DIR` continues from a saved adapter; `--max-rows N` stops after N
  rows.
- With `--test`, the run ends with top-1/top-3/ECE on the test split and the
  shuffled-context control, and the zero-shot baseline is measured first.

!!! note "Fitting an 8 GB card"
    Weights load 4-bit (QLoRA). Gemma 4 E-series models carry a ~2.8B-parameter
    per-layer embedding table that bitsandbytes does not quantise; it is only
    ever indexed, so `load_model` keeps it on the CPU and moves just the looked-up
    rows. With gradient checkpointing, training peaks at about 3.9 GiB.

## Evaluate and shrink

```sh
.venv/bin/openjev lora-eval runs/lora-TASK DATA/test.jsonl --model google/gemma-4-E4B-it [--chat]
.venv/bin/openjev lora-quantize runs/lora-TASK        # -> runs/lora-TASK-int8
```

`lora-eval` reports the adapter, the shuffled-context control and the same
model zero-shot. `lora-quantize` stores the adapter as int8 with one float16
scale per row: 134 MB becomes 36 MB, and the loader dequantises it
transparently. On the movies test the int8 adapter scores the same as the
float32 one (top-1 1.000, ECE 0.0008).

## What the experiments showed

All on synthetic sets of 2000/300/300 rows where validation and test use
phrasings never seen in training (`gen_movies_general.py`, `gen_news.py`,
`gen_noul_categorize.py`), Gemma 4 E4B 4-bit on an RTX 4060 8 GB,
2026-09-22/23.

| run | rows trained | test top-1 | test ECE | shuffled control |
|---|---|---|---|---|
| movies, plain zero-shot | 0 | 0.290 | 0.612 | |
| movies, LoRA, 8 hints per film | 1000 | 0.903 | 0.071 | 0.230 |
| movies, LoRA, 8 hints per film | 2800 | 0.963 | 0.015 | 0.250 |
| movies, LoRA, 32 hints per film (v2) | +600 | 1.000 | 0.001 | 0.270 |
| news, plain LoRA | 600 | 0.893 | 0.088 | 0.113 |
| noul, plain LoRA | 1000 | 0.790 | 0.154 | 0.470 |

- **Data variety beat epochs.** With 8 training hints per film, validation
  stalled at 0.95 however long it trained; 32 hints per film reached 1.00 on
  unseen hints after 600 more rows.
- **The prompt format matters more than the adapter.** Plain-format LoRA on
  noul reached 0.79; the untrained model in chat format gets ~1.00.
- **The shuffled-context control collapses to chance** in every run, so the
  adapters read the input rather than learning option priors.
- **Memorisation looks like success.** On `movies_hard2k`, where test reuses
  the training hints, LoRA hits 1.000 after 1000 rows, but that measures
  recall, not generalisation. Hold phrasings out of training.
