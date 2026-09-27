"""Evaluate AG News categorization on the held-out test split.

Two modes:
  rank     -- score every category as an option with openjev's OptionScorer
              (the same one-pass ranking the server uses) and report top-1 /
              top-3 accuracy plus ECE against the gold category.
  generate -- greedy-decode a category and check it matches exactly.

Backend is whatever openjev.scorer.OptionScorer picks (MLX on Apple silicon,
PyTorch/transformers elsewhere) via `--model`/`--adapter`; the news-lora-2b
adapter was trained against `google/gemma-2-2b-it` (see train_news_torch.py),
so that is the default here, not the chess-eval Gemma 3 4B default.

Usage:
    .venv/bin/python finetune/eval_news.py                              # base model, rank
    .venv/bin/python finetune/eval_news.py --adapter adapters/news-lora-2b/checkpoint-300
    .venv/bin/python finetune/eval_news.py --mode generate --adapter adapters/news-lora-2b/checkpoint-300
"""

from __future__ import annotations

import argparse
import json
import time

from openjev.scorer import OptionScorer, iter_jsonl

DEFAULT_MODEL = "google/gemma-2-2b-it"


def options_from_prompt(prompt: str) -> list[str]:
    line = next(l for l in prompt.splitlines() if l.startswith("Options: "))
    return line[len("Options: ") :].split(", ")


def eval_rank(scorer: OptionScorer, examples: list[dict]) -> dict:
    top1 = top3 = 0
    ranks = []
    conf_bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, sum(conf), sum(correct)
    for i, ex in enumerate(examples):
        if i % 5 == 0:
            print(f"Evaluated {i}/{len(examples)}")
        options = options_from_prompt(ex["prompt"])
        target = ex["completion"].strip()
        if len(options) == 1:  # forced answer; the scorer needs >= 2 options
            rank = 0
            top_prob = 1.0
        else:
            scores = scorer.score(ex["prompt"], [f" {o}" for o in options], norm="sum")
            ordered = sorted(scores, key=lambda s: s.score, reverse=True)
            rank = next(i for i, s in enumerate(ordered) if s.option.strip() == target)
            top_prob = ordered[0].probability
        ranks.append(rank)
        hit = rank == 0
        top1 += hit
        top3 += rank < 3
        if len(options) > 1:
            b = min(9, int(top_prob * 10))
            conf_bins[b][0] += 1
            conf_bins[b][1] += top_prob
            conf_bins[b][2] += hit
    n = len(examples)
    ece = sum(c * abs(sc / c - sh / c) for c, sc, sh in conf_bins if c) / n
    return {
        "n": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "mean_rank": sum(ranks) / n,
        "ece": ece,
        "chance_top1": sum(1 / len(options_from_prompt(e["prompt"])) for e in examples) / n,
    }


def eval_generate(model_path: str, adapter: str | None, examples: list[dict]) -> dict:
    scorer = OptionScorer(model_path, adapter_path=adapter)
    legal = correct = 0
    samples = []
    for i, ex in enumerate(examples):
        if i % 5 == 0:
            print(f"Evaluated {i}/{len(examples)}")
        options = options_from_prompt(ex["prompt"])
        scores = scorer.score(ex["prompt"], [f" {o}" for o in options], norm="sum")
        pred = max(scores, key=lambda s: s.score).option.strip()
        legal += pred in options
        correct += pred == ex["completion"].strip()
        if len(samples) < 5:
            samples.append({"pred": pred, "gold": ex["completion"].strip()})
    n = len(examples)
    return {"n": n, "legal": legal / n, "correct": correct / n, "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/news/test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["rank", "generate"], default="rank")
    args = ap.parse_args()

    examples = list(iter_jsonl(args.data))
    if args.limit:
        examples = examples[: args.limit]

    t0 = time.time()
    if args.mode == "rank":
        scorer = OptionScorer(args.model, batch_size=16, adapter_path=args.adapter)
        result = eval_rank(scorer, examples)
        print("Final result:", result)
    else:
        result = eval_generate(args.model, args.adapter, examples)
    result["adapter"] = args.adapter
    result["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
