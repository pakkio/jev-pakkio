"""Evaluate next-move prediction on the held-out Lichess puzzle split.

Two modes:
  rank     -- score every legal move as an option with openjev's OptionScorer
              (the same one-pass ranking the server uses) and report top-1 /
              top-3 accuracy against the puzzle solution.
  generate -- greedy-decode a move and check it is (a) legal and (b) correct.

Usage:
    .venv/bin/python finetune/eval_chess.py                      # base model, rank
    .venv/bin/python finetune/eval_chess.py --adapter adapters/news-lora
    .venv/bin/python finetune/eval_chess.py --mode generate --adapter adapters/news-lora
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from openjev.scorer import OptionScorer, iter_jsonl


def options_from_prompt(prompt: str) -> list[str]:
    line = next(l for l in prompt.splitlines() if l.startswith("Options: "))
    return line[len("Options: ") :].split(", ")


def eval_rank(scorer: OptionScorer, examples: list[dict]) -> dict:
    top1 = top3 = 0
    ranks = []
    for i, ex in enumerate(examples):
        if i % 5 == 0:
            print(f"Evaluated {i}/{len(examples)}")
        moves = options_from_prompt(ex["prompt"])
        if len(moves) == 1:  # forced move; the scorer needs >= 2 options
            rank = 0
        else:
            scores = scorer.score(ex["prompt"], [f" {m}" for m in moves], norm="sum")
            ordered = sorted(scores, key=lambda s: s.score, reverse=True)
            rank = next(i for i, s in enumerate(ordered) if s.option == ex["completion"])
        ranks.append(rank)
        top1 += rank == 0
        top3 += rank < 3
    n = len(examples)
    return {
        "n": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "mean_rank": sum(ranks) / n,
        "chance_top1": sum(1 / len(options_from_prompt(e["prompt"])) for e in examples) / n,
    }


def eval_generate(model_path: str, adapter: str | None, examples: list[dict]) -> dict:
    from mlx_lm import generate, load

    model, tok = load(model_path, adapter_path=adapter)
    legal = correct = 0
    samples = []
    for i, ex in enumerate(examples):
        if i % 5 == 0:
            print(f"Evaluated {i}/{len(examples)}")
        out = generate(model, tok, prompt=ex["prompt"], max_tokens=6, verbose=False)
        pred = out.strip().split()[0] if out.strip() else ""
        moves = options_from_prompt(ex["prompt"])
        legal += pred in moves
        correct += pred == ex["completion"].strip()
        if len(samples) < 5:
            samples.append({"pred": pred, "gold": ex["completion"].strip()})
    n = len(examples)
    return {"n": n, "legal": legal / n, "correct": correct / n, "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemma-3-4b-it")
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
