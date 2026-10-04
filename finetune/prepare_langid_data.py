"""Build a language-identification set for LoRA scoring from papluca/language-identification.

Each row: the text as `context`, the 20 language names (shuffled per row) as `options`,
the index of the right one as `label`. Train/validation/test are balanced per language
and drawn from the dataset's own splits, so no text is shared between them.

Usage:
    .venv/bin/python finetune/prepare_langid_data.py --train 2000 --valid 400 --test 1000 --out data/langid
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "papluca/language-identification"
NAMES = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English",
         "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
         "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
         "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"}
MAX_CHARS = 300


def load(split: str, cache: Path) -> dict[str, list[str]]:
    path = hf_hub_download(REPO, f"{split}.csv", repo_type="dataset", local_dir=cache)
    by_lang: dict[str, list[str]] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        text = " ".join(r["text"].split())[:MAX_CHARS]
        if text:
            by_lang.setdefault(r["labels"], []).append(text)
    return by_lang


def build(by_lang: dict[str, list[str]], n: int, rng: random.Random) -> list[dict]:
    per = max(1, n // len(NAMES))
    rows = []
    for code, texts in by_lang.items():
        for text in rng.sample(texts, min(per, len(texts))):
            options = list(NAMES.values())
            rng.shuffle(options)
            rows.append({"context": text, "options": options,
                         "label": options.index(NAMES[code]), "lang": code})
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--valid", type=int, default=400)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--out", default="data/langid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split, src, n in (("train", "train", args.train), ("validation", "valid", args.valid),
                          ("test", "test", args.test)):
        rows = build(load(src, out / "raw"), n, rng)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows")


if __name__ == "__main__":
    main()
