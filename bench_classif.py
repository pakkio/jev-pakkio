"""Controller check: AG News as one System-One `choice` question per row.

    uv run --no-sync python bench_classif.py --controller local      # jev_pakkio serve on :8000
    uv run --no-sync python bench_classif.py --controller typesafe   # hosted Jev, key from env or ../.env

Reports accuracy with a 95% Wilson interval, ECE (10 bins on probabilities[choice]) and latency, on the same
rows for both controllers (data/news/agnews_test_1000.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

from bench_mem import api_key
from jev_pakkio.mem import HTTPController

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
QUESTION = {
    "type": "choice",
    "instructions": "Which section of a news site does this article belong in?",
    "criteria": {
        "World": "International news, politics, conflicts and government affairs.",
        "Sports": "Sports events, athletes and teams.",
        "Business": "Companies, markets, economy and finance.",
        "Sci/Tech": "Science, technology, computing, the internet and space.",
    },
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ece(conf: list[float], ok: list[bool], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1.0)]
        if idx:
            total += len(idx) / len(conf) * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=("local", "typesafe"), required=True)
    ap.add_argument("--data", default="data/news/agnews_test_1000.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-label", type=int, default=0, help="stratified: first N rows of each label (overrides --limit)")
    ap.add_argument("--workers", type=int, default=0, help="default: 4 hosted, 1 local")
    ap.add_argument("--out", default=None, help="write per-row predictions as JSON")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    if a.per_label:
        seen = {i: 0 for i in range(len(LABELS))}
        keep = []
        for r in rows:
            if seen[r["label"]] < a.per_label:
                seen[r["label"]] += 1
                keep.append(r)
        rows = keep
    elif a.limit:
        rows = rows[: a.limit]
    if a.controller == "typesafe":
        key = api_key()
        if not key:
            raise SystemExit("TYPESAFE_API_KEY not found in env or ../.env")
        ctl = HTTPController("https://api.typesafe.ai", api_key=key, model="jev-latest", timeout=60)
    else:
        ctl = HTTPController("http://127.0.0.1:8000", timeout=600)
    workers = a.workers or (4 if a.controller == "typesafe" else 1)

    def one(r: dict) -> dict:
        t = time.time()
        ans = ctl.ask(r["text"], {"q": QUESTION})["q"]
        return {"pred": ans["choice"], "p": ans["probabilities"][ans["choice"]], "s": time.time() - t}

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    wall = time.time() - t0

    ok = [r["pred"] == LABELS[row["label"]] for r, row in zip(res, rows)]
    k, n = sum(ok), len(ok)
    lo, hi = wilson(k, n)
    print(f"controller={a.controller} n={n} accuracy={k / n:.3f} (95% CI {lo:.3f}-{hi:.3f}) "
          f"ECE={ece([r['p'] for r in res], ok):.3f} mean_conf={sum(r['p'] for r in res) / n:.3f} "
          f"latency={sum(r['s'] for r in res) / n:.2f}s/row wall={wall:.0f}s")
    for lab in LABELS:
        idx = [i for i, row in enumerate(rows) if LABELS[row["label"]] == lab]
        if idx:
            print(f"  {lab:9s} {sum(ok[i] for i in idx) / len(idx):.3f} ({len(idx)})")
    if a.out:
        json.dump([{**r, "gold": LABELS[row["label"]]} for r, row in zip(res, rows)], open(a.out, "w"))


if __name__ == "__main__":
    main()
