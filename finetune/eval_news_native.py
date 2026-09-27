"""Evaluate zero-shot Jev on the AG News test slice and/or the BBC set.

Unlike the original version, this caches the full answer object (choice +
confidence), not just the choice, so top-1 accuracy and ECE can both be
computed from one pass. Confidence is Jev's own calibration signal for a
`choice` question (see docs/jev-research-2026-09-20.md); ECE bins on it the
same way finetune/eval_news.py does for the local models.
"""

import argparse
import json
import os
import time

import httpx

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]


def get_api_key() -> str | None:
    for path in ("../.env", "/home/pakkio/w/.env", ".env"):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.startswith("TYPESAFE_API_KEY="):
                        return line.strip().split("=", 1)[1]
    return None


def run(data_path: str, cache_path: str, limit: int, api_key: str) -> dict:
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    with open(data_path) as f:
        lines = f.readlines()[:limit] if limit else f.readlines()

    correct = 0
    total = 0
    conf_bins = [[0, 0.0, 0.0] for _ in range(10)]

    for i, line in enumerate(lines):
        ex = json.loads(line)
        item_id = f"item_{i}"
        entry = cache.get(item_id)
        if entry is None or "confidence" not in entry:
            article = ex.get("text") or ex["prompt"].split("Article: ", 1)[1].split("\n\nOptions:")[0]
            req = {
                "state": article,
                "model": "jev-latest",
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": "Classify the news article into one of the options.",
                        "criteria": {m: m for m in CATEGORIES},
                    }
                },
            }
            try:
                resp = httpx.post(
                    "https://api.typesafe.ai/v1/systemone",
                    json=req,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}: {resp.text}")
                    time.sleep(1)
                    continue
                data = resp.json()
                ans = data["answers"]["category"]
                entry = {"choice": ans["choice"], "confidence": ans.get("confidence")}
            except Exception as e:
                print("Exception:", e)
                time.sleep(1)
                continue
            cache[item_id] = entry
            with open(cache_path, "w") as out_f:
                json.dump(cache, out_f)
            time.sleep(0.1)

        target = ex["completion"].strip()
        hit = entry["choice"] == target
        correct += hit
        total += 1
        conf = entry.get("confidence")
        if conf is not None:
            b = min(9, int(conf * 10))
            conf_bins[b][0] += 1
            conf_bins[b][1] += conf
            conf_bins[b][2] += hit
        if total % 50 == 0:
            print(f"Tested {total}, accuracy so far: {correct / total:.2%}")

    n = max(total, 1)
    ece = sum(c * abs(sc / c - sh / c) for c, sc, sh in conf_bins if c) / n
    return {"n": total, "top1": correct / n, "ece": ece}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/news/test.jsonl")
    ap.add_argument("--cache", default="data/news/test_jev_cache_v2.json")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("No TYPESAFE_API_KEY found")
        return

    result = run(args.data, args.cache, args.limit, api_key)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
