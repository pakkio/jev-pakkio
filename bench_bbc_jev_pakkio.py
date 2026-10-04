"""
jev_pakkio classification on bbc_test.jsonl, torch backend on CUDA, with the
fine-tuned news-lora-2b adapter (scorer_torch.OptionScorer now supports
adapter_path after the fix in this session).
"""
import json, time
from jev_pakkio.scorer_torch import OptionScorer

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

examples = [json.loads(l) for l in open("bbc_test.jsonl")]

scorer = OptionScorer("models/gemma-2-2b-it", adapter_path="adapters/news-lora-2b/checkpoint-300")

correct = 0
t0 = time.time()
results = []
for ex in examples:
    scores = scorer.score(ex["prompt"], [f" {c}" for c in CATEGORIES], norm="sum")
    ordered = sorted(scores, key=lambda s: s.probability, reverse=True)
    pred = ordered[0].option.strip()
    ok = pred == ex["label"]
    correct += ok
    results.append({"label": ex["label"], "pred": pred, "ok": ok,
                     "probs": {s.option.strip(): s.probability for s in scores}})
total = time.time() - t0

print(f"jev_pakkio (gemma-2-2b-it + news-lora-2b/checkpoint-300, torch/cuda): {correct}/{len(examples)} = {correct/len(examples):.1%} accuracy")
print(f"Total time: {total:.1f}s ({total/len(examples):.2f}s/article)")

with open("bbc_jev_pakkio_lora_results.json", "w") as f:
    json.dump({"accuracy": correct/len(examples), "total_s": total, "per_article_s": total/len(examples), "results": results}, f, indent=2)
