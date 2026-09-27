import json
import time
import httpx
import torch
from openjev import OptionScorer

def get_api_key():
    with open("../.env") as f:
        for line in f:
            if line.startswith("TYPESAFE_API_KEY="):
                return line.strip().split("=", 1)[1]
    return None

def main():
    api_key = get_api_key()
    
    with open("bbc_test.jsonl") as f:
        lines = [json.loads(line) for line in f.readlines()][:10]
        
    print("Testing Native JEV Speed (10 articles)...")
    start = time.time()
    for ex in lines:
        req = {
            "state": ex["prompt"],
            "model": "jev-latest",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Classify the news article into one of the options.",
                    "criteria": {m: m for m in ["World", "Sports", "Business", "Sci/Tech"]}
                }
            }
        }
        t0 = time.time()
        resp = httpx.post(
            "https://api.typesafe.ai/v1/systemone",
            json=req,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )
        t1 = time.time()
        print(f"  Article took {t1-t0:.2f}s")
    total_native = time.time() - start
    print(f"Total Native JEV time: {total_native:.2f}s (Avg: {total_native/10:.2f}s/article)")

    print("\nLoading Local LoRA (gemma-2-2b-it + checkpoint-300)...")
    scorer = OptionScorer(
        "google/gemma-2-2b-it", 
        batch_size=16, 
        adapter_path="adapters/news-lora-2b/checkpoint-300"
    )
    
    print("\nTesting Local LoRA Speed (10 articles)...")
    start = time.time()
    for ex in lines:
        t0 = time.time()
        options = ["World", "Sports", "Business", "Sci/Tech"]
        res = scorer.score(ex["prompt"], [f" {opt}" for opt in options])
        t1 = time.time()
        print(f"  Article took {t1-t0:.2f}s")
    total_local = time.time() - start
    print(f"Total Local LoRA time: {total_local:.2f}s (Avg: {total_local/10:.2f}s/article)")

if __name__ == "__main__":
    main()
