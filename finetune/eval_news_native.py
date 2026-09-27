import json
import os
import time
import httpx

def get_api_key():
    with open("../.env") as f:
        for line in f:
            if line.startswith("TYPESAFE_API_KEY="):
                return line.strip().split("=", 1)[1]
    return None

def main():
    api_key = get_api_key()
    if not api_key:
        print("No TYPESAFE_API_KEY found")
        return

    cache_file = "data/news/test_jev_cache.json"
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)

    correct = 0
    total = 0

    with open("data/news/test.jsonl") as f:
        lines = f.readlines()

    # Limit to 500 tests to save time and API requests
    lines = lines[:500]

    for i, line in enumerate(lines):
        ex = json.loads(line)
        item_id = f"news_{i}"
        if item_id in cache:
            ans = cache[item_id]
        else:
            prompt = ex["prompt"]
            options = ["World", "Sports", "Business", "Sci/Tech"]
            
            req = {
                "state": prompt,
                "model": "jev-latest",
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": "Classify the news article into one of the options.",
                        "criteria": {m: m for m in options}
                    }
                }
            }
            try:
                resp = httpx.post(
                    "https://api.typesafe.ai/v1/systemone",
                    json=req,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0
                )
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}: {resp.text}")
                    time.sleep(1)
                    continue
                data = resp.json()
                ans = data["answers"]["category"]["choice"]
            except Exception as e:
                print("Exception:", e)
                time.sleep(1)
                continue
                
            cache[item_id] = ans
            with open(cache_file, "w") as out_f:
                json.dump(cache, out_f)
            time.sleep(0.1)
        
        target = ex["completion"].strip()
        if ans == target:
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"Tested {total}, native Jev accuracy: {correct/total:.2%}")
    
    if total > 0:
        print(f"Final accuracy on {total} test news: {correct/total:.2%} ({correct}/{total})")

if __name__ == "__main__":
    main()
