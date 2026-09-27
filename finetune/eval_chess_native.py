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

    cache_file = "data/chess/test_jev_cache.json"
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)

    correct = 0
    total = 0

    with open("data/chess/test.jsonl") as f:
        lines = f.readlines()

    for line in lines:
        ex = json.loads(line)
        puzzle_id = ex["puzzle_id"] + "_" + str(ex.get("ply", 0))
        if puzzle_id in cache:
            ans = cache[puzzle_id]
        else:
            prompt = ex["prompt"]
            legal_moves = []
            for pl in prompt.splitlines():
                if pl.startswith("Legal moves: "):
                    legal_moves = pl[len("Legal moves: "):].split()
                    break
            
            # TypeSafe limits choice keys to 64 chars, and up to 255 options.
            # Legal moves are 4-5 chars, so this is fine.
            req = {
                "state": prompt,
                "model": "jev-latest",
                "questions": {
                    "move": {
                        "type": "choice",
                        "instructions": "Pick the best move for the side to move.",
                        "criteria": {m: f"Move {m}" for m in legal_moves}
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
                ans = data["answers"]["move"]["choice"]
            except Exception as e:
                print("Exception:", e)
                time.sleep(1)
                continue
                
            cache[puzzle_id] = ans
            with open(cache_file, "w") as out_f:
                json.dump(cache, out_f)
            time.sleep(0.2)
        
        target = ex["completion"].strip()
        if ans == target:
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"Tested {total}, native Jev accuracy: {correct/total:.2%}")
    
    if total > 0:
        print(f"Final accuracy on {total} test puzzles: {correct/total:.2%} ({correct}/{total})")

if __name__ == "__main__":
    main()
