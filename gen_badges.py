"""Generate a 100-sample badge classification dataset."""
import json, random

COLORS = ["coral", "azure", "crimson", "amber", "indigo", "bronze", "gold", "green", "silver", "violet"]
ANIMALS = ["ibis", "falcon", "jaguar", "badger", "crane", "dolphin", "heron", "gecko", "otter", "raven"]
BADGES = [f"{c} {a}" for c in COLORS for a in ANIMALS]

def notes():
    return " ".join(random.choice(["north", "south", "east", "west"]) for _ in range(random.randint(6, 10)))

def make_sample(rng, n_options):
    target = rng.choice(BADGES)
    opts = [target]
    while len(opts) < n_options:
        d = rng.choice(BADGES)
        if d not in opts:
            opts.append(d)
    rng.shuffle(opts)
    label = opts.index(target)
    return {
        "context": f"Choose the exact badge {target}. Notes: {notes()}. Badge: {target}.",
        "options": opts,
        "label": label,
    }

rng = random.Random(42)
for split, n in [("train", 70), ("validation", 15), ("test", 15)]:
    with open(f"data/synthetic/badge100/{split}.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps(make_sample(rng, random.randint(2, 8)), ensure_ascii=False) + "\n")
    print(f"{split}: {n} samples")
