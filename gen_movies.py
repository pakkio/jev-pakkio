"""Generate a 100-sample synthetic movie-rating dataset (10 films)."""
import json, random

FILMS = [
    ("Solaris Drift", "sci-fi"),
    ("Neon Harbour", "cyberpunk"),
    ("The Granite Meadow", "drama"),
    ("Cipher of Lanterns", "mystery"),
    ("Velvet Orbit", "romance"),
    ("Quiet Ledger", "thriller"),
    ("Stone River Pact", "western"),
    ("Cloud Engine", "adventure"),
    ("Midnight Market", "comedy"),
    ("Iron Tide", "war"),
]
TITLES = [t for t, _ in FILMS]
GENRE = dict(FILMS)

def notes():
    return " ".join(random.choice(["north", "south", "east", "west"]) for _ in range(random.randint(6, 10)))

def make_sample(rng, n_options):
    target = rng.choice(TITLES)
    genre = GENRE[target]
    stars = rng.randint(3, 5)  # synthetic rating: liked films are 3-5 stars
    opts = [target]
    while len(opts) < n_options:
        d = rng.choice(TITLES)
        if d not in opts:
            opts.append(d)
    rng.shuffle(opts)
    label = opts.index(target)
    return {
        "context": f"User likes {genre} films. Choose the exact film {target} rated {stars} stars. Notes: {notes()}. Film: {target}.",
        "options": opts,
        "label": label,
        "film": target,
        "rating": stars,
    }

rng = random.Random(1234)
for split, n in [("train", 70), ("validation", 15), ("test", 15)]:
    with open(f"data/synthetic/movies10/{split}.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps(make_sample(rng, random.randint(2, 8)), ensure_ascii=False) + "\n")
    print(f"{split}: {n} samples")
