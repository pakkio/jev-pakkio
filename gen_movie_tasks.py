"""Score (1-5 stars) + noul (yes/no) synthetic sets for SystemOne-style heads."""
import json, random

FILMS = {
    "Solaris Drift": "sci-fi", "Neon Harbour": "cyberpunk", "The Granite Meadow": "drama",
    "Cipher of Lanterns": "mystery", "Velvet Orbit": "romance", "Quiet Ledger": "thriller",
    "Stone River Pact": "western", "Cloud Engine": "adventure", "Midnight Market": "comedy",
    "Iron Tide": "war",
}
FAMILY = {
    "sci-fi": "space", "cyberpunk": "space", "adventure": "space",
    "drama": "heart", "romance": "heart", "western": "heart",
    "mystery": "edge", "thriller": "edge", "war": "edge",
    "comedy": "fun",
}
LEVELS = [
    "1 star: hated it, wants a refund",
    "2 stars: disappointed",
    "3 stars: decent, mixed feelings",
    "4 stars: liked it",
    "5 stars: loved it, an instant favourite",
]

def score_label(taste, film):
    if FILMS[film] == taste:
        return 4
    if FAMILY[FILMS[film]] == FAMILY[taste]:
        return 2
    return 0

def make_score(rng, target_class=None):
    taste = rng.choice(sorted(set(FILMS.values())))
    if target_class == 4:  # exact genre hit
        cands = [f for f, g in FILMS.items() if g == taste]
        film = rng.choice(cands)
    elif target_class == 2:  # family hit, genre miss
        cands = [f for f, g in FILMS.items() if g != taste and FAMILY[g] == FAMILY[taste]]
        film = rng.choice(cands) if cands else rng.choice(sorted(FILMS))
    elif target_class == 0:  # family miss
        cands = [f for f, g in FILMS.items() if FAMILY[g] != FAMILY[taste]]
        film = rng.choice(cands)
    else:
        film = rng.choice(sorted(FILMS))
    label = score_label(taste, film)
    return {
        "context": f"Viewer loves {taste} films. Film watched: {film} ({FILMS[film]}). Notes: {rng.choice(['loved the pacing', 'found it slow', 'great ending', 'weak script'])}. Rate the viewing.",
        "options": LEVELS,
        "label": label,
    }

REFUND_YES = ["I want my money back", "this was a waste, refund please", "demanding a refund for this film"]
REFUND_NO = ["what a great evening", "loved every minute", "will recommend to friends"]

def make_noul(rng):
    yes = rng.random() < 0.5
    quote = rng.choice(REFUND_YES if yes else REFUND_NO)
    return {
        "context": f'Viewer review: "{quote}." Film: {rng.choice(sorted(FILMS))}. Does the viewer ask for a refund?',
        "options": ["yes", "no"],
        "label": 0 if yes else 1,
    }

rng = random.Random(99)
cls_cycle = [4, 2, 0]
ci = 0
def next_score(rng):
    global ci
    c = cls_cycle[ci % len(cls_cycle)]
    ci += 1
    return make_score(rng, c)
for name, fn, splits in [("movie_score", next_score, [("train", 210), ("validation", 30), ("test", 30)]),
                         ("movie_noul", make_noul, [("train", 120), ("validation", 20), ("test", 20)])]:
    for split, n in splits:
        with open(f"data/synthetic/{name}/{split}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps(fn(rng), ensure_ascii=False) + "\n")
    print(f"{name}: {sum(n for _, n in splits)} samples")
