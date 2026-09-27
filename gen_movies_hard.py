"""Hard movie-choice dataset: NO title leak in context, semantic genre match required."""
import json, os, random, sys

FILMS = {
    "Solaris Drift": ["deep-space exploration", "a lonely voyage beyond the heliopause", "generation-ship odyssey"],
    "Neon Harbour": ["neon-soaked hacker underworld", "rainy cyberpunk megacity", "black-market android dealers"],
    "The Granite Meadow": ["rural family drama", "a widower rebuilding his farm", "quiet village reconciliation"],
    "Cipher of Lanterns": ["locked-room mystery", "a detective chasing paper-lantern clues", "whodunit in a mountain inn"],
    "Velvet Orbit": ["forbidden love on a station", "romance between rival pilots", "a wedding above the atmosphere"],
    "Quiet Ledger": ["corporate conspiracy thriller", "an accountant uncovering bribes", "paranoid wiretap chase"],
    "Stone River Pact": ["frontier western showdown", "a sheriff defending a river crossing", "duel at high noon"],
    "Cloud Engine": ["high-altitude adventure", "airship expedition over glaciers", "daring mountain rescue flight"],
    "Midnight Market": ["night-bazaar comedy", "bickering merchants after dark", "farce of mistaken stalls"],
    "Iron Tide": ["naval war epic", "a destroyer crew under fire", "beach-landing sacrifice"],
}
TITLES = list(FILMS)

def notes(rng):
    return " ".join(rng.choice(["north", "south", "east", "west"]) for _ in range(rng.randint(6, 10)))

def make_sample(rng, n_options):
    target = rng.choice(TITLES)
    hint = rng.choice(FILMS[target])
    opts = [target]
    while len(opts) < n_options:
        d = rng.choice(TITLES)
        if d not in opts:
            opts.append(d)
    rng.shuffle(opts)
    return {
        "context": f"A viewer asks for {hint}. Notes: {notes(rng)}. Recommend exactly one film from the list.",
        "options": opts,
        "label": opts.index(target),
        "film": target,
    }

# usage: gen_movies_hard.py [OUT_DIR] [N_TRAIN] [N_VAL] [N_TEST] [SEED]
out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/movies_hard"
sizes = [int(x) for x in sys.argv[2:5]] or [200, 30, 30]
rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 7)
os.makedirs(out, exist_ok=True)
for split, n in zip(["train", "validation", "test"], sizes):
    with open(f"{out}/{split}.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps(make_sample(rng, rng.randint(2, 8)), ensure_ascii=False) + "\n")
    print(f"{split}: {n} samples")
