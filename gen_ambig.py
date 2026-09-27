"""Generate an ambiguous badge dataset with multi-label ground truth and certainty index."""
import json, random, math

COLORS = ["coral", "azure", "crimson", "amber", "indigo", "bronze", "gold", "green", "silver", "violet"]
ANIMALS = ["ibis", "falcon", "jaguar", "badger", "crane", "dolphin", "heron", "gecko", "otter", "raven"]
BADGES = [f"{c} {a}" for c in COLORS for a in ANIMALS]

# Semantic groups for ambiguity
BIRDS = {"ibis", "falcon", "crane", "heron", "raven"}
PREDATORS = {"jaguar", "falcon", "raven"}
WATER = {"dolphin", "ibis", "heron", "otter"}
BRIGHT = {"coral", "amber", "gold", "crimson"}
COOL = {"azure", "indigo", "silver", "violet"}

def notes():
    return " ".join(random.choice(["north", "south", "east", "west"]) for _ in range(random.randint(6, 10)))

def ambiguity_type(rng):
    """Pick a semantic dimension that creates multiple valid answers."""
    kind = rng.choice(["bird", "predator", "water", "bright", "cool", "color", "animal"])
    if kind == "bird":
        valid_animals = list(BIRDS)
        prompt = "Choose a badge that is a bird"
    elif kind == "predator":
        valid_animals = list(PREDATORS)
        prompt = "Choose a badge that is a predator"
    elif kind == "water":
        valid_animals = list(WATER)
        prompt = "Choose a badge that is a water creature"
    elif kind == "bright":
        valid_colors = list(BRIGHT)
        prompt = "Choose a badge that has a warm bright color"
        valid = [b for b in BADGES if b.split()[0] in valid_colors]
        return prompt, valid
    elif kind == "cool":
        valid_colors = list(COOL)
        prompt = "Choose a badge that has a cool color"
        valid = [b for b in BADGES if b.split()[0] in valid_colors]
        return prompt, valid
    elif kind == "color":
        color = rng.choice(COLORS)
        prompt = f"Choose a badge that is the color {color}"
        valid = [b for b in BADGES if b.split()[0] == color]
        return prompt, valid
    else:  # animal
        animal = rng.choice(ANIMALS)
        prompt = f"Choose a badge that is a {animal}"
        valid = [b for b in BADGES if b.split()[1] == animal]
        return prompt, valid

    valid = [f"{c} {a}" for c in COLORS for a in valid_animals]
    return prompt, valid

def make_sample(rng, n_options):
    prompt, valid_set = ambiguity_type(rng)
    # Pick the correct answer(s) from valid set
    target = rng.sample(valid_set, min(rng.randint(1, 3), len(valid_set)))
    target = target[:2]  # max 2 valid answers for clarity

    opts = list(target)
    while len(opts) < n_options:
        d = rng.choice(BADGES)
        if d not in opts:
            opts.append(d)
    rng.shuffle(opts)
    labels = [opts.index(t) for t in target]

    return {
        "context": f"{prompt}. Notes: {notes()}.",
        "options": opts,
        "labels": labels,
        "ambiguity": len(target),
    }

rng = random.Random(123)
for split, n in [("train", 70), ("validation", 15), ("test", 15)]:
    with open(f"data/synthetic/badge_ambig/{split}.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps(make_sample(rng, random.randint(3, 8)), ensure_ascii=False) + "\n")
    print(f"{split}: {n} samples")
