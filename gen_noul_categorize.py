"""Noul (yes/no) and categorize (10 genres) sets that test generalisation to unseen phrasings.

noul_general: "Does the viewer ask for a refund?" Reviews are composed from phrase pools
(complaints, praise, refund requests, explicit no-refund remarks). Every pool is split 60/20/20
into train/validation/test pieces, so held-out reviews are built from phrases never seen in
training. Hard cases are included: complaints without a refund request, and praise that
mentions a refund only to rule it out.

categorize_general: a film description -> one of the 10 genres, reusing the hint split of
gen_movies_general.py (train hints include EXTRA_TRAIN; validation/test hints are held out).

usage: gen_noul_categorize.py [N_TRAIN] [N_VAL] [N_TEST] [SEED]
"""
import json, os, random, sys

from gen_movies_general import EXTRA_TRAIN, FILMS, SPLITS

GENRE = {
    "Solaris Drift": "science fiction", "Neon Harbour": "cyberpunk", "The Granite Meadow": "family drama",
    "Cipher of Lanterns": "mystery", "Velvet Orbit": "romance", "Quiet Ledger": "thriller",
    "Stone River Pact": "western", "Cloud Engine": "adventure", "Midnight Market": "comedy", "Iron Tide": "war",
}

COMPLAINTS = [
    "the plot made no sense", "I fell asleep halfway through", "the acting was wooden", "the sound kept cutting out",
    "it was two hours of my life I won't get back", "the ending ruined everything", "the dialogue was painful",
    "the picture was blurry the whole time", "it was nothing like the trailer", "the pacing was unbearably slow",
    "the jokes all fell flat", "the special effects looked cheap", "the story went nowhere", "I walked out after an hour",
    "the characters were impossible to care about", "it was boring from start to finish", "the subtitles were out of sync",
    "the screening started forty minutes late", "the seats were broken and the film was worse", "it felt like a rough draft",
    "the twist was obvious from the first scene", "the music drowned out the actors", "it was a mess", "the stream kept freezing",
    "the whole thing was a letdown",
]
PRAISE = [
    "what a great evening", "loved every minute", "will recommend to friends", "the cast was brilliant",
    "I laughed the whole way through", "the soundtrack was gorgeous", "best film I've seen this year", "the ending gave me chills",
    "absolutely worth the ticket", "the visuals were stunning", "I'd happily watch it again", "a lovely night out",
    "the story kept me hooked", "the director nailed it", "my kids adored it", "a real crowd-pleaser",
    "the performances were moving", "it exceeded my expectations", "a perfect date movie", "it was pure joy",
    "every scene was beautifully shot", "I was on the edge of my seat", "the script was sharp and funny", "a genuine masterpiece",
    "it made my week",
]
REFUND_ASK = [
    "I want my money back", "refund please", "I'm demanding a refund", "please return what I paid",
    "I expect a full refund", "give me my money back", "I'd like to be reimbursed", "charge it back to my card",
    "I want the ticket price returned", "please process a refund", "I'm asking for my money back", "credit my account for this",
    "send me a refund today", "reimburse me for this rental", "I want compensation for the ticket",
    "cancel the charge and refund me", "I paid for nothing, refund it", "I need my money returned", "return my payment",
    "I'd like my purchase refunded",
]
NO_REFUND = [
    "no need for a refund", "I don't want my money back", "not asking for a refund", "keep the money, it was worth it",
    "I won't be requesting a refund", "money well spent", "no complaints about the price", "I'm not after a refund",
    "worth every penny", "happy with what I paid", "I'd pay for it twice", "no refund necessary",
    "I'm keeping my ticket as a souvenir", "not looking for compensation", "the price was fair", "I don't regret paying",
    "no reimbursement needed", "totally worth the money", "I wouldn't dream of asking for a refund", "fair value for the ticket",
]
REVIEW_TEMPLATES = [
    'Viewer review: "{text}." Film: {film}. Does the viewer ask for a refund?',
    'Customer message about {film}: "{text}." Is this customer requesting a refund?',
    'Feedback received for {film}: "{text}." Does the writer want their money back?',
]
CATEGORY_TEMPLATES = [
    "Film description: {hint}. Which genre is this film?",
    "Classify this movie into one genre: {hint}.",
    "A catalogue entry reads: {hint}. Assign the genre.",
]


def split_pool(pool, split):
    n = len(pool)
    a, b = int(n * 0.6), int(n * 0.8)
    return {"train": pool[:a], "validation": pool[a:b], "test": pool[b:]}[split]


def make_noul(rng, split):
    pick = lambda pool: rng.choice(split_pool(pool, split))
    yes = rng.random() < 0.5
    if yes:  # a refund request, usually with a complaint
        parts = [pick(REFUND_ASK)] if rng.random() < 0.2 else [pick(COMPLAINTS), pick(REFUND_ASK)]
    else:
        kind = rng.choice(["praise", "praise_no_refund", "complaint_only", "complaint_no_refund"])
        parts = {
            "praise": [pick(PRAISE)],
            "praise_no_refund": [pick(PRAISE), pick(NO_REFUND)],
            "complaint_only": [pick(COMPLAINTS)],  # hard: unhappy, but no request
            "complaint_no_refund": [pick(COMPLAINTS), pick(NO_REFUND)],
        }[kind]
    if len(parts) == 2 and rng.random() < 0.5:
        parts.reverse()
    text = ", ".join(parts)
    context = rng.choice(REVIEW_TEMPLATES).format(text=text[0].upper() + text[1:], film=rng.choice(list(FILMS)))
    return {"context": context, "options": ["yes", "no"], "label": 0 if yes else 1}


def make_category(rng, split):
    film = rng.choice(list(FILMS))
    pool = FILMS[film][SPLITS[split]] + (EXTRA_TRAIN[film] if split == "train" else [])
    hint = rng.choice(pool)
    opts = list(GENRE.values())
    rng.shuffle(opts)
    context = rng.choice(CATEGORY_TEMPLATES).format(hint=hint)
    return {"context": context, "options": opts, "label": opts.index(GENRE[film]), "genre": GENRE[film], "hint": hint}


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:4]] or [2000, 300, 300]
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 21)
    for name, fn in [("noul_general", make_noul), ("categorize_general", make_category)]:
        out = f"data/synthetic/{name}"
        os.makedirs(out, exist_ok=True)
        for split, n in zip(SPLITS, sizes):
            with open(f"{out}/{split}.jsonl", "w") as f:
                for _ in range(n):
                    f.write(json.dumps(fn(rng, split), ensure_ascii=False) + "\n")
        print(f"{name}: {sizes}")
