"""News-headline categorisation set that tests generalisation to unseen headlines.

Each category has a pool of subjects and a pool of events; a headline is "<subject> <event>".
Both pools are split 60/20/20 into train/validation/test pieces, so held-out headlines are built
entirely from subjects and events never seen in training.

usage: gen_news.py [OUT_DIR] [N_TRAIN] [N_VAL] [N_TEST] [SEED]
"""
import json, os, random, sys

NEWS = {
    "politics": (
        ["The prime minister", "The opposition leader", "Parliament", "The foreign minister", "The ruling coalition",
         "The senate", "The president", "A junior minister", "The regional governor", "The electoral commission",
         "The interior ministry", "Several MPs", "The cabinet", "The mayor of the capital", "The main opposition party"],
        ["calls a snap election", "survives a no-confidence vote", "unveils a plan to reform the constitution",
         "resigns after a coalition collapse", "passes a controversial immigration bill", "announces a cabinet reshuffle",
         "faces protests over new voting rules", "signs a treaty with neighbouring states", "blocks the government's budget",
         "launches its campaign for the referendum", "loses its parliamentary majority", "vetoes a law on public spending",
         "appoints a new ambassador", "wins the regional elections", "promises to cut the number of ministries"],
    ),
    "sports": (
        ["The national football team", "A veteran goalkeeper", "The tennis world number one", "The home side",
         "An Olympic sprinter", "The cycling team", "The basketball champions", "A young striker", "The rugby squad",
         "The Formula 1 leader", "The ski racer", "The boxing champion", "The volleyball team", "The marathon winner",
         "The club captain"],
        ["wins the cup final on penalties", "breaks the world record", "is ruled out for the season with a knee injury",
         "signs a three-year contract", "is knocked out in the semi-final", "clinches the league title",
         "scores a hat-trick in the derby", "retires after a twenty-year career", "is suspended for doping",
         "takes gold at the championships", "comes back from two sets down", "sacks the head coach after a losing streak",
         "qualifies for the World Cup", "wins the stage in the mountains", "defends the title for a third year"],
    ),
    "business": (
        ["The central bank", "A major retailer", "The car maker", "Shares in the airline", "The oil company",
         "The national stock index", "A supermarket chain", "The steel group", "Investors", "The shipping firm",
         "The insurance giant", "The fashion brand", "A family-owned brewery", "The logistics company", "Consumer prices"],
        ["reports record quarterly profits", "announces a merger worth billions", "cuts two thousand jobs",
         "raises interest rates by a quarter point", "falls sharply after a profit warning", "files for bankruptcy",
         "agrees a takeover by a rival", "opens a hundred new stores", "posts its first loss in a decade",
         "rises after strong sales figures", "is fined for price fixing", "plans to list on the stock exchange",
         "moves its headquarters abroad", "freezes wages for all staff", "beats analysts' earnings forecasts"],
    ),
    "technology": (
        ["The smartphone maker", "A software startup", "The social network", "The chip designer", "A cybersecurity firm",
         "The search engine", "The video game studio", "An AI lab", "The cloud provider", "The electric-vehicle software team",
         "The app developer", "The robotics company", "A satellite internet provider", "The browser team", "The messaging app"],
        ["releases an open-source language model", "unveils a foldable phone", "patches a critical security flaw",
         "launches a new operating system update", "suffers a global outage", "introduces end-to-end encryption",
         "debuts a faster processor", "rolls out a feature that summarises emails", "is hit by a massive data breach",
         "ships its first augmented-reality headset", "releases a new programming language", "doubles the battery life of its devices",
         "opens its API to developers", "switches to its own custom chips", "announces a quantum computing breakthrough"],
    ),
    "health": (
        ["Hospitals", "The health ministry", "Doctors", "A new study", "The World Health Organization",
         "Nurses", "Pharmacists", "The national vaccine programme", "Researchers at the medical school", "Family doctors",
         "Public health officials", "The children's hospital", "Cancer specialists", "Dentists", "The blood bank"],
        ["report a sharp rise in flu cases", "warn about rising obesity in children", "approve a new vaccine for older adults",
         "link daily walking to lower heart disease risk", "urge people to cut their salt intake", "extend waiting lists for surgery",
         "find a new treatment for diabetes", "declare the end of the measles outbreak", "call for more mental health funding",
         "launch a free screening programme", "report shortages of antibiotics", "recommend more sleep for teenagers",
         "reduce emergency room waiting times", "warn of a dengue fever outbreak", "publish new advice on alcohol"],
    ),
    "science": (
        ["Astronomers", "Physicists at the particle collider", "Palaeontologists", "Marine biologists", "Climate scientists",
         "The space agency", "Geologists", "Archaeologists", "Chemists", "Researchers in Antarctica",
         "A team of botanists", "Volcanologists", "Neuroscientists", "Oceanographers", "Zoologists"],
        ["discover a planet that could hold water", "find a new species of dinosaur", "detect a mysterious particle",
         "map the deepest part of the ocean", "measure the fastest melting of glaciers on record", "land a probe on an asteroid",
         "uncover a 3,000-year-old city", "identify the oldest known fossil", "observe the collision of two black holes",
         "sequence the genome of an ancient plant", "record a new kind of earthquake", "find microplastics in the Arctic ice",
         "create a new element in the laboratory", "photograph a galaxy from the early universe", "track the migration of whales by satellite"],
    ),
    "entertainment": (
        ["The pop star", "A Hollywood actor", "The streaming service", "The rock band", "The film festival",
         "A reality show", "The comedian", "The famous director", "The music awards", "The theatre company",
         "A TV presenter", "The boy band", "The fantasy series", "The rapper", "The opera singer"],
        ["announces a world tour", "wins the award for best actress", "releases a surprise album",
         "cancels the final season", "breaks box office records", "is cast in a superhero sequel",
         "tops the charts for ten weeks", "splits after twenty years together", "premieres a new documentary",
         "sells out the stadium in minutes", "hosts the red-carpet gala", "reunites for a farewell concert",
         "is nominated for an Oscar", "launches a new talent show", "releases the trailer for the sequel"],
    ),
    "crime": (
        ["Police", "Detectives", "A gang of thieves", "Prosecutors", "The anti-mafia unit",
         "Customs officers", "A jury", "Two suspects", "The fraud squad", "Border guards",
         "The court", "Firearms officers", "A drug cartel", "An escaped prisoner", "Investigators"],
        ["arrest a man over a jewellery robbery", "seize a ton of cocaine at the port", "charge a councillor with bribery",
         "find the stolen paintings in a warehouse", "convict the killer after a long trial", "break up a car theft ring",
         "raid a money-laundering network", "hunt for the bank robbers", "sentence the smuggler to ten years",
         "investigate a string of burglaries", "detain three people after a shooting", "uncover an online fraud scheme",
         "recover weapons hidden in a garage", "arrest a hacker who stole bank data", "open a murder inquiry"],
    ),
}
TEMPLATES = [
    "News headline: {h}. Which category is this news?",
    "Classify this headline: {h}.",
    "Today's news: {h}. Which section of the newspaper does it belong to?",
    "Breaking: {h}. Assign the news category.",
]
SPLITS = {"train": (0.0, 0.6), "validation": (0.6, 0.8), "test": (0.8, 1.0)}


def piece(pool, split):
    a, b = SPLITS[split]
    return pool[int(len(pool) * a):int(len(pool) * b)]


def make_news(rng, split):
    cat = rng.choice(list(NEWS))
    subjects, events = NEWS[cat]
    h = f"{rng.choice(piece(subjects, split))} {rng.choice(piece(events, split))}"
    opts = list(NEWS)
    rng.shuffle(opts)
    return {"context": rng.choice(TEMPLATES).format(h=h), "options": opts, "label": opts.index(cat),
            "category": cat, "headline": h}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/news_general"
    sizes = [int(x) for x in sys.argv[2:5]] or [2000, 300, 300]
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 31)
    os.makedirs(out, exist_ok=True)
    for split, n in zip(SPLITS, sizes):
        with open(f"{out}/{split}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps(make_news(rng, split), ensure_ascii=False) + "\n")
        print(f"{split}: {n} samples")
