"""Movie-choice dataset that tests generalisation: validation and test use genre hints never seen in training.

Each film has 12 hints; per film 8 go to train, 2 to validation, 2 to test. The request wording
varies across templates, so a model has to map the meaning of a hint to a title rather than
memorise a fixed phrase.

usage: gen_movies_general.py [OUT_DIR] [N_TRAIN] [N_VAL] [N_TEST] [SEED] [--rich]
--rich adds EXTRA_TRAIN (24 more train-only hints per film); validation/test hints are unchanged.
"""
import json, os, random, sys

FILMS = {
    "Solaris Drift": [
        "deep-space exploration", "a lonely voyage beyond the heliopause", "generation-ship odyssey",
        "astronauts drifting far from home", "first contact at the edge of the galaxy", "a crew frozen in cryosleep",
        "a probe sent to a dying star", "a mission to the outer planets", "stranded on a distant moon",
        "a slow-burning space voyage", "a pilot alone in orbit around Jupiter", "an interstellar ark losing its way",
    ],
    "Neon Harbour": [
        "neon-soaked hacker underworld", "rainy cyberpunk megacity", "black-market android dealers",
        "a netrunner breaking into a megacorp", "cybernetic implants sold in back alleys", "a rogue AI in the city grid",
        "holographic billboards and street gangs", "a hacker hunted by corporate enforcers", "augmented mercenaries at night",
        "a virtual heist in a digital city", "synthetic humans hiding among citizens", "a detective with a chrome arm",
    ],
    "The Granite Meadow": [
        "rural family drama", "a widower rebuilding his farm", "quiet village reconciliation",
        "siblings returning to the family homestead", "a harvest season that heals old wounds", "a small-town mother and daughter",
        "a farmer facing drought and debt", "grandparents teaching the land", "a gentle story about grief in the countryside",
        "a village gathering after a funeral", "a son taking over the family orchard", "slow life on a sheep farm",
    ],
    "Cipher of Lanterns": [
        "locked-room mystery", "a detective chasing paper-lantern clues", "whodunit in a mountain inn",
        "a murder at a snowed-in hotel", "an inspector questioning every guest", "a poisoning at a dinner party",
        "clues hidden in an old diary", "a secret code left by the victim", "a sleuth unmasking the killer",
        "a puzzle of alibis and footprints", "a missing heir and a stolen will", "a crime solved in the final chapter",
    ],
    "Velvet Orbit": [
        "forbidden love on a station", "romance between rival pilots", "a wedding above the atmosphere",
        "two strangers falling in love in zero gravity", "a love letter sent across the solar system", "star-crossed lovers on a space liner",
        "a romantic comedy on a cruise to Mars", "an engineer who falls for the captain", "a long-distance relationship between planets",
        "a heartfelt reunion on a space elevator", "a love triangle aboard a starship", "a honeymoon on the Moon",
    ],
    "Quiet Ledger": [
        "corporate conspiracy thriller", "an accountant uncovering bribes", "paranoid wiretap chase",
        "a whistleblower hiding from executives", "embezzlement buried in spreadsheets", "a journalist exposing a banking scandal",
        "insider trading and a cover-up", "a clerk who finds the secret books", "shell companies and money laundering",
        "a tense boardroom betrayal", "an auditor followed by men in suits", "political corruption and hidden accounts",
    ],
    "Stone River Pact": [
        "frontier western showdown", "a sheriff defending a river crossing", "duel at high noon",
        "cowboys and a saloon brawl", "outlaws robbing a stagecoach", "a bounty hunter riding across the desert",
        "cattle rustlers on the open range", "a gunslinger seeking revenge", "a dusty town with a corrupt marshal",
        "settlers against bandits on the prairie", "a posse chasing train robbers", "a lone ranger and his horse",
    ],
    "Cloud Engine": [
        "high-altitude adventure", "airship expedition over glaciers", "daring mountain rescue flight",
        "balloonists racing around the world", "sky pirates and floating cities", "a treasure hunt among the clouds",
        "a glider pilot crossing the Himalayas", "an explorer lost in a jungle canyon", "a daring escape by zeppelin",
        "a swashbuckling quest for a lost map", "adventurers climbing a volcano", "a bush pilot delivering supplies",
    ],
    "Midnight Market": [
        "night-bazaar comedy", "bickering merchants after dark", "farce of mistaken stalls",
        "a street vendor's hilarious bad luck", "a slapstick chase through a food market", "rival noodle stands in a prank war",
        "a comedy of errors among shopkeepers", "a clumsy thief at the night fair", "a silly mix-up over a lost parcel",
        "haggling that goes absurdly wrong", "neighbours feuding over a fruit stall", "a lighthearted romp through the bazaar",
    ],
    "Iron Tide": [
        "naval war epic", "a destroyer crew under fire", "beach-landing sacrifice",
        "a submarine hunted in the Atlantic", "sailors surviving a torpedo attack", "an aircraft carrier in a great sea battle",
        "marines storming a fortified island", "a battleship's last stand", "a convoy crossing hostile waters",
        "a wartime captain and his loyal crew", "soldiers waiting for evacuation on the shore", "a fleet clash in the Pacific",
    ],
}

# Extra train-only hints (v2): widen each genre, especially at the borders the held-out hints
# sit on -- space + romance vs space + solitude, cyberpunk detective vs classic whodunit,
# naval war vs sea adventure. None repeats a validation or test hint.
EXTRA_TRAIN = {
    "Solaris Drift": [
        "a science-fiction journey to another solar system", "a scientist alone on a research outpost in space",
        "a derelict spaceship found drifting", "an astronaut's isolation on a long mission", "a wormhole leading to the unknown",
        "a colony ship searching for a new home", "a space station at the edge of a black hole", "a crew studying an alien planet",
        "a rescue mission to a lost space probe", "cosmic mysteries and silent stars", "an expedition to the rings of Saturn",
        "a survival story on a barren planet", "a voyage through an asteroid field", "a meditative film about the cosmos",
        "a commander losing contact with Earth", "a hard sci-fi mission to Mars", "the last survivor on a space station",
        "a telescope crew discovering a signal", "a fuel crisis on a deep-space freighter", "a lonely robot exploring a comet",
        "a spacewalk gone wrong", "a starship crew facing the void", "a quiet odyssey among the planets", "an orbital lab cut off from Earth",
    ],
    "Neon Harbour": [
        "a hacker thriller in a dystopian future city", "cyborgs and street samurai", "a megacorporation controlling the city",
        "a courier carrying stolen data in her brain", "a noir detective in a futuristic city of rain", "robots hunted by a blade-wielding cop",
        "virtual reality addicts in the slums", "an android questioning its memories", "a cyber-criminal gang war",
        "brain implants hacked by terrorists", "a futuristic police state with drones", "a glitching AI companion",
        "hackers fighting a surveillance network", "a chrome-plated assassin", "a cybernetic cop investigating a murder in the sprawl",
        "illegal body modification clinics", "a data thief in a neon metropolis", "a dystopian tech noir",
        "a revolution against the corporate overlords", "a bartender who repairs cyborgs", "digital ghosts in the network",
        "a street kid with a hacked neural jack", "high tech and low life", "an investigator tracking a rogue replicant",
    ],
    "The Granite Meadow": [
        "a family reunion in the countryside", "a heartfelt drama about aging parents", "a farming community coming together",
        "a young woman returning to her hometown", "a quiet film about loss and healing", "a family bakery in a small village",
        "a father and son repairing their relationship", "a rural school teacher's life", "a tender drama about inheritance and family",
        "a countryside wedding with family tensions", "life in a small mountain village", "a mother caring for a sick child on the farm",
        "an old man and his vineyard", "brothers reconciling after years apart", "a slow-paced drama about rural life",
        "a family losing their land", "a grandmother's last summer", "a village doctor and his patients", "seasons changing on a dairy farm",
        "a small-town funeral bringing a family together", "a drama about forgiveness between sisters", "a shepherd's quiet life",
        "a family saving their barn from ruin", "a warm story of neighbours in a hamlet",
    ],
    "Cipher of Lanterns": [
        "a classic murder mystery", "a detective investigating a mansion full of suspects", "a whodunit on a train",
        "an amateur sleuth in a quiet village", "a mystery about a stolen necklace", "a crime puzzle with a surprising twist",
        "a private investigator solving a cold case", "a body found in the library", "an inspector following a trail of riddles",
        "a mystery novel brought to life", "a secret passage and a hidden culprit", "suspects gathered for the final reveal",
        "a detective deducing the murderer from tiny clues", "a missing painting and a clever thief", "a manor house murder",
        "an old-fashioned detective story", "clues and red herrings", "a mysterious letter predicting a death",
        "a sleuth cracking an impossible crime", "a disappearance at a country estate", "a detective and her loyal assistant",
        "a masked stranger and a hidden motive", "an investigation into a poisoned cup of tea", "a puzzle-box mystery",
    ],
    "Velvet Orbit": [
        "a romance set in outer space", "a love story between astronauts", "two lovers separated by a space mission",
        "a romantic drama on a starship", "a couple falling in love on a colony world", "a sci-fi love story",
        "a proposal on a space station", "romance between an android and a human in orbit", "a space pilot and a diplomat in love",
        "a romantic comedy aboard an orbital hotel", "sweethearts reunited after a voyage to the stars", "a love affair on a lunar base",
        "a passionate romance among the stars", "a couple's anniversary in zero gravity", "lovers writing messages across light-years",
        "a romantic getaway on a spaceship", "a forbidden romance between enemy crews", "a love story on a Mars colony",
        "a dance under Earthrise", "a bride and groom on an orbital cruise", "a romance between two space station engineers",
        "an interplanetary love letter", "a wedding on a starliner", "a tender romance in a space habitat",
    ],
    "Quiet Ledger": [
        "a financial crime thriller", "a banker discovering fraud", "a spy thriller inside a corporation",
        "a lawyer exposing a corrupt company", "hidden money and government secrets", "a paranoid thriller about surveillance",
        "an investigation into tax fraud", "a cover-up at a pharmaceutical giant", "blackmail in the business world",
        "a secretary who knows too much", "offshore accounts and bribed officials", "a thriller about stolen corporate files",
        "an executive framed for fraud", "a conspiracy reaching the top of the company", "a leak of confidential documents",
        "an insurance scam unravelling", "a reporter tailed by corporate spies", "a white-collar crime drama",
        "a mole inside the accounting department", "a whistleblower in witness protection", "phone taps and secret meetings",
        "a stock market manipulation scheme", "a thriller about a crooked senator and his donors", "a forensic accountant in danger",
    ],
    "Stone River Pact": [
        "a classic western", "a gunfight in a frontier town", "cowboys driving cattle across the plains",
        "a sheriff standing against a gang of outlaws", "a revenge story in the Wild West", "a bank robbery in the old west",
        "a wagon train heading west", "a marshal hunting a notorious bandit", "a showdown on main street",
        "a rancher defending his land", "horse riders crossing the desert", "a western about gold prospectors",
        "a lawman and a wanted outlaw", "saloon card games and gunslingers", "a frontier town under siege",
        "a cowboy seeking justice", "a stagecoach journey through dangerous territory", "outlaws hiding in the canyons",
        "a wanted poster and a reward", "a feud between two ranching families", "a deputy's last stand",
        "the railroad coming to a western town", "a gunslinger's final duel", "a dusty Wild West adventure",
    ],
    "Cloud Engine": [
        "an adventure film with airships", "explorers flying over uncharted lands", "a thrilling rescue in the mountains",
        "treasure hunters in the sky", "a daring flight across the ocean", "an adventure among floating islands",
        "an expedition to find a lost city", "a pilot's daring aerial adventure", "an action-packed quest in the Andes",
        "an adventurer and a hot air balloon", "a flight through a storm", "a high-flying adventure story",
        "a quest to reach the highest peak", "a biplane race across continents", "an airborne rescue from a crashed plane",
        "an exotic expedition through the rainforest", "sky sailors and hidden treasure", "an aviator crossing the Arctic",
        "an adventure with a map and a compass", "daring aviators in the golden age of flight", "a mountaineering expedition gone wrong",
        "a journey by airship to a forgotten land", "a thrilling escape from a volcano island", "explorers searching for an ancient relic",
    ],
    "Midnight Market": [
        "a comedy set in a night market", "a funny story about street vendors", "a madcap comedy of mistaken identity",
        "a lighthearted comedy after dark", "shopkeepers competing in a silly contest", "a hilarious night of chaos at the fair",
        "a comedy about a food stall", "a slapstick farce with market sellers", "bumbling merchants and a runaway goat",
        "a comedy about a bargain gone wrong", "a funny night of pranks among traders", "a goofy comedy in a busy bazaar",
        "a comic tale of two rival street cooks", "a feel-good comedy about a market family", "a ridiculous chase for a stolen melon",
        "a screwball comedy at a late-night bazaar", "a vendor who accidentally sells a treasure", "a comedy of errors after sunset",
        "a funny film about a chaotic market night", "merchants arguing over the best spot", "a silly caper among the stalls",
        "a sitcom-style comedy in a street market", "a lighthearted farce with a mischievous cat", "haggling comedy with absurd twists",
    ],
    "Iron Tide": [
        "a war film at sea", "a world war naval battle", "sailors fighting on a warship", "a submarine war thriller",
        "a navy crew in combat", "a D-Day style beach invasion", "a battleship sinking in battle", "a war drama about marines",
        "sailors lost at sea during the war", "a heroic navy captain", "a torpedoed ship and its survivors",
        "a naval blockade under fire", "a war epic about an amphibious assault", "gun crews on a burning destroyer",
        "a wartime evacuation by sea", "an admiral commanding a great fleet", "a navy pilot on an aircraft carrier",
        "a bloody battle for a beach", "a minesweeper crew in danger", "a war at sea film",
        "a sinking troop ship", "sailors trapped in a damaged submarine", "a naval war drama in the Atlantic", "a sea battle in the Second World War",
    ],
}

TEMPLATES = [
    "A viewer asks for {hint}. Notes: {notes}. Recommend exactly one film from the list.",
    "Tonight I want to watch something about {hint}. Notes: {notes}. Which film should I pick?",
    "Customer request: {hint}. Notes: {notes}. Choose the single best film.",
    "My friend loves stories with {hint}. Notes: {notes}. Suggest one film.",
    "Looking for a movie: {hint}. Notes: {notes}. Pick exactly one title.",
]
TITLES = list(FILMS)
SPLITS = {"train": slice(0, 8), "validation": slice(8, 10), "test": slice(10, 12)}


def notes(rng):
    return " ".join(rng.choice(["north", "south", "east", "west"]) for _ in range(rng.randint(6, 10)))


def make_sample(rng, split, rich=False):
    target = rng.choice(TITLES)
    pool = FILMS[target][SPLITS[split]]
    if split == "train" and rich:
        pool = pool + EXTRA_TRAIN[target]
    hint = rng.choice(pool)
    opts = [target] + rng.sample([t for t in TITLES if t != target], rng.randint(1, 7))
    rng.shuffle(opts)
    context = rng.choice(TEMPLATES).format(hint=hint, notes=notes(rng))
    return {"context": context, "options": opts, "label": opts.index(target), "film": target, "hint": hint}


if __name__ == "__main__":
    RICH = "--rich" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--rich"]
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/movies_general"
    sizes = [int(x) for x in sys.argv[2:5]] or [2000, 300, 300]
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 13)
    os.makedirs(out, exist_ok=True)
    for split, n in zip(SPLITS, sizes):
        with open(f"{out}/{split}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps(make_sample(rng, split, RICH), ensure_ascii=False) + "\n")
        print(f"{split}: {n} samples")
