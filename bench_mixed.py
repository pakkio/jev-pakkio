"""100 mixed questions with real labels: 40 choice, 30 noul, 30 score, run on every engine in-process.

Sources: AG News topics (choice, and "is it sports?" noul), langid (choice, and "is it written in X?" noul),
BBC sections (choice), IMDb rating band from title/year/genres/votes (score, so it tests film knowledge).

    uv run python bench_mixed.py                      # all engines
    BENCH_ENGINES=jev,4g uv run python bench_mixed.py
    BENCH_OUT=runs/mixed.json uv run python bench_mixed.py   # also keep per-question grades

jev and mercury need TYPESAFE_API_KEY / OPENROUTER_API_KEY (environment or ../.env). 4g/8g use the LoRA adapters
named in OPENJEV_LORA_4G / OPENJEV_LORA_8G, if set, exactly as the MCP server does.
"""
import json, os, random, statistics as st, time

from jev_pakkio import mcp_engines as m

R = random.Random(7)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
def jl(p): return [json.loads(l) for l in open(os.path.join(D, p))]

Q = []  # dict(src, type, state, q, exp)
# ---- choice: agnews topic (15), langid (15), bbc section (10)
AG = ["World", "Sports", "Business", "SciTech"]
AGC = {"World": "International news, politics, conflicts", "Sports": "Sports news", "Business": "Companies, markets, economy",
       "SciTech": "Science and technology"}
ag = jl("news/agnews_test_1000.jsonl")
by = {i: [r for r in ag if r["label"] == i] for i in range(4)}
pick = [r for i in range(4) for r in R.sample(by[i], 4)][:15]
for r in pick:
    Q.append(dict(src="agnews-topic", type="choice", state=r["text"],
                  q={"type": "choice", "instructions": "Which section of a newspaper does this article belong to?", "criteria": AGC},
                  exp=AG[r["label"]]))
lg = jl("langid/test.jsonl")
for r in R.sample(lg, 15):
    opts = r["options"]
    Q.append(dict(src="langid", type="choice", state=r["context"],
                  q={"type": "choice", "instructions": "Which language is this text written in?", "criteria": {o: f"Written in {o}" for o in opts}},
                  exp=opts[r["label"]]))
bb = [r for r in jl("bbc/2026-09-26.jsonl") if r["label"] >= 0]
for r in R.sample(bb, 10):
    opts = r["options"]
    Q.append(dict(src="bbc-section", type="choice", state=r["context"].split("\n", 1)[1],
                  q={"type": "choice", "instructions": "Which newspaper section does this story belong to?", "criteria": {o: f"{o} section" for o in opts}},
                  exp=opts[r["label"]]))
# ---- noul: agnews "is it about sports" (15, 8 true), langid "is it written in X" (15, 8 true)
sp = R.sample(by[1], 8); non = R.sample([r for r in ag if r["label"] != 1], 7)
for r in sp + non:
    Q.append(dict(src="agnews-sports?", type="noul", state=r["text"],
                  q={"type": "noul", "instructions": "Is this article about sports?"}, exp=r["label"] == 1))
for i, r in enumerate(R.sample(lg, 15)):
    opts = r["options"]; true_lang = opts[r["label"]]
    if i < 8: asked, exp = true_lang, True
    else: asked, exp = R.choice([o for o in opts if o != true_lang]), False
    Q.append(dict(src="langid-is?", type="noul", state=r["context"],
                  q={"type": "noul", "instructions": f"Is this text written in {asked}?"}, exp=exp))
# ---- score: imdb rating band from title/year/genres/votes (30); exp = clip(rating-4.5, 0, 4)
LV = ["poor (below 5 on IMDb)", "mediocre (about 5.5)", "decent (about 6.5)", "good (about 7.5)", "excellent (8.5 or more)"]
for r in R.sample(jl("imdb/movies100.jsonl"), 30):
    st_ = f"Film: {r['title']} ({r['year']}), {', '.join(r['genres'])}, {r['runtime_minutes']} minutes, {r['votes']} votes on IMDb."
    Q.append(dict(src="imdb-rating", type="score", state=st_,
                  q={"type": "score", "instructions": "How highly is this film rated by audiences on IMDb?", "criteria": LV},
                  exp=min(4.0, max(0.0, r["rating"] - 4.5))))
assert len(Q) == 100, len(Q)
R.shuffle(Q)

def grade(c, a):
    if c["type"] == "choice": return 1.0 if a.choice == c["exp"] else 0.0
    if c["type"] == "noul": return 1.0 if (a.noul > 0.5) == c["exp"] else 0.0
    return max(0.0, 1 - abs(a.score - c["exp"]) / 4)

ENGINES = os.environ.get("BENCH_ENGINES", "jev,mercury,laya,4g,8g").split(",")
out = {}
for name in ENGINES:
    t0 = time.perf_counter(); eng = m._engine(name); load = time.perf_counter() - t0
    rows = []
    for i, c in enumerate(Q):
        req = m._request(c["state"], {"q": c["q"]}); t = time.perf_counter()
        try:
            a = eng.answer(req).answers["q"]; g = grade(c, a); err = None
        except Exception as e:
            g, err = 0.0, f"{type(e).__name__}: {str(e)[:80]}"
        rows.append(dict(src=c["src"], type=c["type"], g=g, lat=time.perf_counter() - t, err=err))
    eng = None
    out[name] = dict(rows=rows, load=load)
    print(name, "done", f"load {load:.1f}s", "errors", sum(1 for r in rows if r["err"]), flush=True)
    if os.environ.get("BENCH_OUT"):
        json.dump(out, open(os.environ["BENCH_OUT"], "w"))

f = lambda rs: st.mean(r["g"] for r in rs)
print(f"\n{'engine':8} {'choice(40)':10} {'noul(30)':9} {'score(30)':10} {'AVG':6} {'err':4} {'lat med':8} {'lat total'}")
res = []
for n, d in out.items():
    r = d["rows"]
    c, nl, sc = (f([x for x in r if x["type"] == t]) for t in ("choice", "noul", "score"))
    avg = f(r); res.append((avg, n))
    print(f"{n:8} {c:10.3f} {nl:9.3f} {sc:10.3f} {avg:6.3f} {sum(1 for x in r if x['err']):4d} {st.median(x['lat'] for x in r):7.2f}s {sum(x['lat'] for x in r):7.1f}s")
print("\nby source (avg grade):")
srcs = sorted({q["src"] for q in Q})
print(f"{'source':16}" + "".join(f"{n:>9}" for n in out))
for s in srcs:
    print(f"{s:16}" + "".join(f"{f([x for x in d['rows'] if x['src'] == s]):9.2f}" for d in out.values()))
print("BEST:", max(res)[1])
