"""Jev-Mem vs flat top-k retrieval on a small hand-written multi-hop / temporal / causal set.

    uv run --no-sync python bench_mem.py --controller local                 # jev_pakkio serve on :8000
    uv run --no-sync python bench_mem.py --controller typesafe              # hosted Jev, key from env or ../.env
    uv run --no-sync python bench_mem.py --controller engine --engine laya  # in-process engine: jev, laya, 4g, 8g

Metric: gold-evidence recall@K and the share of questions whose gold set is fully retrieved, K small on
purpose so that flat top-K has to choose. Flat baseline = the same vector+BM25 RRF the memory uses for anchors.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

from jev_pakkio.mem import EngineController, HTTPController, JevMem, ReadConfig, SentenceTransformerEmbedder


def ts(month: int, day: int) -> float:
    return datetime(2024, month, day, 10, tzinfo=timezone.utc).timestamp()


TURNS = [  # id, (month, day), text
    ("a1", (5, 14), "Mira: My old bicycle broke."),
    ("a2", (5, 15), "Tom: I'm thinking of repainting the fence this weekend."),
    ("a3", (5, 16), "Mira: I bought a new bicycle yesterday because my old one broke."),
    ("a4", (5, 20), "Mira: The new bicycle is red and has a basket."),
    ("a5", (5, 22), "Tom: I finished the fence, it is now green."),
    ("a6", (5, 25), "Mira: I rode the bicycle to the Lisbon office for the first time."),
    ("b1", (6, 1), "Priya: I got laid off from Acme last week."),
    ("b2", (6, 3), "Priya: I started learning Rust in the evenings."),
    ("b3", (6, 10), "Dan: I prefer tea to coffee."),
    ("b4", (6, 18), "Priya: I accepted a job offer from Globex starting in July."),
    ("b5", (6, 18), "Priya: Globex was impressed by my Rust side project."),
    ("b6", (6, 25), "Dan: I adopted a cat named Miso."),
    ("b7", (7, 2), "Priya: My first day at Globex went well."),
    ("c1", (7, 5), "Dan: Miso knocked a glass off the table and broke it."),
    ("c2", (7, 6), "Tom: The Lisbon weather has been rainy all week."),
]
QUESTIONS = [  # question, gold ids
    ("When did Mira buy a new bicycle, and why?", {"a3", "a1"}),
    ("Why did Priya get her job at Globex?", {"b4", "b5", "b2"}),
    ("What color is Mira's new bicycle?", {"a4"}),
    ("Where did Mira ride her bicycle for the first time?", {"a6"}),
    ("What did Dan adopt and what is it called?", {"b6"}),
    ("What does Dan prefer to drink?", {"b3"}),
    ("What did Priya do after being laid off from Acme?", {"b1", "b2", "b4"}),
    ("What happened to Tom's fence?", {"a2", "a5"}),
    ("Which company did Priya leave and which did she join?", {"b1", "b4"}),
    ("Did Mira's bicycle break before she first rode to Lisbon?", {"a1", "a6"}),
]


def api_key() -> str | None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def score(retrieved: list[str], gold: set[str]) -> tuple[float, bool]:
    hit = len(gold & set(retrieved))
    return hit / len(gold), hit == len(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=("local", "typesafe", "engine"), required=True)
    ap.add_argument("--engine", default="jev", help="with --controller engine: jev, laya, 4g or 8g (in-process)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    label = a.controller + (":" + a.engine if a.controller == "engine" else "")
    if a.controller == "engine":
        from jev_pakkio.engines import get_engine

        ctl = EngineController(get_engine(a.engine))
    elif a.controller == "typesafe":
        key = api_key()
        if not key:
            raise SystemExit("TYPESAFE_API_KEY not found in env or ../.env")
        ctl = HTTPController(a.url or "https://api.typesafe.ai", api_key=key, model=a.model or "jev-latest", timeout=60)
    else:
        ctl = HTTPController(a.url or "http://127.0.0.1:8000", timeout=600)

    mem = JevMem(ctl, SentenceTransformerEmbedder(), read_cfg=ReadConfig(top_k=a.k))
    t0 = time.time()
    for nid, (m, d), text in TURNS:
        mem.add(text, timestamp=ts(m, d), node_id=nid)
    build_s, build_calls = time.time() - t0, ctl.calls
    print(f"controller={label} K={a.k}  build {build_s:.1f}s, {build_calls} calls, "
          f"{len(mem.store)} nodes, {len(mem.store.edges)} edges")

    rows = []
    for q, gold in QUESTIONS:
        flat = [i for i, _ in mem.store.rrf(q, a.k)]
        t = time.time()
        res = mem.query(q)
        dt = time.time() - t
        jm = [n.id for n, _ in res.evidence]
        rows.append((q, gold, flat, jm, dt, res.trace["controller_calls"], res.trace["stop"]))
        print(f"- {q}\n    gold={sorted(gold)} flat={flat} jevmem={jm} stop={res.trace['stop']} "
              f"calls={res.trace['controller_calls']} {dt:.1f}s")

    n = len(rows)
    for name, idx in (("flat top-K", 2), ("Jev-Mem", 3)):
        rec = sum(score(r[idx], r[1])[0] for r in rows) / n
        full = sum(score(r[idx], r[1])[1] for r in rows) / n
        print(f"{name:11s} recall@{a.k}={rec:.3f}  full-gold={full:.2f}")
    print(f"Jev-Mem avg query {sum(r[4] for r in rows) / n:.1f}s, {sum(r[5] for r in rows) / n:.1f} controller calls")


if __name__ == "__main__":
    main()
