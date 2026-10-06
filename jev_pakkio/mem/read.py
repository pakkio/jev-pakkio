"""Read path (paper 3.3): route -> retrieve -> assess -> expand -> reassess, all System-One controlled.

Per query: one routing call, then per round at most one evidence-assessment call and one batched
candidate-scoring call. Hard limits on rounds, candidates, controller calls and wall time bound the loop.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from .control import Controller, noul, p
from .store import GRAPHS, MemoryStore, Node
from .write import _view


@dataclass
class ReadConfig:
    budget: int = 80           # B, total graph-expansion budget
    min_budget: int = 1        # m, floor for each active graph
    gamma: float = 1.0         # allocation concentration
    theta_act: float = 0.10
    d_max: int = 3
    n_anchors: int = 8
    beam_w: int = 6            # W
    top_k: int = 10            # K, evidence handed to System Two
    theta_suff: float = 0.70
    theta_cont: float = 0.30
    max_candidates: int = 16   # per scoring call
    max_calls: int = 12
    max_seconds: float = 60.0
    # lambda_1..5 of eq 23: embedding sim, relevance, graph-need * usefulness, novelty, (edge weight + support)/2
    lambdas: tuple[float, ...] = (0.25, 0.25, 0.20, 0.15, 0.15)
    recency_weight: float = 0.1  # the 0.1 in eq 25


ROUTE_QUESTIONS = {
    "semantic": noul("Does answering query require finding memories on the same topic or fact?",
                     "Topical or factual similarity is needed to answer.", "The question is not about a particular topic or fact."),
    "temporal": noul("Does answering query require event dates, durations, ordering or changes over time?",
                     "A time relation is needed to answer correctly.", "Dates or ordering are incidental to the answer."),
    "causal": noul("Does answering query require explaining a cause, motivation, enabling condition or effect?",
                   "Causal or explanatory evidence is needed.", "Only factual association or chronology is requested."),
    "entity": noul("Does answering query require gathering what is known about specific people, objects or places?",
                   "Facts about one or more named entities are needed.", "No particular entity is central to the question."),
    "multi_hop": noul("Does answering query require combining several separate memories?",
                      "The answer depends on joining facts from different observations.", "A single observation is enough."),
    "recency": noul("Does answering query depend on the most recent information rather than older records?",
                    "The latest state or the most recent event matters.", "Older and newer records are equally relevant."),
}

ASSESS_QUESTIONS = {
    "sufficiency": noul("Does evidence contain everything needed to answer query?",
                        "The evidence answers the query completely.", "Something needed to answer is absent."),
    "utility": noul("Would retrieving more memories connected to evidence likely improve the answer to query?",
                    "Related memories probably add needed information.", "More retrieval is unlikely to help."),
    "missing": noul("Is a specific required piece of information for query missing from evidence?",
                    "A required fact, date or link is clearly absent.", "Nothing required is clearly missing."),
    "contradiction": noul("Do items in evidence contradict each other on a point relevant to query?",
                          "Conflicting accounts bear on the answer.", "The items are consistent or irrelevant to each other."),
}


def candidate_questions(i: int) -> dict[str, dict]:
    pre, c = f"candidate_{i}_", f"candidates[{i}]"
    return {
        pre + "relevance": noul(f"Is {c}.content relevant to answering query?",
                                "It contains information that bears directly on the query.", "It does not help answer the query."),
        pre + "usefulness": noul(f"Is the {c}.graph connection to evidence a useful route for answering query?",
                                 "Following this relation leads to information the query needs.", "The relation is incidental to the query."),
        pre + "novelty": noul(f"Does {c}.content add information that evidence does not already contain?",
                              "It adds new facts or details.", "It only repeats what evidence has."),
        pre + "support": noul(f"Does {c}.content support or complete the information already in evidence for query?",
                              "It corroborates or fills a gap in the evidence.", "It does not connect to the evidence."),
    }


# ------------------------------------------------------------------ maths
def allocate_budget(need: dict[str, float], B: int, theta_act: float = 0.10, gamma: float = 1.0, m: int = 1) -> dict[str, int]:
    """Eq 12-14: active graphs get a floor m, the rest splits by p^gamma with largest-remainder rounding."""
    active = [g for g in GRAPHS if need.get(g, 0.0) >= theta_act]
    if not active:  # nothing crosses the threshold: still search the most likely view
        active = [max(GRAPHS, key=lambda g: need.get(g, 0.0))]
    if B < m * len(active):  # budget too small for everyone: keep the most needed views
        active = sorted(active, key=lambda g: -need[g])[: max(B // max(m, 1), 1)]
    z = sum(need[g] ** gamma for g in active)
    w = {g: (need[g] ** gamma / z if z > 0 else 1 / len(active)) for g in active}
    extra = B - m * len(active)
    raw = {g: extra * w[g] for g in active}
    out = {g: m + math.floor(raw[g]) for g in active}
    left = B - sum(out.values())
    for g in sorted(active, key=lambda g: -(raw[g] - math.floor(raw[g])))[: max(left, 0)]:
        out[g] += 1
    return out


def max_depth(h: float, d_max: int) -> int:
    """Eq 15."""
    return min(d_max, max(1, math.ceil(d_max * h)))


def should_stop(s: float, u: float, m: float, c: float, cfg: ReadConfig) -> str | None:
    """Eq 21-22. Returns the stop reason or None."""
    if s >= cfg.theta_suff and m < cfg.theta_cont and c < cfg.theta_cont:
        return "sufficient"
    if u < cfg.theta_cont:
        return "low_utility"
    return None


def transition_score(z: float, a: float, p_g: float, l: float, n: float, pi_e: float, c: float, lam) -> float:
    """Eq 23."""
    return (lam[0] * z + lam[1] * a + lam[2] * p_g * l + lam[3] * n + lam[4] * (pi_e + c) / 2) / sum(lam)


def recency_adjust(s: float, r: float, tau_star: float, tau: float | None, w: float = 0.1) -> float:
    """Eq 24-25."""
    if tau is None:
        return s
    rho = 1.0 / (1.0 + max(0.0, tau_star - tau) / 86400.0)
    return (s + w * r * rho) / (1 + w * r)


# ---------------------------------------------------------------- retrieval
@dataclass
class Result:
    evidence: list[tuple[Node, float]]
    trace: dict = field(default_factory=dict)
    answer: str | None = None


def retrieve(store: MemoryStore, ctl: Controller, cfg: ReadConfig, query: str,
             system_two: Callable[[str, list[Node]], str] | None = None) -> Result:
    t0, calls0 = time.time(), ctl.calls
    if not len(store):
        return Result([], {"stop": "empty_memory"})

    ans = ctl.ask({"query": query}, ROUTE_QUESTIONS)
    need = {g: p(ans, g) for g in GRAPHS}
    h, r = p(ans, "multi_hop"), p(ans, "recency")
    budgets, depth = allocate_budget(need, cfg.budget, cfg.theta_act, cfg.gamma, cfg.min_budget), max_depth(h, cfg.d_max)
    trace: dict = {"need": need, "multi_hop": h, "recency": r, "budgets": dict(budgets), "depth": depth, "rounds": []}

    qvec = store.embedder.encode([query])[0]
    z = {nid: max(0.0, min(1.0, s)) for nid, s in store.cosine(qvec)}
    stamps = [n.timestamp for n in store.nodes.values() if n.timestamp is not None]
    tau_star = max(stamps) if stamps else 0.0

    anchors = [nid for nid, _ in store.rrf(query, cfg.n_anchors)]
    score: dict[str, float] = {nid: z[nid] for nid in anchors}  # entry points start at embedding similarity
    visited, frontier, remaining = set(anchors), list(anchors), dict(budgets)
    trace["anchors"] = list(anchors)

    def top(n: int) -> list[str]:
        return sorted(score, key=lambda i: -score[i])[:n]

    stop = "max_depth"
    for d in range(1, depth + 1):
        if ctl.calls - calls0 >= cfg.max_calls - 1 or time.time() - t0 > cfg.max_seconds:
            stop = "limit"
            break
        ev = [_view(store.nodes[i]) for i in top(cfg.top_k)]
        a = ctl.ask({"query": query, "evidence": ev}, ASSESS_QUESTIONS)
        s_d, u_d, m_d, c_d = p(a, "sufficiency"), p(a, "utility"), p(a, "missing"), p(a, "contradiction")
        rnd = {"round": d, "sufficiency": s_d, "utility": u_d, "missing": m_d, "contradiction": c_d}
        trace["rounds"].append(rnd)
        reason = should_stop(s_d, u_d, m_d, c_d, cfg)
        if reason:
            stop = reason
            break

        # expand: strongest edges first, per-graph budgets b_g
        options = []
        for g in remaining:
            for u in frontier:
                for v, e in store.neighbors(u, g):
                    if v not in visited:
                        options.append((e.weight, g, v, e))
        options.sort(key=lambda x: -x[0])
        cands: dict[str, tuple[str, object]] = {}
        for _, g, v, e in options:
            if v not in cands and remaining[g] > 0 and len(cands) < cfg.max_candidates:
                cands[v] = (g, e)
                remaining[g] -= 1
        rnd["expanded"] = len(cands)
        if not cands:
            stop = "exhausted"
            break

        ids = list(cands)
        qs: dict[str, dict] = {}
        for i in range(len(ids)):
            qs.update(candidate_questions(i))
        view = []
        for nid in ids:
            g, e = cands[nid]
            view.append({**_view(store.nodes[nid]), "graph": g, "relation": e.relation, "weight": round(e.weight, 3),
                         "source": e.src, "target": e.dst})
        sa = ctl.ask({"query": query, "evidence": ev, "candidates": view}, qs)
        scored = []
        for i, nid in enumerate(ids):
            g, e = cands[nid]
            s = transition_score(z[nid], p(sa, f"candidate_{i}_relevance"), need[g], p(sa, f"candidate_{i}_usefulness"),
                                 p(sa, f"candidate_{i}_novelty"), max(0.0, min(1.0, e.weight)), p(sa, f"candidate_{i}_support"),
                                 cfg.lambdas)
            scored.append((recency_adjust(s, r, tau_star, store.nodes[nid].timestamp, cfg.recency_weight), nid))
        scored.sort(reverse=True)
        visited.update(ids)
        beam = scored[: cfg.beam_w]
        for s, nid in beam:
            score[nid] = s
        frontier = [nid for _, nid in beam]
        rnd["beam"] = frontier

    trace["stop"], trace["controller_calls"], trace["seconds"] = stop, ctl.calls - calls0, round(time.time() - t0, 3)
    evidence = [(store.nodes[i], score[i]) for i in top(cfg.top_k)]
    res = Result(evidence, trace)
    if system_two is not None:
        res.answer = system_two(query, [n for n, _ in evidence])
    return res
