"""Write path (paper 3.2): observation -> typing -> candidates -> relations -> memory.

Cost is bounded: one batched typing call, and one batched relation call over at most `k_cand` pairs.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from .control import Controller, choice, noul, p
from .store import Edge, MemoryStore, Node, guess_entities


@dataclass
class WriteConfig:
    k_cand: int = 10          # K_w
    theta_rel: float = 0.60   # eq 9
    consolidate_every: int = 20
    consolidate_pairs: int = 3
    merge_threshold: float = 0.85
    # weights of the deterministic candidate score s_cand (eq 8); the paper does not give values
    w_vec: float = 0.5
    w_lex: float = 0.2
    w_ent: float = 0.2
    w_time: float = 0.1
    time_scale_days: float = 7.0


TYPE_QUESTIONS = {
    "episodic": noul(
        "Does observation describe a particular experience or event involving a participant?",
        "A specific past, current or planned event, even if its exact time is unstated.",
        "Only a general fact, procedure or preference with no particular event."),
    "semantic": noul(
        "Does observation state a general fact, definition or stable piece of knowledge?",
        "A fact about a person, thing or the world that is not tied to a single event.",
        "Only a one-off event, a procedure or a preference."),
    "procedural": noul(
        "Does observation describe how to do something, a routine or a repeatable procedure?",
        "Steps, a method or a standing routine that could be reused.",
        "No procedure; a plain fact, event or preference."),
    "preference": noul(
        "Does observation express a participant's preference, aversion or habitual choice?",
        "An attributable like, dislike, preferred option or habitual choice.",
        "An isolated action alone, another person's unattributed preference, or no preference evidence."),
}


def _view(n: Node) -> dict:
    return {"id": n.id, "content": n.content, "timestamp": _iso(n.timestamp), "entities": n.entities}


def _iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


def relation_questions(i: int) -> dict[str, dict]:
    pre = f"pair_{i}_"
    cmp_ = f"Compare new_memory.content with candidates[{i}].content. "
    return {
        pre + "semantic": noul(
            cmp_ + "Would a semantic link between these observations help retrieve a shared specific topic or fact?",
            "A specific shared topic, fact or event makes the connection useful.",
            "Only generic conversational vocabulary or no meaningful semantic connection."),
        pre + "caused_by": noul(
            cmp_ + f"Does the candidate event (candidates[{i}]) cause, enable or explain the event in new_memory.content?",
            "The supplied accounts support this direction of causal influence.",
            "Only similarity, chronology, a shared entity, or insufficient causal evidence."),
        pre + "causes": noul(
            cmp_ + f"Does the event in new_memory.content cause, enable or explain the candidate event (candidates[{i}])?",
            "The supplied accounts support this direction of causal influence.",
            "Only similarity, chronology, a shared entity, or insufficient causal evidence."),
    }


def find_candidates(store: MemoryStore, node: Node, cfg: WriteConfig) -> list[tuple[str, float]]:
    """Deterministic candidate discovery: vector + lexical + shared entities + temporal proximity."""
    others = {i for i in store.order if i != node.id}
    if not others:
        return []
    vec = {i: s for i, s in store.cosine(store.vec(node.id), exclude={node.id})}
    lex = dict(store.bm25(node.content, exclude={node.id}))
    top_lex = max(lex.values(), default=1.0)
    mine = set(node.entities)
    scored = []
    for i in others:
        o = store.nodes[i]
        ent = len(mine & set(o.entities)) / max(len(mine | set(o.entities)), 1)
        t = 0.0
        if node.timestamp is not None and o.timestamp is not None:
            t = 1.0 / (1.0 + abs(node.timestamp - o.timestamp) / 86400.0 / cfg.time_scale_days)
        s = cfg.w_vec * max(vec.get(i, 0.0), 0.0) + cfg.w_lex * lex.get(i, 0.0) / top_lex + cfg.w_ent * ent + cfg.w_time * t
        scored.append((i, s))
    scored.sort(key=lambda x: -x[1])
    return scored[: cfg.k_cand]


def write(store: MemoryStore, ctl: Controller, cfg: WriteConfig, text: str, timestamp: float | None = None,
          entities: list[str] | None = None, provenance: str = "", node_id: str | None = None) -> Node | None:
    text = (text or "").strip()
    if not text:
        return None
    node = Node(id=node_id or uuid.uuid4().hex[:8], content=text, timestamp=timestamp,
                entities=sorted(set(entities)) if entities is not None else guess_entities(text), provenance=provenance)

    # 1. typing: overlapping scores, never a store-or-discard gate
    ans = ctl.ask({"observation": text}, TYPE_QUESTIONS)
    node.types = {k: p(ans, k) for k in TYPE_QUESTIONS}

    store.add_node(node)
    cands = find_candidates(store, node, cfg)

    # 2. deterministic relations
    for cid, _ in cands:
        c = store.nodes[cid]
        shared = set(node.entities) & set(c.entities)
        if shared:
            store.add_edge(Edge(cid, node.id, "entity", 1.0, "shared_entity:" + ",".join(sorted(shared))))
        if node.timestamp is not None and c.timestamp is not None and node.timestamp != c.timestamp:
            a, b = (c, node) if c.timestamp < node.timestamp else (node, c)
            store.add_edge(Edge(a.id, b.id, "temporal", 1.0, "before"))
        elif node.timestamp is not None and node.timestamp == c.timestamp:
            store.add_edge(Edge(cid, node.id, "temporal", 1.0, "same_time"))

    # 3. learned relations over the bounded candidate set, one batched call
    if cands:
        qs: dict[str, dict] = {}
        for i in range(len(cands)):
            qs.update(relation_questions(i))
        ans = ctl.ask({"new_memory": _view(node), "candidates": [_view(store.nodes[c]) for c, _ in cands]}, qs)
        for i, (cid, _) in enumerate(cands):
            pre = f"pair_{i}_"
            sem, cb, cs = p(ans, pre + "semantic"), p(ans, pre + "caused_by"), p(ans, pre + "causes")
            if sem >= cfg.theta_rel:
                store.add_edge(Edge(cid, node.id, "semantic", sem, "semantic"))
            if cb >= cfg.theta_rel:
                store.add_edge(Edge(cid, node.id, "causal", cb, "caused_by"))
            if cs >= cfg.theta_rel:
                store.add_edge(Edge(node.id, cid, "causal", cs, "causes"))

    if cfg.consolidate_every and len(store) % cfg.consolidate_every == 0:
        consolidate(store, ctl, cfg, node, cands[: cfg.consolidate_pairs])
    return node


def consolidate(store: MemoryStore, ctl: Controller, cfg: WriteConfig, node: Node, cands: list[tuple[str, float]]) -> list[dict]:
    """Periodic maintenance over a bounded neighbourhood. Records decisions and links; never deletes evidence."""
    out = []
    for cid, _ in cands:
        qs = {
            "redundancy": noul("Compare new_memory.content with candidates[0].content. Do they report the same fact or event?",
                               "Both accounts state the same fact or event.", "They differ in substance."),
            "contradiction": noul("Compare new_memory.content with candidates[0].content. Do they contradict each other?",
                                  "They cannot both be true.", "They are compatible or unrelated."),
            "obsolescence": noul("Compare new_memory.content with candidates[0].content. Does new_memory supersede the candidate?",
                                 "The new account updates or replaces the candidate.", "The candidate is still current."),
            "link_usefulness": noul("Compare new_memory.content with candidates[0].content. Would a link help later retrieval?",
                                    "A specific shared topic, fact or event makes the link useful.", "No useful link."),
            "representation": choice(
                "Compare new_memory.content with candidates[0].content. Which representation best fits the relationship between "
                "these two observations? Judge from the supplied accounts; do not assume answers to other questions.",
                {"keep_separate": "Contradictory accounts, unique details that a combined representation would lose, or distinct facts/events without a supported general pattern.",
                 "merge": "Compatible accounts of the same fact or event can be combined without losing unique details.",
                 "promote": "Distinct repeated episodes explicitly support a stable general pattern suitable for semantic abstraction; prefer this over merge for repeated events.",
                 "uncertain": "Insufficient evidence to choose a safe combined or separate representation."}),
        }
        ans = ctl.ask({"new_memory": _view(node), "candidates": [_view(store.nodes[cid])]}, qs)
        rep = ans["representation"]
        rec = {"new": node.id, "candidate": cid, "redundancy": p(ans, "redundancy"), "contradiction": p(ans, "contradiction"),
               "obsolescence": p(ans, "obsolescence"), "link_usefulness": p(ans, "link_usefulness"),
               "representation": rep["choice"], "representation_p": rep["probabilities"][rep["choice"]]}
        rec["escalate_to_system_two"] = (rec["representation"] in ("merge", "promote")
                                         and rec["representation_p"] >= cfg.merge_threshold and rec["contradiction"] < cfg.merge_threshold)
        if rec["link_usefulness"] >= cfg.theta_rel:
            store.add_edge(Edge(cid, node.id, "semantic", rec["link_usefulness"], "consolidation"))
        store.decisions.append(rec)
        out.append(rec)
    return out
