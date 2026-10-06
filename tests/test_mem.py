import os
import tempfile
import unittest
from datetime import datetime, timezone

from jev_pakkio.mem import EngineController, HashEmbedder, JevMem, ReadConfig
from jev_pakkio.mem.service import MemoryService
from jev_pakkio.systemone import ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage
from jev_pakkio.mem.read import allocate_budget, max_depth, recency_adjust, should_stop, transition_score
from jev_pakkio.mem.store import GRAPHS


def ts(day: int) -> float:
    return datetime(2024, 5, day, 10, tzinfo=timezone.utc).timestamp()


class FakeController:
    """Keyword rules standing in for System One, so the tests need no model."""

    def __init__(self):
        self.calls = 0
        self.log = []

    def ask(self, state, questions):
        self.calls += 1
        self.log.append(set(questions))
        out = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                out[qid] = {"choice": "keep_separate", "probabilities": {k: 0.25 for k in q["criteria"]}}
            else:
                out[qid] = {"noul": self.noul(qid, state)}
        return out

    def noul(self, qid, state):
        if qid in ("episodic", "semantic", "procedural", "preference"):
            return 0.9 if qid == "episodic" else 0.1
        if qid.startswith("pair_"):
            i, kind = int(qid.split("_")[1]), qid.split("_", 2)[2]
            new, cand = state["new_memory"]["content"].lower(), state["candidates"][i]["content"].lower()
            if kind == "semantic":
                return 0.9 if "bicycle" in new and "bicycle" in cand else 0.1
            if kind == "caused_by":
                return 0.9 if "broke" in cand and "because" in new else 0.1
            return 0.1
        if qid.startswith("candidate_"):
            return 0.8
        if qid in ("redundancy", "contradiction", "obsolescence", "link_usefulness"):
            return 0.1
        if qid in ("semantic", "temporal", "causal", "entity", "multi_hop", "recency"):
            q = state["query"].lower()
            return {"temporal": 0.9 if "when" in q else 0.0, "causal": 0.9 if "why" in q else 0.0,
                    "semantic": 0.8, "entity": 0.5, "multi_hop": 0.8, "recency": 0.1}[qid]
        text = " ".join(e["content"].lower() for e in state.get("evidence", []))
        both = "bought" in text and "broke" in text
        return {"sufficiency": 0.9 if both else 0.2, "utility": 0.8, "missing": 0.1 if both else 0.7, "contradiction": 0.0}[qid]


def build(ctl=None, **kw):
    mem = JevMem(ctl or FakeController(), HashEmbedder(), **kw)
    mem.add("Mira: My old bicycle broke.", timestamp=ts(14), entities=["Mira", "bicycle"], node_id="m1")
    mem.add("Mira: I bought a new bicycle yesterday because my old one broke.", timestamp=ts(16),
            entities=["Mira", "bicycle"], node_id="m2")
    mem.add("Tom: The weather in Lisbon is lovely this week.", timestamp=ts(17), entities=["Tom", "Lisbon"], node_id="d1")
    mem.add("Tom: I prefer tea to coffee.", timestamp=ts(18), entities=["Tom"], node_id="d2")
    return mem


class WritePathTests(unittest.TestCase):
    def test_bicycle_example_builds_all_four_views(self):
        mem = build()
        kinds = {(e.src, e.dst, e.kind, e.relation.split(":")[0]) for e in mem.store.edges}
        self.assertIn(("m1", "m2", "causal", "caused_by"), kinds)
        self.assertIn(("m1", "m2", "temporal", "before"), kinds)
        self.assertIn(("m1", "m2", "semantic", "semantic"), kinds)
        self.assertIn(("m1", "m2", "entity", "shared_entity"), kinds)
        self.assertNotIn(("m2", "m1", "causal", "causes"), kinds)

    def test_every_nonempty_observation_is_kept_and_typed(self):
        mem = build()
        self.assertEqual(len(mem.store), 4)
        self.assertAlmostEqual(mem.store.nodes["m1"].types["episodic"], 0.9)
        self.assertIsNone(mem.add("   "))

    def test_write_cost_is_two_batched_calls(self):
        ctl = FakeController()
        mem = JevMem(ctl, HashEmbedder())
        mem.add("Mira: My old bicycle broke.", timestamp=ts(14), node_id="m1")
        self.assertEqual(ctl.calls, 1)  # typing only: no candidates yet
        mem.add("Mira: I bought a new bicycle.", timestamp=ts(16), node_id="m2")
        self.assertEqual(ctl.calls, 3)  # typing + one relation batch

    def test_consolidation_records_decisions_without_deleting(self):
        mem = JevMem(FakeController(), HashEmbedder())
        mem.write_cfg.consolidate_every = 2
        mem.add("Mira: My old bicycle broke.", node_id="a")
        mem.add("Mira: I bought a new bicycle.", node_id="b")
        self.assertEqual(len(mem.store), 2)
        self.assertEqual(len(mem.store.decisions), 1)
        self.assertEqual(mem.store.decisions[0]["representation"], "keep_separate")


class ReadPathTests(unittest.TestCase):
    def test_query_finds_cause_and_event(self):
        mem = build()
        res = mem.query("When did Mira buy a new bicycle, and why?")
        ids = [n.id for n, _ in res.evidence]
        self.assertIn("m1", ids)
        self.assertIn("m2", ids)
        self.assertLessEqual(res.trace["controller_calls"], mem.read_cfg.max_calls)

    def test_routing_activates_temporal_and_causal(self):
        res = build().query("When did Mira buy a new bicycle, and why?")
        self.assertIn("temporal", res.trace["budgets"])
        self.assertIn("causal", res.trace["budgets"])
        self.assertEqual(sum(res.trace["budgets"].values()), 80)

    def test_system_two_only_sees_final_evidence(self):
        seen = []
        mem = build(system_two=lambda q, nodes: seen.append([n.id for n in nodes]) or "answer")
        self.assertEqual(mem.query("When did Mira buy a bicycle?").answer, "answer")
        self.assertEqual(len(seen), 1)

    def test_empty_memory(self):
        res = JevMem(FakeController(), HashEmbedder()).query("anything")
        self.assertEqual(res.evidence, [])

    def test_persistence_roundtrip(self):
        mem = build()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.json")
            mem.save(path)
            again = JevMem.load(path, FakeController(), HashEmbedder())
        self.assertEqual(len(again.store), len(mem.store))
        self.assertEqual(len(again.store.edges), len(mem.store.edges))


class MathTests(unittest.TestCase):
    def test_budget_sums_to_B_with_floor(self):
        b = allocate_budget({"semantic": 0.8, "temporal": 0.9, "causal": 0.9, "entity": 0.05}, 80)
        self.assertEqual(sum(b.values()), 80)
        self.assertNotIn("entity", b)  # below theta_act
        self.assertTrue(all(v >= 1 for v in b.values()))
        self.assertGreater(b["temporal"], b["semantic"])

    def test_budget_proportional(self):
        b = allocate_budget({"semantic": 0.6, "temporal": 0.2, "causal": 0.2, "entity": 0.0}, 10, m=1)
        self.assertEqual(sum(b.values()), 10)
        self.assertGreater(b["semantic"], b["temporal"])

    def test_budget_falls_back_when_nothing_active(self):
        b = allocate_budget({g: 0.01 for g in GRAPHS}, 8)
        self.assertEqual(sum(b.values()), 8)
        self.assertEqual(len(b), 1)

    def test_depth(self):
        self.assertEqual(max_depth(0.0, 3), 1)
        self.assertEqual(max_depth(0.4, 3), 2)
        self.assertEqual(max_depth(1.0, 3), 3)
        self.assertEqual(max_depth(1.0, 5), 5)

    def test_stop_rule(self):
        cfg = ReadConfig()
        self.assertEqual(should_stop(0.9, 0.9, 0.1, 0.1, cfg), "sufficient")
        self.assertIsNone(should_stop(0.9, 0.9, 0.5, 0.1, cfg))  # sufficient but something missing
        self.assertEqual(should_stop(0.2, 0.1, 0.9, 0.0, cfg), "low_utility")

    def test_transition_and_recency(self):
        lam = (1, 1, 1, 1, 1)
        self.assertAlmostEqual(transition_score(1, 1, 1, 1, 1, 1, 1, lam), 1.0)
        self.assertAlmostEqual(transition_score(0, 0, 0, 0, 0, 0, 0, lam), 0.0)
        newest, week_old = ts(20), ts(13)
        fresh = recency_adjust(0.5, 1.0, newest, newest)
        stale = recency_adjust(0.5, 1.0, newest, week_old)
        self.assertGreater(fresh, stale)
        self.assertEqual(recency_adjust(0.5, 1.0, newest, None), 0.5)
        self.assertEqual(recency_adjust(0.5, 0.0, newest, week_old), 0.5)  # r(q)=0 disables the adjustment


class FakeEngine:
    """Engine protocol (answer(req) -> SystemOneResponse) backed by the keyword fake."""
    name = "fake"

    def __init__(self):
        self.fake = FakeController()
        self.adapters = []

    def answer(self, req, adapter=None):
        self.adapters.append(adapter)
        qs = {k: q.model_dump(mode="json") for k, q in req.questions.items()}
        out = {}
        for k, a in self.fake.ask(req.state, qs).items():
            out[k] = NoulAnswer(noul=a["noul"]) if "noul" in a else ChoiceAnswer(
                choice=a["choice"], probabilities=a["probabilities"], confidence=0.5)
        return SystemOneResponse(model="fake", answers=out, usage=Usage(input_tokens=0, output_tokens=0))


class EngineControllerTests(unittest.TestCase):
    def test_engine_drives_memory_end_to_end(self):
        eng = FakeEngine()
        mem = build(EngineController(eng))
        self.assertIn(("m1", "m2", "causal"), {(e.src, e.dst, e.kind) for e in mem.store.edges})
        ids = [n.id for n, _ in mem.query("When did Mira buy a new bicycle, and why?").evidence]
        self.assertTrue({"m1", "m2"} <= set(ids))

    def test_adapter_is_forwarded_only_when_set(self):
        eng = FakeEngine()
        EngineController(eng).ask({"observation": "s"}, {"episodic": {"type": "noul", "instructions": "x"}})
        EngineController(eng, adapter="noul").ask({"observation": "s"}, {"episodic": {"type": "noul", "instructions": "x"}})
        self.assertEqual(eng.adapters, [None, "noul"])


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engines = []

        def get_engine(name):
            self.engines.append(name)
            return FakeEngine()

        self.svc = MemoryService(get_engine, HashEmbedder, root=self.tmp.name)

    def test_add_query_persist_and_reload(self):
        r1 = self.svc.add("Mira: My old bicycle broke.", "t", timestamp="2024-05-14T10:00", entities=["Mira", "bicycle"])
        r2 = self.svc.add("Mira: I bought a new bicycle yesterday because my old one broke.", "t",
                          timestamp="2024-05-16T10:00", entities=["Mira", "bicycle"])
        self.assertEqual((r1["nodes"], r2["nodes"]), (1, 2))
        self.assertGreaterEqual(r2["edges"], 1)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "t.json")))
        fresh = MemoryService(lambda n: FakeEngine(), HashEmbedder, root=self.tmp.name)  # new process
        out = fresh.query("When did Mira buy a new bicycle, and why?", "t", top_k=5)
        self.assertEqual(len(out["evidence"]), 2)
        self.assertEqual(fresh.stores(), {"t": {"loaded": True, "nodes": 2, "edges": r2["edges"]}})

    def test_engine_can_change_between_calls(self):
        self.svc.add("Mira: My old bicycle broke.", "s", engine="laya")
        self.svc.query("bicycle", "s", engine="8g")
        self.assertEqual(self.engines, ["laya", "8g"])

    def test_store_names_cannot_escape_the_directory(self):
        for bad in ("../x", "a/b", "", "a b", "x" * 65):
            with self.assertRaises(ValueError):
                self.svc.add("hello there", bad)

    def test_empty_text_is_not_persisted(self):
        self.assertIsNone(self.svc.add("   ", "e")["id"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "e.json")))


if __name__ == "__main__":
    unittest.main()
