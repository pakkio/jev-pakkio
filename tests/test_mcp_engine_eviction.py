from jev_pakkio import mcp_engines as m


def test_one_gpu_engine_resident_at_a_time(monkeypatch):
    monkeypatch.setattr(m, "_loaded", {})
    monkeypatch.setattr(m, "get_engine", lambda name: object())
    for name in ["jev", "laya", "4g", "8g", "mercury", "laya"]:
        m._engine(name)
        assert len([n for n in m._loaded if n in m._GPU_ENGINES]) <= 1
    assert {"jev", "mercury"} <= set(m._loaded)  # API engines hold no VRAM and stay cached


def test_compare_releases_previous_engine_before_loading_next(monkeypatch):
    import gc
    import weakref

    class FakeResp:
        def model_dump(self, mode="json"):
            return {"answers": {}}

    class FakeEngine:
        def answer(self, req):
            return FakeResp()

    alive: list = []
    seen: list = []

    def fake_get(name):
        gc.collect()
        seen.append(sum(r() is not None for r in alive))  # engines still alive when this one loads
        e = FakeEngine()
        alive.append(weakref.ref(e))
        return e

    class FakeMCP:
        tools: dict = {}

        def tool(self):
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

    monkeypatch.setattr(m, "_loaded", {})
    monkeypatch.setattr(m, "get_engine", fake_get)
    monkeypatch.setattr(m, "_memory", None)
    mcp = FakeMCP()
    m.register_tools(mcp)
    mcp.tools["compare"]("text", {"q": {"type": "noul", "instructions": "x?"}}, engines=["laya", "4g", "8g"])
    assert seen == [0, 0, 0]


def test_eviction_closes_gpu_engines(monkeypatch):
    closed = []

    class Eng:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    monkeypatch.setattr(m, "_loaded", {})
    monkeypatch.setattr(m, "get_engine", lambda name: Eng(name))
    for name in ["laya", "4g", "jev", "8g"]:
        m._engine(name)
    assert closed == ["laya", "4g"]  # jev holds no VRAM so it is neither closed nor evicted
