"""LocalEngine with LoRA adapters, using a fake LoraEngine (no GPU, no model download)."""
import json

import pytest

from jev_pakkio import engines, lora_serve
from jev_pakkio.systemone import SystemOneRequest


class FakeLora:
    calls: list = []

    def __init__(self, model_path, adapters, quantize="4bit"):
        self.model_path, self.adapters, self.quantize = model_path, adapters, quantize
        self.last_timing = {"context_tokens": 3.0, "option_tokens": 2.0}

    def score(self, context, options, adapter=None, **_):
        FakeLora.calls.append(adapter)
        return [0.0] * len(options), [1.0 / len(options)] * len(options)


@pytest.fixture
def lora_dirs(tmp_path, monkeypatch):
    for name, base in (("langid", "google/gemma-4-E2B-it"), ("choice", "google/gemma-4-E2B-it")):
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    monkeypatch.setattr(lora_serve, "LoraEngine", FakeLora)
    monkeypatch.setenv("OPENJEV_LORA_8G", f"langid={tmp_path/'langid'},choice={tmp_path/'choice'}")
    FakeLora.calls = []
    return tmp_path


REQ = SystemOneRequest.model_validate({"state": "Ciao", "questions": {
    "c": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}},
    "n": {"type": "noul", "instructions": "y"}}})


def test_adapters_parsed_and_base_from_adapter_config(lora_dirs):
    eng = engines.LocalEngine("8g")
    assert sorted(eng.adapters) == ["choice", "langid"]
    assert eng.model == "google/gemma-4-E2B-it"  # not the gemma-3n preset
    assert eng.lora.quantize == "4bit"


def test_adapter_per_question_type_then_explicit(lora_dirs):
    eng = engines.LocalEngine("8g")
    FakeLora.calls = []
    resp = eng.answer(REQ)
    assert FakeLora.calls == ["choice", None]  # noul has no adapter -> zero-shot
    assert set(resp.answers) == {"c", "n"} and resp.usage.input_tokens == 6
    FakeLora.calls = []
    eng.answer(REQ, adapter="langid")
    assert FakeLora.calls == ["langid", "langid"]
    with pytest.raises(ValueError, match="unknown adapter"):
        eng.answer(REQ, adapter="nope")


def test_bad_spec_and_mismatched_bases(lora_dirs, monkeypatch):
    monkeypatch.setenv("OPENJEV_LORA_4G", "oops")
    with pytest.raises(ValueError, match="NAME=PATH"):
        engines.lora_adapters("4g")
    (lora_dirs / "choice" / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "other"}))
    with pytest.raises(ValueError, match="share one base"):
        engines.LocalEngine("8g")


def test_mercury_posts_to_openrouter_decisions(monkeypatch):
    import httpx

    seen = {}

    class R:
        def raise_for_status(self): ...
        def json(self):
            return {"model": "inception/mercury-decide:free", "usage": {"input_tokens": 1, "output_tokens": 0},
                    "answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1},
                                      "confidence": 0.9},
                                "n": {"type": "noul", "noul": 0.2}}}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, body=json, auth=headers["Authorization"])
        return R()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    resp = engines.get_engine("mercury").answer(REQ)
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["body"]["model"] == "inception/mercury-decide:free" and seen["auth"] == "Bearer test-key"
    assert resp.answers["c"].choice == "a" and "mercury" in engines.ENGINE_NAMES


def test_mercury_needs_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir("/")  # no ../.env fallback
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        engines.get_engine("mercury").answer(REQ)


def test_jev_pakkio_mcp_tools_use_configured_engine(monkeypatch):
    from jev_pakkio import mcp_server
    from jev_pakkio.systemone import SystemOneResponse

    seen = []

    class FakeEngine:
        def answer(self, req):
            seen.append(req)
            ans = {"choice": {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}, "confidence": 0.8},
                   "score": {"type": "score", "score": 1.5, "confidence": 0.7, "legend": {"0": "lo", "1": "hi"},
                            "probabilities": {"0": 0.5, "1": 0.5}},
                   "noul": {"type": "noul", "noul": 0.25}}[next(iter(req.questions.values())).type]
            return SystemOneResponse.model_validate({"model": "fake", "answers": {"q": ans},
                                                     "usage": {"input_tokens": 1, "output_tokens": 0}})

    monkeypatch.setattr(engines, "get_engine", lambda name: FakeEngine())
    monkeypatch.setattr(mcp_server, "_STATE", {})
    mcp_server.configure(engine="laya")
    assert mcp_server.classify("hello", {"a": "A", "b": "B"}) == {
        "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}, "confidence": 0.8}
    assert mcp_server.rate("hello", ["lo", "hi"])["score"] == 1.5
    assert mcp_server.noul("hello") == {"noul": 0.25}
    assert len(seen) == 3 and mcp_server._STATE["engine"] == "laya"  # no local scorer was loaded
    assert "scorer" not in mcp_server._STATE


def test_jev_pakkio_mcp_per_call_engine(monkeypatch):
    from jev_pakkio import mcp_server
    from jev_pakkio.systemone import SystemOneResponse

    asked = []

    class Fake:
        def __init__(self, name): self.name = name
        def answer(self, req):
            asked.append(self.name)
            return SystemOneResponse.model_validate({"model": self.name, "usage": {"input_tokens": 0, "output_tokens": 0},
                "answers": {"q": {"type": "noul", "noul": 0.5}}})

    monkeypatch.setattr(engines, "get_engine", Fake)
    monkeypatch.setattr(mcp_server, "_STATE", {})
    mcp_server.configure(engine="laya")
    mcp_server.noul("x")
    mcp_server.noul("x", engine="jev")
    mcp_server.noul("x", engine="8g")
    assert asked == ["laya", "jev", "8g"]
    with pytest.raises(ValueError, match="unknown engine"):
        mcp_server.noul("x", engine="gpt")
