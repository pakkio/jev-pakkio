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
