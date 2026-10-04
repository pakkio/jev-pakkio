"""Named engines behind one interface: the TypeSafe System One contract (see systemone.py).

    engine = get_engine("laya")
    resp = engine.answer(SystemOneRequest(state="...", questions={...}))

jev   TypeSafe's hosted Jev (https://api.typesafe.ai/v1/systemone), needs TYPESAFE_API_KEY
laya  convaiinnovations/laya, a 421M encoder that answers typed questions in one forward pass
4g    Qwen scored through OptionScorer, 4-bit, sized for a 4 GB GPU
8g    Gemma E2B scored through OptionScorer, bf16, sized for an 8 GB GPU

The local model ids can be overridden with OPENJEV_ENGINE_4G / OPENJEV_ENGINE_8G.
"""
from __future__ import annotations

import os
from typing import Protocol

from .systemone import SystemOneRequest, SystemOneResponse, system_one

JEV_URL = "https://api.typesafe.ai/v1/systemone"
LAYA_REPO = "convaiinnovations/laya"
# name -> (default model id, quantize). 4g/8g are presets for OptionScorer.
LOCAL_PRESETS = {
    "4g": ("Qwen/Qwen2.5-1.5B-Instruct", "4bit"),
    "8g": ("google/gemma-3n-E2B-it", "none"),
}
ENGINE_NAMES = ("jev", "laya", *LOCAL_PRESETS)


class Engine(Protocol):
    name: str

    def answer(self, req: SystemOneRequest) -> SystemOneResponse: ...


def local_preset(name: str) -> tuple[str, str]:
    """(model id, quantize) for a 4g/8g engine, honouring OPENJEV_ENGINE_<NAME>."""
    model, quantize = LOCAL_PRESETS[name]
    return os.environ.get(f"OPENJEV_ENGINE_{name.upper()}", model), quantize


def _typesafe_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and os.path.exists("../.env"):  # same fallback the benchmark scripts used
        with open("../.env") as f:
            for line in f:
                if line.startswith("TYPESAFE_API_KEY="):
                    key = line.strip().split("=", 1)[1]
    if not key:
        raise RuntimeError("engine 'jev' needs TYPESAFE_API_KEY (environment or ../.env)")
    return key


class JevEngine:
    name = "jev"

    def __init__(self, model: str = "jev-latest", timeout: float = 30.0):
        self.model, self.timeout = model, timeout

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        import httpx

        body = req.model_dump(exclude_none=True, mode="json")
        body["model"] = req.model or self.model
        r = httpx.post(JEV_URL, json=body, headers={"Authorization": f"Bearer {_typesafe_key()}"},
                       timeout=self.timeout)
        r.raise_for_status()
        return SystemOneResponse.model_validate(r.json())


class LayaEngine:
    name = "laya"

    def __init__(self, repo: str = LAYA_REPO, subfolder: str | None = None):
        import laya

        self.repo = repo
        self.agent = laya.load(repo, subfolder=subfolder) if subfolder else laya.load(repo)

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        questions = {k: q.model_dump(exclude_none=True, mode="json") for k, q in req.questions.items()}
        out = self.agent.predict(req.state, questions)
        # Laya already speaks System One; extra keys (answer_confidence, action) are dropped by pydantic.
        return SystemOneResponse.model_validate({
            "model": req.model or f"laya:{self.repo}",
            "answers": out["answers"],
            "usage": out.get("usage") or {"input_tokens": 0, "output_tokens": 0},
        })


class LocalEngine:
    """A decoder LLM scored with OptionScorer (torch backend), one prefix-shared pass per question."""

    def __init__(self, name: str, model: str | None = None, quantize: str | None = None,
                 batch_size: int = 8, backend: str = "torch"):
        preset_model, preset_quant = local_preset(name) if name in LOCAL_PRESETS else (None, "none")
        self.name = name
        self.model = model or preset_model
        self.quantize = quantize or preset_quant
        from .server import _get_scorer_class  # lazy backend import

        kwargs = {"quantize": self.quantize} if backend == "torch" else {}
        self.scorer = _get_scorer_class(backend)(self.model, batch_size=batch_size, **kwargs)
        self.scorer.score("warm up", ["a", "b"])

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        return system_one(self.scorer, req, model_name=req.model or os.path.basename(self.model))


def get_engine(name: str) -> Engine:
    if name == "jev":
        return JevEngine()
    if name == "laya":
        return LayaEngine()
    if name in LOCAL_PRESETS:
        return LocalEngine(name)
    raise ValueError(f"unknown engine {name!r}; choose from {', '.join(ENGINE_NAMES)}")
