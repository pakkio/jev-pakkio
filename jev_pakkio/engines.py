"""Named engines behind one interface: the TypeSafe System One contract (see systemone.py).

    engine = get_engine("laya")
    resp = engine.answer(SystemOneRequest(state="...", questions={...}))

jev   TypeSafe's hosted Jev (https://api.typesafe.ai/v1/systemone), needs TYPESAFE_API_KEY
mercury  Inception's Mercury Decide on OpenRouter's Decisions API (free tier, same System One schema),
         needs OPENROUTER_API_KEY; docs: https://openrouter.ai/docs/api/api-reference/alphadecisions/
laya  convaiinnovations/laya, a 421M encoder that answers typed questions in one forward pass
4g    Qwen scored through OptionScorer, 4-bit, sized for a 4 GB GPU
8g    Gemma E2B scored through OptionScorer, bf16, sized for an 8 GB GPU

The local model ids can be overridden with OPENJEV_ENGINE_4G / OPENJEV_ENGINE_8G.

LoRA adapters: OPENJEV_LORA_4G / OPENJEV_LORA_8G = "NAME=PATH,NAME=PATH". When set, that engine loads the
adapters' base model (from adapter_config.json, unless OPENJEV_ENGINE_<NAME> is set) 4-bit through
lora_serve.LoraEngine. Each question uses the adapter named after its type ("choice", "score", "noul")
if one is loaded, or the adapter passed to answer(); otherwise the base model answers zero-shot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from .systemone import SystemOneRequest, SystemOneResponse, Usage, system_one

JEV_URL = "https://api.typesafe.ai/v1/systemone"
MERCURY_URL = "https://openrouter.ai/api/alpha/decisions"
MERCURY_MODEL = "inception/mercury-decide:free"
LAYA_REPO = "convaiinnovations/laya"
# name -> (default model id, quantize). 4g/8g are presets for OptionScorer.
LOCAL_PRESETS = {
    "4g": ("Qwen/Qwen2.5-1.5B-Instruct", "4bit"),
    "8g": ("google/gemma-3n-E2B-it", "none"),
}
ENGINE_NAMES = ("jev", "mercury", "laya", *LOCAL_PRESETS)


def lora_adapters(name: str) -> dict[str, str]:
    """Adapters configured for a 4g/8g engine via OPENJEV_LORA_<NAME>="NAME=PATH,..."."""
    adapters: dict[str, str] = {}
    for spec in filter(None, (x.strip() for x in os.environ.get(f"OPENJEV_LORA_{name.upper()}", "").split(","))):
        adapter, sep, path = spec.partition("=")
        if not sep or not adapter or not path:
            raise ValueError(f"OPENJEV_LORA_{name.upper()} expects NAME=PATH[,NAME=PATH], got {spec!r}")
        adapters[adapter] = path
    return adapters


def _adapter_base(adapters: dict[str, str]) -> str:
    bases = {json.loads((Path(p) / "adapter_config.json").read_text())["base_model_name_or_path"]
             for p in adapters.values()}
    if len(bases) != 1:
        raise ValueError(f"adapters must share one base model, got {sorted(bases)}")
    return bases.pop()


class _LoraScorer:
    """OptionScorer-shaped view of a LoraEngine with one adapter fixed, so system_one can drive it."""

    def __init__(self, engine, adapter: str | None) -> None:
        self.engine, self.adapter = engine, adapter

    @property
    def last_timing(self) -> dict:
        return self.engine.last_timing

    def score(self, prompt: str, labels: list[str], norm: str = "sum", chat: bool = False, sep: str = ""):
        _, probs = self.engine.score(prompt, labels, adapter=self.adapter)
        return [SimpleNamespace(probability=p) for p in probs]


class Engine(Protocol):
    name: str

    def answer(self, req: SystemOneRequest) -> SystemOneResponse: ...


def local_preset(name: str) -> tuple[str, str]:
    """(model id, quantize) for a 4g/8g engine, honouring OPENJEV_ENGINE_<NAME>."""
    model, quantize = LOCAL_PRESETS[name]
    return os.environ.get(f"OPENJEV_ENGINE_{name.upper()}", model), quantize


def _api_key(var: str, engine: str) -> str:
    key = os.environ.get(var)
    if not key and os.path.exists("../.env"):  # same fallback the benchmark scripts used
        with open("../.env") as f:
            for line in f:
                if line.startswith(f"{var}="):
                    key = line.strip().split("=", 1)[1]
    if not key:
        raise RuntimeError(f"engine {engine!r} needs {var} (environment or ../.env)")
    return key


def _typesafe_key() -> str:
    return _api_key("TYPESAFE_API_KEY", "jev")


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


class MercuryEngine:
    """Mercury Decide through OpenRouter's Decisions API; same request/response schema as Jev."""
    name = "mercury"

    def __init__(self, model: str = MERCURY_MODEL, timeout: float = 30.0):
        self.model, self.timeout = os.environ.get("OPENJEV_ENGINE_MERCURY", model), timeout

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        import httpx

        body = req.model_dump(exclude_none=True, mode="json")
        body["model"] = req.model or self.model
        key = _api_key("OPENROUTER_API_KEY", "mercury")
        r = httpx.post(MERCURY_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=self.timeout)
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
        self.adapters = lora_adapters(name) if name in LOCAL_PRESETS else {}
        if self.adapters and not model and f"OPENJEV_ENGINE_{name.upper()}" not in os.environ:
            model = _adapter_base(self.adapters)  # an adapter only works on the base it was trained on
        self.model = model or preset_model
        self.quantize = quantize or preset_quant
        if self.adapters:
            from .lora_serve import LoraEngine

            self.lora = LoraEngine(self.model, self.adapters, "4bit" if self.quantize == "none" else self.quantize)
            self.lora.score("warm up", ["a", "b"])
            return
        from .server import _get_scorer_class  # lazy backend import

        kwargs = {"quantize": self.quantize} if backend == "torch" else {}
        self.scorer = _get_scorer_class(backend)(self.model, batch_size=batch_size, **kwargs)
        self.scorer.score("warm up", ["a", "b"])

    def close(self) -> None:
        """Give the weights back to the GPU now.

        Dropping the last reference does not: a quantised model's Linear4bit modules outlive it and keep their
        packed weights, so empty every parameter and buffer in place instead of waiting for them to be collected.
        """
        import gc

        import torch

        for owner in (getattr(self, "scorer", None), getattr(self, "lora", None)):
            model = getattr(owner, "model", None)
            if model is not None:
                for t in (*model.parameters(), *model.buffers()):
                    t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def answer(self, req: SystemOneRequest, adapter: str | None = None) -> SystemOneResponse:
        """adapter: LoRA adapter for every question; default is the one named after each question's type."""
        model_name = req.model or os.path.basename(self.model)
        if not self.adapters:
            if adapter:
                raise ValueError(f"engine {self.name!r} has no adapters (set OPENJEV_LORA_{self.name.upper()})")
            return system_one(self.scorer, req, model_name=model_name)
        if adapter and adapter not in self.adapters:
            raise ValueError(f"unknown adapter {adapter!r}; loaded: {sorted(self.adapters)}")
        answers: dict = {}
        in_tok = out_tok = 0
        for qid, q in req.questions.items():  # one question at a time: the adapter depends on its type
            use = adapter or (q.type if q.type in self.adapters else None)
            one = system_one(_LoraScorer(self.lora, use), req.model_copy(update={"questions": {qid: q}}), model_name)
            answers.update(one.answers)
            in_tok += one.usage.input_tokens
            out_tok += one.usage.output_tokens
        return SystemOneResponse(model=model_name, answers=answers, usage=Usage(input_tokens=in_tok, output_tokens=out_tok))


def get_engine(name: str) -> Engine:
    if name == "jev":
        return JevEngine()
    if name == "mercury":
        return MercuryEngine()
    if name == "laya":
        return LayaEngine()
    if name in LOCAL_PRESETS:
        return LocalEngine(name)
    raise ValueError(f"unknown engine {name!r}; choose from {', '.join(ENGINE_NAMES)}")
