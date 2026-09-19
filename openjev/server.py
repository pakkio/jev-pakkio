"""HTTP server: one loaded model, many scoring requests.

    openjev serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import os
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .systemone import SystemOneRequest, SystemOneResponse, system_one


def _resolve_backend(backend: str | None = None) -> str:
    """Resolve the scoring backend.

    Precedence: explicit arg > OPENJEV_BACKEND env var > "mlx".
    """
    if backend and backend in ("mlx", "torch"):
        return backend
    env_backend = os.environ.get("OPENJEV_BACKEND")
    if env_backend in ("mlx", "torch"):
        return env_backend
    return "mlx"


def _get_scorer_class(backend: str):
    """Return the OptionScorer class for the given backend.

    Imported lazily: mlx is Apple-silicon only, so importing it on a CUDA or
    CPU host fails at load time even when the torch backend is the one wanted.
    """
    if backend == "torch":
        from .scorer_torch import OptionScorer as TorchOptionScorer
        return TorchOptionScorer
    from .scorer import OptionScorer as MLXOptionScorer
    return MLXOptionScorer


def _default_model(backend: str) -> str:
    """Each backend ships its own default checkpoint format."""
    if backend == "torch":
        from .scorer_torch import DEFAULT_MODEL
    else:
        from .scorer import DEFAULT_MODEL
    return DEFAULT_MODEL


class ScoreRequest(BaseModel):
    context: str
    options: list[str] = Field(min_length=2)
    norm: Literal["mean", "sum", "pmi"] = "mean"
    chat: bool = False
    sep: str = ""


class OptionOut(BaseModel):
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float


class ScoreResponse(BaseModel):
    best: str
    best_index: int
    options: list[OptionOut]
    timing: dict[str, float]


def _resolve_quantize(quantize: str | None = None) -> str:
    """Resolve the weight quantisation: explicit arg > OPENJEV_QUANTIZE > none."""
    for candidate in (quantize, os.environ.get("OPENJEV_QUANTIZE")):
        if candidate in ("none", "8bit", "4bit"):
            return candidate
    return "none"


def create_app(model_path: str | None = None, batch_size: int = 8, backend: str | None = None,
               device: str = "auto", quantize: str | None = None) -> FastAPI:
    backend = _resolve_backend(backend)
    quantize = _resolve_quantize(quantize)
    model_path = model_path or os.environ.get("OPENJEV_MODEL") or _default_model(backend)
    adapter_path = os.environ.get("OPENJEV_ADAPTER")  # optional LoRA adapter dir
    OptionScorer = _get_scorer_class(backend)
    app = FastAPI(title="openjev", version="0.1.0")
    state: dict = {}
    api_key = os.environ.get("OPENJEV_API_KEY")  # if set, /v1/systemone requires "Authorization: Bearer ***"
    model_name = os.path.basename(model_path.rstrip("/"))

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    @app.on_event("startup")
    def _load() -> None:
        t = time.perf_counter()
        kwargs = {"quantize": quantize} if backend == "torch" else {}
        scorer = OptionScorer(model_path, batch_size=batch_size, adapter_path=adapter_path, device=device, **kwargs)
        scorer.score("warm up", ["a", "b"])  # compile kernels / warm up before the first request
        state["scorer"] = scorer
        state["load_s"] = time.perf_counter() - t

    @app.get("/health")
    def health() -> dict:
        return {"ok": "scorer" in state, "model": model_path, "backend": backend,
                "quantize": quantize, "device": getattr(state.get("scorer"), "device", None),
                "load_s": state.get("load_s")}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        try:
            res = scorer.score(req.context, req.options, norm=req.norm, chat=req.chat, sep=req.sep)
        except ValueError as e:
            raise HTTPException(400, str(e))
        best = max(range(len(res)), key=lambda i: res[i].score)
        return ScoreResponse(
            best=res[best].option,
            best_index=best,
            options=[OptionOut(**r.to_dict()) for r in res],
            timing=scorer.last_timing,
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(_auth)])
    def systemone(req: SystemOneRequest) -> SystemOneResponse:
        """TypeSafe System One contract: state + typed questions -> typed answers (docs.typesafe.ai)."""
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        try:
            return system_one(scorer, req, model_name=req.model or model_name)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int, backend: str | None = None,
          device: str = "auto", quantize: str | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, batch_size, backend, device, quantize), host=host, port=port, workers=1)
