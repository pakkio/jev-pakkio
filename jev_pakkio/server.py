"""HTTP server: one loaded model, many scoring requests.

    jev_pakkio serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import os
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .systemone import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    confidence,
    render_choice,
    render_noul,
    render_score,
    system_one,
)


def _resolve_backend(backend: str | None = None) -> str:
    """Resolve the scoring backend.

    Precedence: explicit arg > OPENJEV_BACKEND env var > "auto" (mlx on Apple
    silicon, torch everywhere else -- see cli.resolve_backend).
    """
    from .cli import resolve_backend

    if backend and backend in ("mlx", "torch", "auto"):
        return resolve_backend(backend)
    env_backend = os.environ.get("OPENJEV_BACKEND")
    if env_backend in ("mlx", "torch", "auto"):
        return resolve_backend(env_backend)
    return resolve_backend("auto")


def _auto_model(backend: str) -> str:
    """The largest model this machine can hold; MLX has no VRAM to size."""
    if backend != "torch":
        return _default_model(backend)
    from .scorer_torch import auto_model

    return auto_model()


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


class LoraScoreRequest(BaseModel):
    context: str
    options: list[str] = Field(min_length=2)
    adapter: str | None = None  # None = the base model zero-shot
    chat: bool = True
    sep: str = "\nChoice: "


class LoraScoreResponse(BaseModel):
    best: str
    best_index: int
    adapter: str | None
    chat: bool
    probabilities: list[float]
    logprob_sums: list[float]
    timing: dict[str, float]


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


def _resolve_heads(heads: dict | None) -> dict:
    """Checkpoint paths per question type: explicit dict > OPENJEV_HEAD_<TYPE> env > none."""
    out: dict = {}
    for qtype in ("choice", "score", "noul"):
        path = (heads or {}).get(qtype) or os.environ.get(f"OPENJEV_HEAD_{qtype.upper()}")
        if path:
            out[qtype] = path
    return out


def create_app(model_path: str | None = None, batch_size: int = 8, backend: str | None = None,
               device: str = "auto", quantize: str | None = None, heads: dict | None = None,
               lora: dict | None = None, engine: str | None = None) -> FastAPI:
    """engine: a name from engines.ENGINE_NAMES. 4g/8g select a model + quantisation preset on the torch
    backend; jev/laya serve /v1/systemone from the hosted API or the Laya encoder instead of a local scorer.

    lora: {name: adapter_dir}. When given (torch only), the server runs in LoRA mode: one
    4-bit model through lora_serve.LoraEngine instead of OptionScorer, /v1/lora/score picks an
    adapter per request, and /score + /v1/systemone use the chat format with the adapter named
    after the question type ("choice", "score", "noul") when one is loaded, zero-shot otherwise.
    """
    from .engines import ENGINE_NAMES, LOCAL_PRESETS, local_preset

    if engine and engine not in ENGINE_NAMES:
        raise ValueError(f"unknown engine {engine!r}; choose from {', '.join(ENGINE_NAMES)}")
    remote = engine in ("jev", "laya")
    if engine in LOCAL_PRESETS:
        backend, (model_path, quantize) = "torch", local_preset(engine)
    backend = _resolve_backend(backend)
    if lora and backend != "torch":
        raise ValueError("LoRA serving needs the torch backend")
    quantize = _resolve_quantize(quantize)
    head_paths = _resolve_heads(heads)
    model_path = model_path or os.environ.get("OPENJEV_MODEL") or (engine if remote else _default_model(backend))
    if model_path == "auto":
        # Same rule as the CLI: largest model this machine's VRAM can hold.
        model_path = _auto_model(backend)
    OptionScorer = _get_scorer_class(backend)
    app = FastAPI(title="jev_pakkio", version="0.1.0")
    state: dict = {}
    api_key = os.environ.get("OPENJEV_API_KEY")  # if set, /v1/systemone requires "Authorization: Bearer ***"
    model_name = os.path.basename(model_path.rstrip("/"))

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    @app.on_event("startup")
    def _load() -> None:
        t = time.perf_counter()
        if remote:
            from .engines import get_engine

            state["engine"] = get_engine(engine)
            state["heads"] = {}
            state["load_s"] = time.perf_counter() - t
            return
        if lora:
            from .lora_serve import LoraEngine

            state["lora"] = LoraEngine(model_path, lora, quantize if quantize != "none" else "4bit")
            state["lora"].score("warm up", ["a", "b"])
            state["heads"] = {}
            state["load_s"] = time.perf_counter() - t
            return
        kwargs = {"quantize": quantize} if backend == "torch" else {}
        if backend == "torch" and device and device != "auto":
            kwargs["device"] = device
        scorer = OptionScorer(model_path, batch_size=batch_size, **kwargs)
        scorer.score("warm up", ["a", "b"])  # compile kernels / warm up before the first request
        state["scorer"] = scorer
        state["heads"] = {}
        if head_paths:
            from .head_serve import HeadScorer

            for qtype, path in head_paths.items():
                state["heads"][qtype] = HeadScorer(scorer, backend, path)
        state["load_s"] = time.perf_counter() - t

    @app.get("/health")
    def health() -> dict:
        lora_engine = state.get("lora")
        return {"ok": "scorer" in state or "engine" in state or lora_engine is not None, "engine": engine,
                "model": model_path, "backend": backend,
                "quantize": quantize, "heads": sorted(state.get("heads", {})),
                "lora": sorted(lora_engine.adapters) if lora_engine else [], "load_s": state.get("load_s")}

    def _lora_probs(qtype: str, context: str, options: list[str]) -> tuple[list[float], list[float]]:
        engine = state["lora"]
        try:
            return engine.score(context, options, adapter=qtype if qtype in engine.adapters else None)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/v1/lora/score", response_model=LoraScoreResponse)
    def lora_score(req: LoraScoreRequest) -> LoraScoreResponse:
        engine = state.get("lora")
        if engine is None:
            raise HTTPException(503 if "scorer" not in state else 404,
                                "model still loading" if "scorer" not in state else "server not started with --lora")
        try:
            sums, probs = engine.score(req.context, req.options, adapter=req.adapter, chat=req.chat, sep=req.sep)
        except ValueError as e:
            raise HTTPException(400, str(e))
        best = max(range(len(probs)), key=lambda i: probs[i])
        return LoraScoreResponse(best=req.options[best], best_index=best, adapter=req.adapter, chat=req.chat,
                                 probabilities=probs, logprob_sums=sums, timing=engine.last_timing)

    def _head_probs(qtype: str, context: str, options: list[str]) -> list[float] | None:
        head = state.get("heads", {}).get(qtype)
        if head is None:
            return None
        try:
            return head.answer(context, options)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        if "lora" in state:  # LoRA mode: chat format, "choice" adapter if loaded
            import math

            sums, probs = _lora_probs("choice", req.context, req.options)
            best = max(range(len(probs)), key=lambda i: probs[i])
            return ScoreResponse(
                best=req.options[best], best_index=best,
                options=[OptionOut(option=o, n_tokens=0, logprob_sum=s, logprob_mean=s, logprob_uncond=None,
                                   score=math.log(max(p, 1e-12)), probability=p)
                         for o, s, p in zip(req.options, sums, probs)],
                timing={"lora_mode": 1.0, **state["lora"].last_timing},
            )
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        probs = _head_probs("choice", req.context, req.options)
        if probs is not None:  # head mode: softmax over head logits; score=logit-equivalent log-prob
            import math

            best = max(range(len(probs)), key=lambda i: probs[i])
            return ScoreResponse(
                best=req.options[best],
                best_index=best,
                options=[OptionOut(option=o, n_tokens=0, logprob_sum=math.log(max(p, 1e-12)),
                                   logprob_mean=math.log(max(p, 1e-12)), logprob_uncond=None,
                                   score=math.log(max(p, 1e-12)), probability=p)
                         for o, p in zip(req.options, probs)],
                timing={"head_mode": 1.0, **{k: float(v) for k, v in scorer.last_timing.items()}},
            )
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
        """TypeSafe System One contract: state + typed questions -> typed answers (docs.typesafe.ai).

        Per-question head mode (Route A): when a checkpoint is configured for the
        question type, the prompt is rendered with the same systemone renderers
        used at train time and answered from the head softmax instead of zero-shot
        next-token likelihoods. NOTE: heads must be trained on these rendered
        prompts; a head trained on a different rendering serves answers that are
        well-formed but off-distribution.
        """
        if "engine" in state:  # jev / laya: answered by the engine, no local scorer
            try:
                return state["engine"].answer(req)
            except Exception as e:  # upstream HTTP errors, missing key, laya failures
                raise HTTPException(502, f"engine {engine!r} failed: {e}")
        if remote:
            raise HTTPException(503, "engine still loading")
        scorer: OptionScorer = state.get("scorer")
        lora_mode = "lora" in state
        if scorer is None and not lora_mode:
            raise HTTPException(503, "model still loading")
        if not state.get("heads") and not lora_mode:
            try:
                return system_one(scorer, req, model_name=req.model or model_name)
            except ValueError as e:
                raise HTTPException(400, str(e))
        from .systemone import ChoiceQuestion, NoulQuestion, ScoreQuestion

        answers: dict = {}
        in_tok = out_tok = 0
        for qid, q in req.questions.items():
            if isinstance(q, ChoiceQuestion):
                prompt, labels = render_choice(req.state, q)
                qtype = "choice"
            elif isinstance(q, ScoreQuestion):
                prompt, labels = render_score(req.state, q)
                qtype = "score"
            elif isinstance(q, NoulQuestion):
                prompt, labels = render_noul(req.state, q)
                qtype = "noul"
            else:
                raise HTTPException(400, f"unknown question type for {qid!r}")
            if lora_mode:
                _, probs = _lora_probs(qtype, prompt, labels)
                in_tok += int(state["lora"].last_timing["context_tokens"])
                out_tok += int(state["lora"].last_timing["option_tokens"])
            else:
                probs = _head_probs(qtype, prompt, labels)
            if probs is None:  # no head for this type: fall back to zero-shot
                try:
                    res = scorer.score(prompt, labels, norm="sum", chat=False, sep="")
                except ValueError as e:
                    raise HTTPException(400, str(e))
                probs = [r.probability for r in res]
                in_tok += int(scorer.last_timing["context_tokens"])
                out_tok += int(scorer.last_timing["option_tokens"])
            if isinstance(q, ChoiceQuestion):
                best = max(range(len(labels)), key=lambda i: probs[i])
                answers[qid] = ChoiceAnswer(choice=labels[best], probabilities=dict(zip(labels, probs)),
                                            confidence=confidence(probs))
            elif isinstance(q, ScoreQuestion):
                answers[qid] = ScoreAnswer(score=sum(i * p for i, p in enumerate(probs)),
                                           confidence=confidence(probs),
                                           legend={str(i): c for i, c in enumerate(q.criteria)},
                                           probabilities={str(i): p for i, p in enumerate(probs)})
            else:
                answers[qid] = NoulAnswer(noul=probs[0])
        usage = {"input_tokens": in_tok, "output_tokens": out_tok}
        return SystemOneResponse(model=req.model or model_name, answers=answers,
                                 usage=usage)

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int, backend: str | None = None,
          device: str = "auto", quantize: str | None = None, heads: dict | None = None,
          lora: dict | None = None, engine: str | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, batch_size, backend, device, quantize, heads, lora, engine),
                host=host, port=port, workers=1)
