"""MCP server: noul, rate, classify and jev scoring tools over streamable-http,
plus async LoRA training and status.

    jev_pakkio mcp --port 5001

`jev` takes `cases: [{"context", "options"}, ...]` and races each case against
TypeSafe's hosted Jev (api.typesafe.ai/v1/systemone, model jev-latest; needs
`TYPESAFE_API_KEY` in env, ../.env or ./.env, plus the optional `typesafe-sdk`
dependency -- `pip install jev_pakkio[typesafe]`) and the local model (through
the normal auto-VRAM model sizing, see scorer_torch.auto_model) concurrently.
The merged answer is always jev's when jev succeeds; the local model only
takes over if jev itself errors.

The port requires `Authorization: Bearer <password>` on every call (default
"pakkio 62", override with JEV_MCP_PASSWORD).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from starlette.responses import JSONResponse

from .server import _auto_model, _default_model, _get_scorer_class, _resolve_backend, _resolve_quantize
from .systemone import (
    ChoiceQuestion,
    Entry,
    NoulQuestion,
    ScoreQuestion,
    confidence,
    render_choice,
    render_noul,
    render_score,
)

mcp = MCPServer("jev_pakkio")
_STATE: dict[str, Any] = {}
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def configure(model_path: str | None = None, backend: str | None = None, device: str = "auto",
             quantize: str | None = None) -> None:
    """Set the model/backend every tool below scores with. Called once from
    `jev_pakkio mcp` before `mcp.run`; the scorer itself loads lazily on first use.
    """
    _STATE["configured"] = True
    _STATE["backend"] = _resolve_backend(backend)
    _STATE["device"] = device
    _STATE["quantize"] = _resolve_quantize(quantize)
    _STATE["model_path"] = model_path or os.environ.get("OPENJEV_MODEL") or _default_model(_STATE["backend"])


def _unload_scorer() -> None:
    """Drop the loaded scorer and release its VRAM, so a training job can have the card.

    Deleting the Python reference alone leaves the memory in torch's caching
    allocator; empty_cache() is what actually gives it back to nvidia-smi.
    """
    if "scorer" in _STATE:
        del _STATE["scorer"]
        import gc

        gc.collect()
        if _STATE.get("backend") == "torch":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _scorer():
    """The one shared scorer, loaded on first tool call.

    Refuses to (re)load while a training job is running: training already
    unloaded this scorer for its own VRAM (see train_lora), and reloading here
    would just race it back into an OOM on an 8GB card.
    """
    if "configured" not in _STATE:
        configure()
    if _STATE.get("training_job"):
        raise RuntimeError(
            f"training job {_STATE['training_job']!r} is running; scoring is paused to keep "
            "its VRAM free -- check status() and retry once it finishes"
        )
    if "scorer" not in _STATE:
        backend = _STATE["backend"]
        model_path = _STATE["model_path"]
        if model_path == "auto":
            model_path = _auto_model(backend)
        OptionScorer = _get_scorer_class(backend)
        kwargs = {"quantize": _STATE["quantize"]} if backend == "torch" else {}
        if backend == "torch" and _STATE["device"] != "auto":
            kwargs["device"] = _STATE["device"]
        scorer = OptionScorer(model_path, **kwargs)
        scorer.score("warm up", ["a", "b"])  # compile kernels before the first real call
        _STATE["scorer"] = scorer
        _STATE["model_path"] = model_path
    return _STATE["scorer"]


def _ask(prompt: str, labels: list[str]) -> list[float]:
    res = _scorer().score(prompt, labels, norm="sum", chat=False, sep="")
    return [r.probability for r in res]


@mcp.tool()
def classify(state: Entry, criteria: dict[str, Entry], instructions: Entry = None) -> dict:
    """Choose exactly one of 2+ named criteria for `state` (TypeSafe "choice" question)."""
    q = ChoiceQuestion(type="choice", instructions=instructions, criteria=criteria)
    prompt, labels = render_choice(state, q)
    probs = _ask(prompt, labels)
    best = max(range(len(labels)), key=lambda i: probs[i])
    return {"choice": labels[best], "probabilities": dict(zip(labels, probs)), "confidence": confidence(probs)}


@mcp.tool()
def rate(state: Entry, criteria: list[Entry], instructions: Entry = None) -> dict:
    """Rate `state` on an ordered list of 2+ levels; returns the probability-weighted level."""
    q = ScoreQuestion(type="score", instructions=instructions, criteria=criteria)
    prompt, labels = render_score(state, q)
    probs = _ask(prompt, labels)
    return {
        "score": sum(i * p for i, p in enumerate(probs)),
        "confidence": confidence(probs),
        "legend": {str(i): c for i, c in enumerate(criteria)},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
    }


@mcp.tool()
def noul(state: Entry, true_criteria: Entry = None, false_criteria: Entry = None,
        instructions: Entry = None) -> dict:
    """Answer yes/no on `state`; returns P(yes) in [0, 1] (TypeSafe "noul" question)."""
    crit = {k: v for k, v in (("true", true_criteria), ("false", false_criteria)) if v is not None}
    q = NoulQuestion(type="noul", instructions=instructions, criteria=crit or None)
    prompt, labels = render_noul(state, q)
    probs = _ask(prompt, labels)
    return {"noul": probs[0]}


_LOCAL_SCORER_LOCK = threading.Lock()


def _jev(context: str, options: list[str], norm: str = "mean") -> dict:
    """Locked: the one shared local scorer isn't safe under concurrent forward
    passes, and jev_combined races it against jev's network call on its own
    thread per case."""
    with _LOCAL_SCORER_LOCK:
        scorer = _scorer()
        res = scorer.score(context, options, norm=norm)
    best = max(range(len(res)), key=lambda i: res[i].score)
    return {"best": res[best].option, "best_index": best, "model": _STATE["model_path"],
            "options": [r.to_dict() for r in res]}


def _load_typesafe_key() -> str:
    """TYPESAFE_API_KEY from the environment, then ../.env, then ./.env (same
    lookup compare_jev.py uses). Never logged or returned to the caller."""
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    root = Path(__file__).resolve().parent.parent
    for path in (root.parent / ".env", root / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return ""


def _jev_typesafe(context: str, options: list[str]) -> dict:
    """One System One `choice` question against TypeSafe's hosted Jev."""
    key = _load_typesafe_key()
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY not found (env, ../.env or ./.env) -- jev calls TypeSafe's "
            "hosted Jev and needs it; use jev-pakkio for the local model instead"
        )
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError as exc:
        raise RuntimeError(
            "typesafe-sdk not installed -- jev calls TypeSafe's hosted Jev and needs it "
            "(pip install jev_pakkio[typesafe]); use jev-pakkio for the local model instead"
        ) from exc

    client = TypeSafeClient(api_key=key)
    resp = client.system_one(
        state=context,
        model="jev-latest",
        questions={"pick": {
            "type": "choice",
            "instructions": "Which option best continues or answers the state?",
            "criteria": {opt: opt for opt in options},
        }},
    )
    answer = resp.answers["pick"]
    probs = dict(getattr(answer, "probabilities", None) or {})
    best_index = options.index(answer.choice) if answer.choice in options else max(
        range(len(options)), key=lambda i: probs.get(options[i], 0.0))
    return {
        "best": answer.choice,
        "best_index": best_index,
        "model": "jev-latest",
        "options": [{"option": opt, "probability": probs.get(opt)} for opt in options],
    }


def _run_cases(cases: list[dict[str, Any]], one: Callable[[dict[str, Any]], dict],
               concurrent: bool) -> dict:
    """Run `one` over every case, isolating failures per case, and time it so
    concurrency is measurable rather than assumed. `concurrent=True` fans the
    calls out on threads (worth it when `one` is a network round trip);
    otherwise cases run one at a time, as they must against the one shared
    local scorer. `wall_ms` is the actual elapsed time for the whole batch;
    `sum_case_ms` is what it would have cost run serially -- their ratio is
    the speedup concurrency bought (1.0x when cases ran one at a time)."""
    results: list[dict] = [{}] * len(cases)
    case_ms: list[float] = [0.0] * len(cases)

    def run(i: int, case: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        try:
            results[i] = one(case)
        except Exception as exc:
            results[i] = {"error": f"{type(exc).__name__}: {exc}"}
        case_ms[i] = (time.perf_counter() - t0) * 1000

    t_start = time.perf_counter()
    if concurrent and len(cases) > 1:
        threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(cases)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for i, c in enumerate(cases):
            run(i, c)
    wall_ms = (time.perf_counter() - t_start) * 1000

    for r, ms in zip(results, case_ms):
        r["case_ms"] = round(ms, 1)
    sum_ms = sum(case_ms)
    return {
        "concurrent": concurrent,
        "wall_ms": round(wall_ms, 1),
        "sum_case_ms": round(sum_ms, 1),
        "speedup": round(sum_ms / wall_ms, 2) if wall_ms else None,
        "results": results,
    }


def _jev_combined_one(case: dict[str, Any], norm: str, confidence_threshold: float) -> dict:
    """Local-first: answer from the local GPU/CPU model (jev-pakkio) unless it's
    uncertain or fails, in which case fall back to TypeSafe's hosted Jev as the
    tie-breaker. Keeps every request off the network and off TypeSafe's bill
    unless the local model actually needs help."""
    context, options = case["context"], case["options"]
    out: dict[str, Any] = {}

    try:
        local = _jev(context, options, case.get("norm", norm))
        out["jev_pakkio"] = local
        probs = [o["probability"] for o in local["options"]]
        local_confidence = confidence(probs)
        out["local_confidence"] = round(local_confidence, 3)
    except Exception as exc:
        local = None
        out["jev_pakkio_error"] = f"{type(exc).__name__}: {exc}"
        local_confidence = -1.0

    hosted = None
    if local is None or local_confidence < confidence_threshold:
        try:
            hosted = _jev_typesafe(context, options)
            out["jev"] = hosted
        except Exception as exc:
            out["jev_error"] = f"{type(exc).__name__}: {exc}"

    if local is not None and hosted is None and "jev_error" not in out:
        out["best"], out["best_index"], out["source"] = local["best"], local["best_index"], "jev-pakkio"
        out["fallback_reason"] = None
    elif hosted is not None:
        out["best"], out["best_index"], out["source"] = hosted["best"], hosted["best_index"], "jev"
        out["fallback_reason"] = "local uncertain" if local is not None else "local failed"
    elif local is not None:
        out["best"], out["best_index"], out["source"] = local["best"], local["best_index"], "jev-pakkio"
        out["fallback_reason"] = "uncertain, jev unavailable"
    else:
        out["best"], out["best_index"], out["source"] = None, None, "none"
        out["fallback_reason"] = "both unavailable"
    out["agree"] = (hosted["best"] == local["best"]) if hosted and local else None
    return out


@mcp.tool(name="jev")
def jev(cases: list[dict[str, Any]], norm: str = "mean", confidence_threshold: float = 0.85) -> dict:
    """Score one or more cases, local-first: each case answers from the local
    model (sized to this machine's VRAM, GPU if present) and only calls
    TypeSafe's hosted Jev (api.typesafe.ai, model jev-latest; needs
    TYPESAFE_API_KEY in env, ../.env or ./.env plus the optional `typesafe-sdk`
    dependency) when the local answer is uncertain -- its top probability
    below `confidence_threshold` (1 - normalised entropy over the options) --
    or when the local model itself fails. That keeps most requests local and
    free; the hosted call is the tie-breaker, not the default. `source` is
    always exactly `"jev"` or `"jev-pakkio"` (whichever answered); why it fell
    back, if it did, is in `fallback_reason` (`None` when local answered
    outright). `agree` says whether the two would have matched when both ran.

    Each case is `{"context": str, "options": [str, ...], "norm": str
    (optional, overrides the local model's norm)}`; pass several to test many
    things in one call -- they run concurrently (local calls serialize safely
    against the one shared scorer; hosted calls, when needed, overlap with
    everything else). Returns `{"wall_ms", "sum_case_ms", "speedup",
    "results": [...]}` so concurrency is measurable, not assumed."""
    return _run_cases(
        cases, lambda c: _jev_combined_one(c, norm, confidence_threshold), concurrent=True
    )


# ------------------------------------------------------------- async LoRA training
def _lora_job(job_id: str) -> None:
    job = _JOBS[job_id]
    kwargs = job["kwargs"]
    out = kwargs["out"]
    Path(out).mkdir(parents=True, exist_ok=True)
    log_path = Path(out) / "mcp_train.log"
    cmd = [sys.executable, "-m", "jev_pakkio.cli", "lora", kwargs["train"],
           "--validation", kwargs["validation"], "--out", out, "--model", kwargs["model"],
           "--quantize", kwargs["quantize"], "--rank", str(kwargs["rank"]),
           "--alpha", str(kwargs["alpha"]), "--epochs", str(kwargs["epochs"]),
           "--learning-rate", str(kwargs["lr"]), "--sep", kwargs["sep"]]
    if kwargs.get("test"):
        cmd += ["--test", kwargs["test"]]
    if kwargs.get("chat"):
        cmd.append("--chat")
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            with _JOBS_LOCK:
                job["pid"], job["log"] = proc.pid, str(log_path)
            proc.wait()
        with _JOBS_LOCK:
            job["status"] = "done" if proc.returncode == 0 else "failed"
            job["returncode"] = proc.returncode
            job["finished_at"] = time.time()
    finally:
        # Always release the slot, even if the subprocess never started -- otherwise
        # a crash here would leave every scoring tool permanently refusing to load.
        _STATE["training_job"] = None


@mcp.tool()
def train_lora(train: str, validation: str, out: str | None = None, model: str | None = None,
               test: str | None = None, quantize: str = "4bit", rank: int = 16, alpha: int = 32,
               epochs: int = 1, learning_rate: float = 2e-4, sep: str = "\nChoice: ",
               chat: bool = False) -> dict:
    """Start LoRA fine-tuning on a JSONL dataset ({"context", "options", "label"} rows)
    in its own subprocess; returns immediately with a job_id. Poll `status(job_id)` for
    progress -- training is asynchronous. Unloads this process's scorer first to free its
    VRAM for the job (an 8GB card can't hold both); scoring tools refuse to reload until
    the job finishes, then pick the model back up lazily on the next call.
    """
    if _STATE.get("training_job"):
        return {"error": f"training job {_STATE['training_job']!r} is already running; "
                         "wait for it to finish (one job at a time on this GPU)"}
    job_id = uuid.uuid4().hex[:12]
    kwargs = dict(train=train, validation=validation, test=test,
                  out=out or f"runs/lora-{job_id}", model=model or _STATE.get("model_path") or "auto",
                  quantize=quantize, rank=rank, alpha=alpha, epochs=epochs, lr=learning_rate,
                  sep=sep, chat=chat)
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "kwargs": kwargs, "started_at": time.time()}
    _STATE["training_job"] = job_id
    _unload_scorer()
    threading.Thread(target=_lora_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "out": kwargs["out"], "status": "running", "freed_vram": True}


def _job_summary(job_id: str, job: dict) -> dict:
    out = {"job_id": job_id, "status": job["status"], "out": job["kwargs"]["out"],
           "started_at": job["started_at"]}
    if "returncode" in job:
        out["returncode"] = job["returncode"]
    log_path = job.get("log")
    if log_path and Path(log_path).exists():
        out["log_tail"] = Path(log_path).read_text().splitlines()[-15:]
    result_path = Path(job["kwargs"]["out"]) / "jev_pakkio_lora.json"
    if result_path.exists():
        out["result"] = json.loads(result_path.read_text())
    return out


@mcp.tool()
def status(job_id: str | None = None) -> dict:
    """Status of one `train_lora` job, or every job started in this process when job_id is omitted."""
    with _JOBS_LOCK:
        if job_id is not None:
            job = _JOBS.get(job_id)
            return _job_summary(job_id, job) if job else {"error": f"unknown job_id {job_id!r}"}
        return {jid: _job_summary(jid, j) for jid, j in _JOBS.items()}


# ------------------------------------------------------------------------ auth
DEFAULT_MCP_PASSWORD = "pakkio 62"


def _load_api_key() -> str:
    """The password every MCP call must present as `Authorization: Bearer <password>`.

    Override with JEV_MCP_PASSWORD; otherwise the agreed default below.
    """
    return os.environ.get("JEV_MCP_PASSWORD", "").strip() or DEFAULT_MCP_PASSWORD


class _BearerAuth:
    """ASGI middleware: require `Authorization: Bearer <key>` on every HTTP request.

    Lifespan and other non-"http" scopes pass straight through, so uvicorn still
    drives the MCP session manager's startup/shutdown via the wrapped app.
    """

    def __init__(self, app, key: str) -> None:
        self.app, self.key = app, key

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            if headers.get(b"authorization", b"").decode() != f"Bearer {self.key}":
                await JSONResponse({"error": "invalid or missing API key"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def run(host: str = "127.0.0.1", port: int = 5001, model_path: str | None = None,
        backend: str | None = None, device: str = "auto", quantize: str | None = None) -> None:
    import uvicorn

    configure(model_path, backend, device, quantize)
    app = mcp.streamable_http_app(host=host)
    key = _load_api_key()
    if key:
        app = _BearerAuth(app, key)
    else:
        print("warning: TYPESAFE_API_KEY not found (env, ../.env or ./.env) -- "
              "MCP port is unauthenticated", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
