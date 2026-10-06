"""Engine-choosing tools for the single `jev` MCP server (registered onto it by mcp_server.py).

Tools
  list_engines()                                   the engine names and what backs them (incl. LoRA adapters)
  list_adapters()                                  LoRA adapters per local engine (OPENJEV_LORA_4G / _8G)
  ask(state, questions, engine="laya", adapter=)   one engine, System One response
  compare(state, questions, engines=[...])         the same request on several engines, with latency
  memory_add(text, store=, engine=, timestamp=)    write an observation into a named Jev-Mem store
  memory_query(query, store=, engine=, top_k=)     retrieve evidence from a store (engine = System-One controller)
  memory_stores()                                  stores on disk (JEVMEM_DIR, default ./jevmem)

Engines load on first use; only one GPU engine (laya/4g/8g) stays resident, loading another evicts it.
"""
from __future__ import annotations

import os
import time
from typing import Any, Literal

# Must be set before torch initialises CUDA. Without it the allocator keeps 8g's weights in one huge segment that a
# few live MiB pin, so evicting 8g frees nothing and the next engine OOMs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from pydantic import TypeAdapter

from .engines import ENGINE_NAMES, LOCAL_PRESETS, Engine, SystemOneRequest, get_engine, local_preset, lora_adapters

try:  # mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

EngineName = Literal["jev", "mercury", "laya", "4g", "8g"]
_loaded: dict[str, Engine] = {}
_memory = None


_GPU_ENGINES = {"laya", *LOCAL_PRESETS}  # the API engines (jev, mercury) hold no VRAM and stay cached


def _evict_gpu_engines() -> None:
    """Drop every loaded GPU engine and hand its VRAM back; laya + 4g + 8g together overflow an 8 GB card."""
    for name in [n for n in _loaded if n in _GPU_ENGINES]:
        eng = _loaded.pop(name)
        if hasattr(eng, "close"):
            eng.close()  # dropping the reference alone leaves 4g/8g weights on the GPU
        del eng
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _engine(name: str) -> Engine:
    if name not in _loaded:
        if name in _GPU_ENGINES:
            _evict_gpu_engines()  # one GPU engine resident at a time
        _loaded[name] = get_engine(name)
    return _loaded[name]


def _memory_service():
    global _memory
    if _memory is None:  # imported lazily: needs the `mem` extra (sentence-transformers)
        from .mem.service import MemoryService
        from .mem.store import SentenceTransformerEmbedder

        _memory = MemoryService(_engine, SentenceTransformerEmbedder)
    return _memory


def _request(state: Any, questions: dict[str, Any]) -> SystemOneRequest:
    return TypeAdapter(SystemOneRequest).validate_python({"state": state, "questions": questions})


INSTRUCTIONS = (
    "Answer typed questions about a piece of text. questions maps an id to "
    '{"type": "choice", "instructions": ..., "criteria": {option: description}}, '
    '{"type": "score", "instructions": ..., "criteria": [level descriptions]} or {"type": "noul", "instructions": ...}. '
    "engine picks the model: jev (TypeSafe API), mercury (Mercury Decide, free on OpenRouter), laya (fast encoder), 4g (Qwen, 4 GB GPU), 8g (Gemma E2B, 8 GB GPU).")


def register_tools(mcp: _Server) -> None:

    @mcp.tool()
    def list_engines() -> dict[str, str]:
        """The available engines and the model behind each."""
        info = {"jev": "TypeSafe hosted Jev (needs TYPESAFE_API_KEY)",
                "mercury": "Inception Mercury Decide via OpenRouter, free tier (needs OPENROUTER_API_KEY)", "laya": "convaiinnovations/laya encoder"}
        for name in LOCAL_PRESETS:
            model, quant = local_preset(name)
            adapters = lora_adapters(name)
            info[name] = f"{model} ({quant})" + (f" + LoRA {sorted(adapters)}" if adapters else "")
        return info

    @mcp.tool()
    def list_adapters() -> dict[str, dict[str, str]]:
        """LoRA adapters configured per local engine (OPENJEV_LORA_4G / OPENJEV_LORA_8G): name -> path."""
        return {name: lora_adapters(name) for name in LOCAL_PRESETS}

    @mcp.tool()
    def ask(state: str | dict | list, questions: dict[str, Any], engine: EngineName = "laya",
            adapter: str | None = None) -> dict:
        """Answer typed questions about `state` with the chosen engine; returns the System One response.

        adapter: a LoRA adapter name from list_adapters (4g/8g only), used for every question. Default: the
        adapter named after each question's type (choice/score/noul) if loaded, else the base model."""
        try:  # a plain exception reaches the client as just "Error executing tool ask"; ToolError carries the reason
            if adapter and engine not in LOCAL_PRESETS:
                raise ValueError(f"engine {engine!r} has no LoRA adapters; adapters are for {', '.join(LOCAL_PRESETS)}")
            eng = _engine(engine)
            req = _request(state, questions)
            resp = eng.answer(req, adapter=adapter) if adapter else eng.answer(req)
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e
        return resp.model_dump(mode="json")

    @mcp.tool()
    def compare(state: str | dict | list, questions: dict[str, Any],
                engines: list[EngineName] = list(ENGINE_NAMES), adapter: str | None = None) -> dict[str, dict]:
        """Run the same request on several engines; each entry has latency_s and answers, or an error.

        adapter is applied only to engines that have it loaded (4g/8g)."""
        req = _request(state, questions)
        out: dict[str, dict] = {}
        for name in engines:
            try:
                eng = _engine(name)
                t = time.perf_counter()
                resp = eng.answer(req, adapter=adapter) if adapter and name in LOCAL_PRESETS else eng.answer(req)
                out[name] = {"latency_s": round(time.perf_counter() - t, 3),
                             "answers": resp.model_dump(mode="json")["answers"]}
            except Exception as e:  # one engine failing must not hide the others
                out[name] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                eng = resp = None  # a live local would keep this engine's VRAM pinned while the next one loads
        return out

    @mcp.tool()
    def memory_add(text: str, store: str = "default", engine: EngineName = "jev", timestamp: str | None = None,
                   entities: list[str] | None = None) -> dict:
        """Write one observation into a Jev-Mem store (created on first use) and persist it.

        The engine is the System-One controller that types the observation and judges its relations to similar
        memories. timestamp: ISO time the statement was observed (UTC if no zone). entities: ids shared across
        memories; default is a capitalised-word heuristic."""
        return _memory_service().add(text, store, engine, timestamp, entities)

    @mcp.tool()
    def memory_query(query: str, store: str = "default", engine: EngineName = "jev", top_k: int = 10) -> dict:
        """Retrieve the evidence most relevant to `query` from a Jev-Mem store.

        The engine routes the query over the semantic/temporal/causal/entity graphs, scores candidates and decides
        when to stop. Returns evidence (id, score, content) and a trace of the routing, rounds and stop reason.
        Answer synthesis is left to the caller."""
        return _memory_service().query(query, store, engine, top_k)

    @mcp.tool()
    def memory_stores() -> dict[str, dict]:
        """Jev-Mem stores found on disk, with node and edge counts for those loaded in this process."""
        return _memory_service().stores()
