"""MCP server: ask typed System One questions, choosing the engine on every call.

    openjev mcp                      # stdio, for Claude Code / Claude Desktop
    claude mcp add openjev -- uv run openjev mcp

Tools
  list_engines()                                   the engine names and what backs them
  ask(state, questions, engine="laya")             one engine, System One response
  compare(state, questions, engines=[...])         the same request on several engines, with latency

Engines load on first use and stay loaded, so only the ones you call cost memory.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import TypeAdapter

from .engines import ENGINE_NAMES, LOCAL_PRESETS, Engine, SystemOneRequest, get_engine, local_preset

try:  # mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

EngineName = Literal["jev", "laya", "4g", "8g"]
_loaded: dict[str, Engine] = {}


def _engine(name: str) -> Engine:
    if name not in _loaded:
        _loaded[name] = get_engine(name)
    return _loaded[name]


def _request(state: Any, questions: dict[str, Any]) -> SystemOneRequest:
    return TypeAdapter(SystemOneRequest).validate_python({"state": state, "questions": questions})


def build_server() -> _Server:
    mcp = _Server("openjev", instructions=(
        "Answer typed questions about a piece of text. questions maps an id to "
        '{"type": "choice", "instructions": ..., "criteria": {option: description}}, '
        '{"type": "score", "criteria": [level descriptions]} or {"type": "noul", "instructions": ...}. '
        "engine picks the model: jev (TypeSafe API), laya (fast encoder), 4g (Qwen, 4 GB GPU), 8g (Gemma E2B, 8 GB GPU)."))

    @mcp.tool()
    def list_engines() -> dict[str, str]:
        """The available engines and the model behind each."""
        info = {"jev": "TypeSafe hosted Jev (needs TYPESAFE_API_KEY)", "laya": "convaiinnovations/laya encoder"}
        for name in LOCAL_PRESETS:
            model, quant = local_preset(name)
            info[name] = f"{model} ({quant})"
        return info

    @mcp.tool()
    def ask(state: str | dict | list, questions: dict[str, Any], engine: EngineName = "laya") -> dict:
        """Answer typed questions about `state` with the chosen engine; returns the System One response."""
        return _engine(engine).answer(_request(state, questions)).model_dump(mode="json")

    @mcp.tool()
    def compare(state: str | dict | list, questions: dict[str, Any],
                engines: list[EngineName] = list(ENGINE_NAMES)) -> dict[str, dict]:
        """Run the same request on several engines; each entry has latency_s and answers, or an error."""
        req = _request(state, questions)
        out: dict[str, dict] = {}
        for name in engines:
            try:
                eng = _engine(name)
                t = time.perf_counter()
                resp = eng.answer(req)
                out[name] = {"latency_s": round(time.perf_counter() - t, 3),
                             "answers": resp.model_dump(mode="json")["answers"]}
            except Exception as e:  # one engine failing must not hide the others
                out[name] = {"error": f"{type(e).__name__}: {e}"}
        return out

    return mcp


def run() -> None:
    build_server().run()  # stdio
