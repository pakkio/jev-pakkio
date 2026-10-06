"""System-One control plane: typed questions in, probabilities out. Nothing is generated.

A controller is anything with `ask(state, questions) -> {id: answer_dict}` following the
`/v1/systemone` wire format. `HTTPController` talks to `jev_pakkio serve`; `LocalController` wraps an
in-process scorer. Decisions that share one state go out as a single batched call.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol


class Controller(Protocol):
    calls: int

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, dict]: ...


def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


class HTTPController:
    def __init__(self, url: str = "http://127.0.0.1:8000", api_key: str | None = None, timeout: float = 120.0,
                 model: str | None = None):
        self.url, self.api_key, self.timeout, self.model, self.calls = url.rstrip("/"), api_key, timeout, model, 0

    def ask(self, state, questions):
        if not questions:
            return {}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url + "/v1/systemone",
            data=json.dumps({"state": state, "questions": questions, **({"model": self.model} if self.model else {})}).encode(),
            headers=headers,
        )
        self.calls += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["answers"]


class LocalController:
    def __init__(self, scorer, model_name: str = "local", norm: str = "sum"):
        self.scorer, self.model_name, self.norm, self.calls = scorer, model_name, norm, 0

    def ask(self, state, questions):
        if not questions:
            return {}
        from ..systemone import SystemOneRequest, system_one

        self.calls += 1
        resp = system_one(self.scorer, SystemOneRequest(state=state, questions=questions), self.model_name, self.norm)
        return {k: v.model_dump() for k, v in resp.answers.items()}


class EngineController:
    """Runs on any registered engine (jev, laya, 4g, 8g) in-process, no HTTP server needed."""

    def __init__(self, engine, adapter: str | None = None):
        self.engine, self.adapter, self.calls = engine, adapter, 0

    def ask(self, state, questions):
        if not questions:
            return {}
        from ..systemone import SystemOneRequest

        self.calls += 1
        req = SystemOneRequest.model_validate({"state": state, "questions": questions})
        resp = self.engine.answer(req, adapter=self.adapter) if self.adapter else self.engine.answer(req)
        return {k: v.model_dump() for k, v in resp.answers.items()}


def p(answers: dict[str, dict], qid: str, default: float = 0.0) -> float:
    a = answers.get(qid)
    return float(a["noul"]) if a else default
