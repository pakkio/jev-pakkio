"""Tiny HTTP client for the jev_pakkio server. Standard library only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Decision:
    probs: dict[str, float]  # action -> probability, in option order
    best: str
    latency_s: float
    context: str  # what the model actually read (for the record file / panel)


class ServerDown(RuntimeError):
    pass


class Client:
    def __init__(self, url: str, api: str = "score", api_key: str | None = None, norm: str = "mean") -> None:
        self.url = url.rstrip("/")
        self.api = api
        self.api_key = api_key or os.environ.get("OPENJEV_API_KEY")
        self.norm = norm

    def _post(self, path: str, body: dict, timeout: float = 60.0) -> dict:
        data = json.dumps(body).encode()
        headers = {"content-type": "application/json"}
        if self.api_key and path.startswith("/v1/"):
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
        except urllib.error.URLError as e:
            raise ServerDown(f"cannot reach {self.url}: {e.reason}") from None

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return json.loads(r.read())
        except urllib.error.URLError as e:
            raise ServerDown(f"cannot reach {self.url}: {e.reason}") from None

    # -- /score: plain context + options, the native jev_pakkio endpoint -------------------
    def decide_score(self, context: str, actions: list[str]) -> Decision:
        # Options carry their own leading space: a " " separator would become a lone space token,
        # which the tokenizer never produces mid-sentence and the model scores as garbage.
        res = self._post("/score", {"context": context, "options": [" " + a for a in actions],
                                    "norm": self.norm, "chat": False, "sep": ""})
        probs = {o["option"].strip(): o["probability"] for o in res["options"]}
        return Decision(probs=probs, best=res["best"].strip(), latency_s=res["timing"]["total_s"], context=context)

    # -- /v1/systemone: TypeSafe System One contract, one "choice" question ---------------
    def decide_systemone(self, state: dict, instructions: str, criteria: dict[str, str]) -> Decision:
        import time

        t = time.perf_counter()
        res = self._post("/v1/systemone", {
            "state": state,
            "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        })
        ans = res["answers"]["action"]
        probs = {a: ans["probabilities"][a] for a in criteria}
        ctx = json.dumps(state, indent=1) + f"\n\nconfidence={ans['confidence']:.2f} usage={res['usage']}"
        return Decision(probs=probs, best=ans["choice"], latency_s=time.perf_counter() - t, context=ctx)
