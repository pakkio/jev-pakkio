"""Named, persistent Jev-Mem stores driven by a chosen engine. Used by the MCP memory tools.

Each store is one JSON file under `root`. The engine is only the controller, so it can differ between calls
on the same store.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

from . import EngineController, JevMem, MemoryStore, ReadConfig
from .store import Embedder

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class MemoryService:
    def __init__(self, get_engine: Callable[[str], Any], embedder: Callable[[], Embedder], root: str | None = None):
        self._get_engine, self._embedder_factory = get_engine, embedder
        self.root = root or os.environ.get("JEVMEM_DIR", "jevmem")
        self._embedder: Embedder | None = None
        self._mems: dict[str, JevMem] = {}

    def _path(self, store: str) -> str:
        if not _NAME.match(store):
            raise ValueError(f"store name must match {_NAME.pattern}, got {store!r}")
        return os.path.join(self.root, store + ".json")

    def _mem(self, store: str, engine: str, adapter: str | None = None) -> JevMem:
        path = self._path(store)
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        ctl = EngineController(self._get_engine(engine), adapter)
        if store not in self._mems:
            if os.path.exists(path):
                self._mems[store] = JevMem(ctl, self._embedder, MemoryStore.load(path, self._embedder))
            else:
                self._mems[store] = JevMem(ctl, self._embedder)
        mem = self._mems[store]
        mem.ctl = ctl  # the engine is per call; the store is not tied to it
        return mem

    def add(self, text: str, store: str = "default", engine: str = "jev", timestamp: str | None = None,
            entities: list[str] | None = None, adapter: str | None = None) -> dict:
        ts = None
        if timestamp:
            dt = datetime.fromisoformat(timestamp)
            ts = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
        mem = self._mem(store, engine, adapter)
        before = mem.ctl.calls
        node = mem.add(text, timestamp=ts, entities=entities)
        if node is not None:
            os.makedirs(self.root, exist_ok=True)
            mem.save(self._path(store))
        return {"id": node.id if node else None, "types": node.types if node else None, "nodes": len(mem.store),
                "edges": len(mem.store.edges), "controller_calls": mem.ctl.calls - before}

    def query(self, query: str, store: str = "default", engine: str = "jev", top_k: int = 10,
              adapter: str | None = None) -> dict:
        mem = self._mem(store, engine, adapter)
        mem.read_cfg = ReadConfig(top_k=top_k)
        res = mem.query(query)
        return {"evidence": [{"id": n.id, "score": round(s, 4), "content": n.content, "timestamp": n.timestamp}
                             for n, s in res.evidence], "trace": res.trace}

    def stores(self) -> dict[str, dict]:
        if not os.path.isdir(self.root):
            return {}
        out = {}
        for f in sorted(os.listdir(self.root)):
            if f.endswith(".json"):
                m = self._mems.get(f[:-5])
                out[f[:-5]] = {"loaded": m is not None, **({"nodes": len(m.store), "edges": len(m.store.edges)} if m else {})}
        return out
