"""Jev-Mem: System-One-controlled agentic memory (Jiang, Li, Li; arXiv:2609.23986).

    from jev_pakkio.mem import JevMem, HTTPController
    mem = JevMem(HTTPController("http://127.0.0.1:8000"))
    mem.add("Mira: My old bicycle broke.", timestamp=...)
    result = mem.query("When did Mira buy a new bicycle, and why?")
"""
from __future__ import annotations

from typing import Callable

from .control import Controller, HTTPController, LocalController
from .read import ReadConfig, Result, retrieve
from .store import Embedder, HashEmbedder, MemoryStore, Node, SentenceTransformerEmbedder
from .write import WriteConfig, write

__all__ = ["JevMem", "HTTPController", "LocalController", "ReadConfig", "WriteConfig", "Result", "MemoryStore",
           "Node", "HashEmbedder", "SentenceTransformerEmbedder"]


class JevMem:
    def __init__(self, controller: Controller, embedder: Embedder | None = None, store: MemoryStore | None = None,
                 write_cfg: WriteConfig | None = None, read_cfg: ReadConfig | None = None,
                 system_two: Callable[[str, list[Node]], str] | None = None):
        self.ctl = controller
        self.store = store or MemoryStore(embedder or SentenceTransformerEmbedder())
        self.write_cfg, self.read_cfg, self.system_two = write_cfg or WriteConfig(), read_cfg or ReadConfig(), system_two

    def add(self, text: str, timestamp: float | None = None, entities: list[str] | None = None,
            provenance: str = "", node_id: str | None = None) -> Node | None:
        return write(self.store, self.ctl, self.write_cfg, text, timestamp, entities, provenance, node_id)

    def query(self, query: str) -> Result:
        return retrieve(self.store, self.ctl, self.read_cfg, query, self.system_two)

    def save(self, path: str) -> None:
        self.store.save(path)

    @classmethod
    def load(cls, path: str, controller: Controller, embedder: Embedder | None = None, **kw) -> "JevMem":
        emb = embedder or SentenceTransformerEmbedder()
        return cls(controller, emb, MemoryStore.load(path, emb), **kw)
