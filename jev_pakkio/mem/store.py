"""Memory plane for Jev-Mem: canonical nodes, four typed edge views, vector + lexical indexes.

All relation views share the same nodes; several typed edges may connect one pair (paper eq 5-6).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

GRAPHS = ("semantic", "temporal", "causal", "entity")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_ENTITY = re.compile(r"\b[A-Z][a-z]{2,}\b")
_STOP = {"The", "This", "That", "And", "But", "When", "What", "Why", "How", "Who", "Where", "Because"}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def guess_entities(text: str) -> list[str]:
    """Cheap capitalised-word heuristic; pass explicit entities for anything real."""
    return sorted({m for m in _ENTITY.findall(text) if m not in _STOP})


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...  # (n, d), L2-normalised


class HashEmbedder:
    """Dependency-free bag-of-words hashing embedder. For tests and offline use only."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokens(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class SentenceTransformerEmbedder:
    """Needs the optional extra: `uv sync --extra mem`."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


@dataclass
class Node:
    id: str
    content: str
    timestamp: float | None = None  # unix seconds; when the statement was observed
    entities: list[str] = field(default_factory=list)
    provenance: str = ""
    types: dict[str, float] = field(default_factory=dict)  # episodic/semantic/procedural/preference


@dataclass
class Edge:
    src: str
    dst: str
    kind: str  # one of GRAPHS
    weight: float = 1.0
    relation: str = ""  # e.g. "before", "caused_by", "shared_entity"


class MemoryStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []
        self.vecs: list[np.ndarray] = []
        self.edges: list[Edge] = []
        self.adj: dict[str, list[Edge]] = {}
        self.decisions: list[dict] = []  # consolidation records
        self._df: dict[str, int] = {}
        self._tf: dict[str, dict[str, int]] = {}
        self._len: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    # ---------------------------------------------------------------- nodes
    def add_node(self, node: Node, vec: np.ndarray | None = None) -> Node:
        self.nodes[node.id] = node
        self.order.append(node.id)
        self.vecs.append(vec if vec is not None else self.embedder.encode([node.content])[0])
        toks = tokens(node.content)
        self._len[node.id] = len(toks)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self._tf[node.id] = tf
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1
        self.adj.setdefault(node.id, [])
        return node

    def vec(self, nid: str) -> np.ndarray:
        return self.vecs[self.order.index(nid)]

    # ---------------------------------------------------------------- edges
    def add_edge(self, e: Edge) -> bool:
        if e.kind not in GRAPHS or e.src == e.dst:
            return False
        for old in self.adj[e.src]:
            if old.kind == e.kind and old.relation == e.relation and {old.src, old.dst} == {e.src, e.dst}:
                return False
        self.edges.append(e)
        self.adj[e.src].append(e)
        self.adj[e.dst].append(e)
        return True

    def neighbors(self, nid: str, kind: str) -> list[tuple[str, Edge]]:
        return [(e.dst if e.src == nid else e.src, e) for e in self.adj.get(nid, []) if e.kind == kind]

    # -------------------------------------------------------------- search
    def cosine(self, q: np.ndarray, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.vecs:
            return []
        sims = np.stack(self.vecs) @ q
        ranked = [(nid, float(s)) for nid, s in zip(self.order, sims) if not exclude or nid not in exclude]
        return sorted(ranked, key=lambda x: -x[1])

    def bm25(self, text: str, exclude: set[str] | None = None, k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
        n = len(self.nodes)
        if not n:
            return []
        avg = sum(self._len.values()) / n
        q = set(tokens(text))
        out = []
        for nid in self.order:
            if exclude and nid in exclude:
                continue
            tf, s = self._tf[nid], 0.0
            for t in q & tf.keys():
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * self._len[nid] / max(avg, 1e-9)))
            if s > 0:
                out.append((nid, s))
        return sorted(out, key=lambda x: -x[1])

    def rrf(self, text: str, top: int, kappa: int = 60) -> list[tuple[str, float]]:
        """Reciprocal-rank fusion of vector and lexical rankings (paper eq 16)."""
        lists = [self.cosine(self.embedder.encode([text])[0]), self.bm25(text)]
        score: dict[str, float] = {}
        for L in lists:
            for rank, (nid, _) in enumerate(L[: max(top * 4, 20)], start=1):
                score[nid] = score.get(nid, 0.0) + 1.0 / (kappa + rank)
        return sorted(score.items(), key=lambda x: -x[1])[:top]

    # ---------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        data = {
            "nodes": [asdict(self.nodes[i]) for i in self.order],
            "vecs": [v.tolist() for v in self.vecs],
            "edges": [asdict(e) for e in self.edges],
            "decisions": self.decisions,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, embedder: Embedder) -> "MemoryStore":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        s = cls(embedder)
        for nd, v in zip(data["nodes"], data["vecs"]):
            s.add_node(Node(**nd), vec=np.asarray(v, dtype=np.float32))
        for e in data["edges"]:
            s.add_edge(Edge(**e))
        s.decisions = data.get("decisions", [])
        return s
