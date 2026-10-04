"""jevlike's AttentionHead, ported to MLX: one cross-attention layer that scores each option
against the context tokens. Options are queries, context tokens are keys/values, and the
per-option logit is the dot product of the option query with its attended context vector.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

NEG = -1e9


class AttentionHead(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.hidden, self.rank = hidden, rank
        self.ctx_norm = nn.LayerNorm(hidden)
        self.opt_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, rank, bias=False)
        self.key = nn.Linear(hidden, rank, bias=False)
        self.value = nn.Linear(hidden, rank, bias=False)

    def __call__(self, ctx, ctx_mask, opt, opt_mask):
        """ctx (B, Lc, H), ctx_mask (B, Lc), opt (B, N, H), opt_mask (B, N) -> logits (B, N)."""
        ctx = self.ctx_norm(ctx.astype(mx.float32))
        opt = self.opt_norm(opt.astype(mx.float32))
        q = self.query(opt)                       # (B, N, r)
        k = self.key(ctx)                         # (B, Lc, r)
        v = self.value(ctx)
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.rank)   # (B, N, Lc)
        scores = mx.where(ctx_mask[:, None, :] > 0, scores, NEG)
        attn = mx.softmax(scores, axis=-1)
        attended = attn @ v                       # (B, N, r)
        logits = (q * attended).sum(-1) / math.sqrt(self.rank)       # (B, N)
        return mx.where(opt_mask > 0, logits, NEG)

    # ------------------------------------------------------------ persistence
    def save(self, path: str, config: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.save_weights(str(p))
        p.with_suffix(".json").write_text(json.dumps({"hidden": self.hidden, "rank": self.rank, **config}, indent=2))

    @classmethod
    def load(cls, path: str) -> tuple["AttentionHead", dict]:
        cfg = json.loads(Path(path).with_suffix(".json").read_text())
        head = cls(cfg["hidden"], cfg["rank"])
        head.load_weights(path)
        return head, cfg
