"""jevlike's AttentionHead, ported to PyTorch: one cross-attention layer that scores each option
against the context tokens. Options are queries, context tokens are keys/values, and the
per-option logit is the dot product of the option query with its attended context vector.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

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

    def forward(self, ctx, ctx_mask, opt, opt_mask):
        """ctx (B, Lc, H), ctx_mask (B, Lc), opt (B, N, H), opt_mask (B, N) -> logits (B, N)."""
        ctx = self.ctx_norm(ctx.float())
        opt = self.opt_norm(opt.float())
        q = self.query(opt)                        # (B, N, r)
        k = self.key(ctx)                           # (B, Lc, r)
        v = self.value(ctx)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.rank)  # (B, N, Lc)
        scores = torch.where(ctx_mask[:, None, :] > 0, scores, torch.tensor(NEG, device=ctx.device))
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)             # (B, N, r)
        logits = (q * attended).sum(-1) / math.sqrt(self.rank)            # (B, N)
        return torch.where(opt_mask > 0, logits, torch.tensor(NEG, device=ctx.device))

    # ------------------------------------------------------------ persistence
    def save(self, path: str, config: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(p))
        p.with_suffix(".json").write_text(json.dumps({"hidden": self.hidden, "rank": self.rank, **config}, indent=2))

    @classmethod
    def load(cls, path: str) -> tuple["AttentionHead", dict]:
        cfg = json.loads(Path(path).with_suffix(".json").read_text())
        head = cls(cfg["hidden"], cfg["rank"])
        head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return head, cfg
