"""Serve trained Route-A heads: single-request features + head softmax.

Backend-agnostic wrapper used by the HTTP server (and anything else that wants
head answers online). One HeadScorer per question type; the checkpoint config
selects hidden size / rank, the live scorer provides the frozen encoder.
"""
from __future__ import annotations

import numpy as np


def _pad_single(ctx_h: np.ndarray, opt_h: np.ndarray):
    """Pad one (Lc, H) context + (N, H) options to B=1 batch + masks (FeatureSet.batch format)."""
    H = ctx_h.shape[1]
    Lc, N = ctx_h.shape[0], opt_h.shape[0]
    ctx = np.zeros((1, Lc, H), np.float16)
    cm = np.zeros((1, Lc), np.float32)
    opt = np.zeros((1, N, H), np.float16)
    om = np.zeros((1, N), np.float32)
    ctx[0, :Lc] = ctx_h
    cm[0, :Lc] = 1
    opt[0, :N] = opt_h
    om[0, :N] = 1
    return ctx, cm, opt, om


class HeadScorer:
    """One loaded AttentionHead over a frozen encoder. answer() -> probabilities."""

    def __init__(self, scorer, backend: str, checkpoint: str) -> None:
        self.backend = backend
        if backend == "torch":
            import torch

            from .features_torch import FeatureExtractor
            from .head_torch import AttentionHead

            self._torch = torch
            self.fx = FeatureExtractor(scorer, scorer.model, scorer.tokenizer)
            self.head, self.cfg = AttentionHead.load(checkpoint)
            self.head.eval()
        elif backend == "mlx":
            import mlx.core as mx

            from .features import FeatureExtractor
            from .head import AttentionHead

            self._mx = mx
            self.fx = FeatureExtractor(scorer)
            self.head, self.cfg = AttentionHead.load(checkpoint)
        else:
            raise ValueError(f"backend must be 'torch' or 'mlx', got {backend!r}")

    def answer(self, context: str, options: list[str]) -> list[float]:
        """Head softmax over options. Raises ValueError on bad input."""
        if len(options) < 2:
            raise ValueError("need at least two options")
        ctx_h, opt_h = self.fx.extract(context, options)
        ctx, cm, opt, om = _pad_single(np.asarray(ctx_h), np.asarray(opt_h))
        if self.backend == "torch":
            t = self._torch
            with t.no_grad():
                logits = self.head(t.from_numpy(ctx).float(), t.from_numpy(cm).float(),
                                   t.from_numpy(opt).float(), t.from_numpy(om).float())
                probs = t.softmax(logits, dim=-1)[0]
            return [float(p) for p in probs.tolist()]
        mx = self._mx
        logits = self.head(mx.array(ctx), mx.array(cm), mx.array(opt), mx.array(om))
        probs = mx.softmax(logits, axis=-1)[0]
        mx.eval(probs)
        return [float(p) for p in probs.tolist()]
