"""Train and evaluate the head on cached Gemma features (jevlike's train.py / eval.py, in MLX)."""
from __future__ import annotations

import json
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .features import FeatureSet
from .head import AttentionHead


def loss_fn(head, ctx, cm, opt, om, labels):
    logits = head(ctx, cm, opt, om)
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def evaluate(head: AttentionHead, fs: FeatureSet, batch_size: int = 64, shuffle_context: bool = False) -> dict:
    """top1, top3, 10-bin ECE (on the max probability), n."""
    idx_all = list(range(len(fs)))
    top1 = top3 = 0
    conf_bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, sum(conf), sum(correct)
    for s in range(0, len(fs), batch_size):
        idx = idx_all[s : s + batch_size]
        ctx, cm, opt, om, labels = fs.batch(idx, shuffle_context=shuffle_context)
        probs = mx.softmax(head(ctx, cm, opt, om), axis=-1)
        order = mx.argsort(-probs, axis=-1)
        pmax = probs.max(axis=-1)
        mx.eval(order, pmax)
        order, pmax, labels = order.tolist(), pmax.tolist(), labels.tolist()
        for o, p, y in zip(order, pmax, labels):
            hit = o[0] == y
            top1 += hit
            top3 += y in o[:3]
            b = min(9, int(p * 10))
            conf_bins[b][0] += 1
            conf_bins[b][1] += p
            conf_bins[b][2] += hit
    n = len(fs)
    ece = sum(c * abs(sc / c - sh / c) for c, sc, sh in conf_bins if c) / n
    return {"top1": top1 / n, "top3": top3 / n, "ece": ece, "examples": n}


def train(train_path: str, val_path: str, out: str, rank: int = 256, epochs: int = 8, batch_size: int = 64,
          lr: float = 5e-4, weight_decay: float = 1e-4, seed: int = 7) -> dict:
    random.seed(seed)
    mx.random.seed(seed)
    tr, va = FeatureSet(train_path), FeatureSet(val_path)
    head = AttentionHead(tr.meta["hidden"], rank)
    n_params = sum(v.size for _, v in tree_flatten(head.parameters()))
    print(f"train {len(tr)} / val {len(va)} examples, hidden {tr.meta['hidden']}, rank {rank}, {n_params / 1e6:.2f}M head params", flush=True)
    opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    step = nn.value_and_grad(head, loss_fn)
    best, best_state, history = -1.0, None, []
    idx_all = list(range(len(tr)))
    for epoch in range(1, epochs + 1):
        t = time.perf_counter()
        random.shuffle(idx_all)
        total = 0.0
        for s in range(0, len(tr), batch_size):
            batch = tr.batch(idx_all[s : s + batch_size])
            loss, grads = step(head, *batch)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state, loss)
            total += loss.item() * len(batch[-1])
        val = evaluate(head, va, batch_size)
        rec = {"epoch": epoch, "train_loss": total / len(tr), **{f"val_{k}": v for k, v in val.items() if k != "examples"},
               "seconds": round(time.perf_counter() - t, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["top1"] > best:
            best = val["top1"]
            best_state = [(k, mx.array(v)) for k, v in tree_flatten(head.parameters())]
    head.load_weights(best_state)
    head.save(out, {"train": train_path, "validation": val_path, "epochs": epochs, "lr": lr,
                    "batch_size": batch_size, "seed": seed, "best_val_top1": best,
                    "features_meta": tr.meta, "history": history})
    return {"best_val_top1": best, "checkpoint": out}
