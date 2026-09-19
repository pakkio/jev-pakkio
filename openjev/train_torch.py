"""Train and evaluate the head on cached Gemma features (jevlike's train.py / eval.py, in PyTorch)."""
from __future__ import annotations

import json
import random
import time

import torch
import torch.nn as nn
import numpy as np

from .features_torch import FeatureSet
from .head_torch import AttentionHead


def loss_fn(head, ctx, cm, opt, om, labels):
    logits = head(ctx, cm, opt, om)
    return nn.functional.cross_entropy(logits, labels, reduction="mean")


@torch.no_grad()
def evaluate(head: AttentionHead, fs: FeatureSet, batch_size: int = 64, shuffle_context: bool = False) -> dict:
    """top1, top3, 10-bin ECE (on the max probability), n."""
    idx_all = list(range(len(fs)))
    top1 = top3 = 0
    conf_bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, sum(conf), sum(correct)
    for s in range(0, len(fs), batch_size):
        idx = idx_all[s : s + batch_size]
        ctx, cm, opt, om, labels = fs.batch(idx, shuffle_context=shuffle_context)
        # Convert numpy arrays to torch tensors
        ctx_t = torch.from_numpy(ctx).float() if not isinstance(ctx, torch.Tensor) else ctx.float()
        cm_t = torch.from_numpy(cm).float() if not isinstance(cm, torch.Tensor) else cm.float()
        opt_t = torch.from_numpy(opt).float() if not isinstance(opt, torch.Tensor) else opt.float()
        om_t = torch.from_numpy(om).float() if not isinstance(om, torch.Tensor) else om.float()
        labels_t = torch.from_numpy(labels).long() if not isinstance(labels, torch.Tensor) else labels.long()
        
        logits = head(ctx_t, cm_t, opt_t, om_t)
        probs = torch.softmax(logits, dim=-1)
        order = torch.argsort(-probs, dim=-1)
        pmax = probs.max(dim=-1).values
        order_list, pmax_list, labels_list = order.tolist(), pmax.tolist(), labels_t.tolist()
        for o, p, y in zip(order_list, pmax_list, labels_list):
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va = FeatureSet(train_path), FeatureSet(val_path)
    head = AttentionHead(tr.meta["hidden"], rank)
    n_params = sum(v.numel() for v in head.parameters())
    print(f"train {len(tr)} / val {len(va)} examples, hidden {tr.meta['hidden']}, rank {rank}, {n_params / 1e6:.2f}M head params", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, history = -1.0, None, []
    idx_all = list(range(len(tr)))
    
    for epoch in range(1, epochs + 1):
        t = time.perf_counter()
        head.train()
        random.shuffle(idx_all)
        total = 0.0
        for s in range(0, len(tr), batch_size):
            batch = tr.batch(idx_all[s : s + batch_size])
            ctx, cm, opt, om, labels = batch
            # Convert numpy arrays to torch tensors
            ctx_t = torch.from_numpy(ctx).float()
            cm_t = torch.from_numpy(cm).float()
            opt_t = torch.from_numpy(opt).float()
            om_t = torch.from_numpy(om).float()
            labels_t = torch.from_numpy(labels).long()
            
            loss = loss_fn(head, ctx_t, cm_t, opt_t, om_t, labels_t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            
            total += loss.item() * len(labels_t)
        
        val = evaluate(head, va, batch_size)
        rec = {"epoch": epoch, "train_loss": total / len(tr), **{f"val_{k}": v for k, v in val.items() if k != "examples"},
               "seconds": round(time.perf_counter() - t, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["top1"] > best:
            best = val["top1"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    
    head.load_state_dict(best_state)
    head.save(out, {"train": train_path, "validation": val_path, "epochs": epochs, "lr": lr,
                    "batch_size": batch_size, "seed": seed, "best_val_top1": best,
                    "features_meta": tr.meta, "history": history})
    return {"best_val_top1": best, "checkpoint": out}
