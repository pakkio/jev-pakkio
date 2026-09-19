"""Frozen-Gemma feature extraction for the trainable head (Route A, PyTorch backend).

Same contract as openjev/features.py but uses HuggingFace transformers + PyTorch
instead of MLX. Loads the model with ``device_map="auto"`` in ``bfloat16``,
runs inference under ``torch.no_grad()``, and saves the extracted hidden states
as numpy arrays in ``.npz`` files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .scorer_torch import OptionScorer, _cache_tensors, _device_map, _repeat_kv, iter_jsonl


class FeatureExtractor:
    """Extract hidden states from a frozen Gemma model for head training.

    Uses HuggingFace transformers + PyTorch instead of MLX.  The model is loaded
    with ``device_map="auto"`` and ``torch_dtype=torch.bfloat16``; inference
    runs under ``torch.no_grad()``.
    """

    def __init__(
        self,
        scorer: OptionScorer,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        contextual: bool = False,
        layer: int = -1,
    ) -> None:
        self.s = scorer
        self.contextual = contextual
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer  # -1 = last layer output (after final norm)

        # Navigate to the text transformer (without LM head).
        # Text-only:  AutoModelForCausalLM -> .model
        # Multimodal: -> .model (Gemma3Model) -> .language_model (Gemma3TextModel).
        # The multimodal wrapper's config has no hidden_size -- the text tower's
        # does -- and the vision branch is dead weight for option scoring.
        core = model.model if hasattr(model, "model") else model
        self.core = getattr(core, "language_model", core)
        cfg = self.core.config
        cfg = getattr(cfg, "text_config", cfg)
        self.hidden = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers

        # Device for input_ids (embedding layer device)
        try:
            self.device = self.core.get_input_embeddings().weight.device
        except AttributeError:
            self.device = next(self.core.parameters()).device

    def _forward(
        self,
        input_ids: torch.Tensor,
        cache=None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Run the transformer and return (hidden_states, past_key_values).

        Args:
            input_ids: (B, L) token ids
            cache: past_key_values from a previous pass (for prefix sharing)

        Returns:
            hidden_states: (B, L, H) from the requested layer
            past_key_values: updated cache
        """
        kwargs: dict = {}
        if cache is not None:
            kwargs["past_key_values"] = cache

        # Only request hidden_states if we need a specific intermediate layer
        if self.layer >= 0:
            kwargs["output_hidden_states"] = True

        with torch.no_grad():
            outputs = self.core(input_ids=input_ids, **kwargs)

        if self.layer < 0:
            h = outputs.last_hidden_state  # after all layers + final norm
        else:
            # hidden_states[0] = embedding output, hidden_states[k+1] = after layer k
            h = outputs.hidden_states[self.layer + 1]

        return h, outputs.past_key_values

    @staticmethod
    def _expand_cache(kv, n: int):
        """Build a fresh batch-expanded Cache from snapshotted prefix KV tensors.

        A new Cache per chunk is required: the model appends to whatever cache
        it is handed, so reusing one across chunks would corrupt the prefix.
        """
        if kv is None:
            return None
        return _repeat_kv(kv, n)

    def extract(
        self,
        context: str,
        options: list[str],
        chat: bool | None = None,
        sep: str | None = None,
    ):
        """Return (context_h (Lc, H) float16, options_h (N, H) float16).

        Context hidden states are all token vectors from the last layer.
        Options are masked-mean-pooled to one vector each.
        """
        ctx_ids = self.s.context_ids(context, chat=chat, sep=sep)
        opts = [self.s.option_ids(o) for o in options]

        ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
        ctx_h, cache = self._forward(ctx_tensor)
        ctx_h = ctx_h[0].cpu()  # (Lc, H)
        kv = _cache_tensors(cache)

        if not self.contextual:
            # Options stand alone: prefix is just BOS
            bos_id = getattr(self.s, "bos_id", None)
            if bos_id is None:
                bos_id = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id or 0
            bos_tensor = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)
            _, cache = self._forward(bos_tensor)
            kv = _cache_tensors(cache)

        pooled = []
        for start in range(0, len(opts), self.s.batch_size):
            chunk = opts[start : start + self.s.batch_size]
            n = len(chunk)
            L = max(len(x) for x in chunk)

            # Pad sequences to max length
            padded = [x + [self.s.pad_id] * (L - len(x)) for x in chunk]
            opt_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)

            # Create mask (n, L) - 1 for real tokens, 0 for padding
            mask = torch.tensor(
                [[1.0] * len(x) + [0.0] * (L - len(x)) for x in chunk],
                dtype=torch.float32,
                device=self.device,
            )

            # Expand cache for batch
            expanded_cache = self._expand_cache(kv, n)

            h, _ = self._forward(opt_tensor, cache=expanded_cache)  # (n, L, H)

            # Masked mean pooling over option tokens
            mean = (h.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)  # (n, H)
            pooled.append(mean.cpu().to(torch.float16).numpy())

        return ctx_h.cpu().to(torch.float16).numpy(), np.concatenate(pooled, 0)


def extract_dataset(
    scorer: OptionScorer,
    data: str,
    out: str,
    limit: int = 0,
    chat: bool = False,
    sep: str = "",
    contextual: bool = False,
    model_path: str | None = None,
    layer: int = -1,
    device_map: str = "auto",
) -> dict:
    """Cache features for a jevlike JSONL {context, options, label} file into one .npz.

    Uses PyTorch/transformers to extract hidden states from a frozen Gemma model.
    Compatible with the same CLI entrypoint as the MLX version.
    """
    t_load = time.perf_counter()

    if model_path is None:
        # The scorer already holds the frozen model on device -- reuse it rather
        # than loading a second copy, which would double the VRAM footprint.
        model_path = scorer.model_path
        tokenizer = scorer.tokenizer
        model = scorer.model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        model.eval()

    load_time = time.perf_counter() - t_load
    print(f"loaded {model_path} in {load_time:.1f}s", file=sys.stderr)

    fx = FeatureExtractor(scorer, model, tokenizer, contextual=contextual, layer=layer)

    rows = list(iter_jsonl(data))
    if limit:
        rows = rows[:limit]

    arrays: dict[str, np.ndarray] = {}
    labels = []
    t = time.perf_counter()

    for i, row in enumerate(rows):
        c, o = fx.extract(row["context"], row["options"], chat=chat, sep=sep)
        arrays[f"ctx_{i}"] = c
        arrays[f"opt_{i}"] = o
        labels.append(int(row.get("label", -1)))
        if (i + 1) % 100 == 0:
            print(
                f"  {i + 1}/{len(rows)} examples, "
                f"{(time.perf_counter() - t) / (i + 1) * 1000:.0f} ms each",
                flush=True,
            )

    arrays["labels"] = np.array(labels, dtype=np.int32)
    meta = {
        "n": len(rows),
        "hidden": fx.hidden,
        "source": data,
        "chat": chat,
        "sep": sep,
        "contextual": contextual,
        "layer": layer,
        "backend": "torch",
        "seconds": round(time.perf_counter() - t, 1),
    }
    arrays["meta"] = np.array(json.dumps(meta))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return meta


class FeatureSet:
    """In-memory view of a cached .npz: lists of (Lc, H) contexts, (N, H) options, labels.

    Returns numpy arrays (compatible with both MLX and PyTorch training).
    This is the torch-compatible version that returns numpy arrays instead of
    MLX arrays, so ``train_torch.py`` can convert them to ``torch.Tensor``.
    """

    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.labels = z["labels"]
        self.ctx = [z[f"ctx_{i}"] for i in range(len(self.labels))]
        self.opt = [z[f"opt_{i}"] for i in range(len(self.labels))]

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: list[int], shuffle_context: bool = False):
        """Pad a batch: returns ctx (B, Lc, H), ctx_mask (B, Lc), opt (B, N, H), opt_mask (B, N), labels (B,).

        All arrays are numpy (float16/float32/int32) - no MLX or torch dependencies.
        Compatible with ``train_torch.py`` which converts them to ``torch.Tensor``.
        """
        ctx_src = [self.ctx[i] for i in idx]
        if shuffle_context:  # jevlike's control: every example sees another example's context
            ctx_src = ctx_src[1:] + ctx_src[:1]
        H = self.meta["hidden"]
        Lc = max(c.shape[0] for c in ctx_src)
        N = max(self.opt[i].shape[0] for i in idx)
        B = len(idx)
        ctx = np.zeros((B, Lc, H), np.float16)
        cm = np.zeros((B, Lc), np.float32)
        opt = np.zeros((B, N, H), np.float16)
        om = np.zeros((B, N), np.float32)
        for b, (c, i) in enumerate(zip(ctx_src, idx)):
            ctx[b, : c.shape[0]] = c
            cm[b, : c.shape[0]] = 1
            o = self.opt[i]
            opt[b, : o.shape[0]] = o
            om[b, : o.shape[0]] = 1
        return ctx, cm, opt, om, self.labels[idx]
