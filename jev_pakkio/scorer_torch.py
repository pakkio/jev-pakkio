"""Zero-shot option scoring with Gemma 3 via PyTorch/transformers.

Port of the MLX scorer. Context is prefilled once; its KV cache is expanded
across the batch so every option is scored in one padded forward pass.
Score = log-probability of option tokens given the context.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DynamicCache)


DEFAULT_MODEL = "google/gemma-4-E2B-it"
NORMS = ("mean", "sum", "pmi")
QUANTIZATIONS = ("none", "8bit", "4bit")

# Pick a model that fits the card it is standing on. A 4-bit Gemma 4 E2B needs
# ~3.2GB, so an 8GB card takes it; below that Qwen 2B (4-bit, ~1.5GB) is the
# smallest thing that still reads natural text well. Override with --model.
VRAM_TIERS = (
    (6 * 1024**3, "google/gemma-4-E2B-it"),
    (3 * 1024**3, "Qwen/Qwen2.5-1.5B-Instruct"),
)
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def auto_model(vram_bytes: int | None = None) -> str:
    """The largest model in VRAM_TIERS that fits, else the smallest listed."""
    if vram_bytes is None:
        if not torch.cuda.is_available():
            return SMALL_MODEL
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
    for needed, repo in VRAM_TIERS:
        if vram_bytes >= needed:
            return repo
    return SMALL_MODEL

# Gemma 4 E-series keeps a per-layer embedding table (~2.8B params in E4B) that
# bitsandbytes cannot quantise. Left on the GPU it does not fit a small card, so
# it stays on the CPU and only the looked-up rows are moved -- same trick the LoRA
# path uses (lora_torch.load_model). Without this, `score --model gemma-4-*`
# tries to allocate the whole table and OOMs.
CPU_MODULES = ("audio_tower", "vision_tower", "embed_vision", "embed_audio")


def is_gemma4_per_layer(cfg) -> bool:
    """True for Gemma 4 E-series configs, which have the per-layer embedding table."""
    text = getattr(cfg, "text_config", cfg)
    return bool(getattr(text, "hidden_size_per_layer_input", 0))


def gemma4_device_map(cfg, quantize: str | None, device: str, on_gpu: bool) -> str | dict:
    """Placement for a Gemma 4 E-series model: table on CPU, layers on GPU 0."""
    lm = "model.language_model"
    device_map = {f"{lm}.{k}": 0 for k in ("embed_tokens", "layers", "norm", "rotary_emb",
                                           "per_layer_model_projection", "per_layer_projection_norm")}
    device_map["lm_head"] = 0
    device_map[f"{lm}.embed_tokens_per_layer"] = "cpu"
    device_map.update({f"model.{m}": "cpu" for m in CPU_MODULES})
    return device_map


def cpu_embedding_lookup(model) -> None:
    """Run the per-layer embedding lookup on the CPU and move only the rows.

    accelerate's offload hook would stream the whole table to the GPU on every
    call; patching forward keeps the lookup on the CPU.
    """
    from accelerate.hooks import remove_hook_from_module

    table = model.model.language_model.embed_tokens_per_layer
    remove_hook_from_module(table)
    table.to("cpu")
    fwd = table.forward
    table.forward = lambda ids, _f=fwd: _f(ids.to("cpu")).to("cuda")


def quantization_config(quantize: str | None) -> BitsAndBytesConfig | None:
    """bitsandbytes config for `quantize`, or None for full bf16 weights.

    4bit is NF4 with double quantization and a bfloat16 compute dtype: roughly
    a quarter of the bf16 footprint, which is what lets a 4B model fit on a 4GB
    card. Scoring reads logits only, so the quantization error costs a little
    calibration, not the ranking.
    """
    if not quantize or quantize == "none":
        return None
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"quantize must be one of {QUANTIZATIONS}")


def gemma4_cpu_module_map(cfg) -> dict:
    """Placement for a Gemma 4 E-series model that fits a small card.

    Every submodule is named explicitly. A catch-all entry makes accelerate infer
    placement for the weights it does not recognise -- including the 2.35B-param
    per-layer embedding table it then tries to move to the GPU. Listing the
    language model's parts individually is what keeps that table on the CPU.

    This mirrors lora_torch.load_model, which is the path already known to load
    these models on an 8GB card.
    """
    lm = "model.language_model"
    device_map = {f"{lm}.{k}": 0 for k in ("embed_tokens", "layers", "norm", "rotary_emb",
                                           "per_layer_model_projection", "per_layer_projection_norm")}
    device_map["lm_head"] = 0
    device_map[f"{lm}.embed_tokens_per_layer"] = "cpu"
    device_map.update({f"model.{m}": "cpu" for m in CPU_MODULES})
    return device_map


def patch_cpu_embedding_lookup(model) -> None:
    """Replace accelerate's offload hook with a CPU lookup that moves only rows.

    accelerate's hook would stream the whole table to the GPU on every call.
    `table.to("cpu")` is what actually materialises the weights: removing the
    hook alone leaves them on the meta device and the forward pass then fails
    with "Cannot copy out of meta tensor". Same sequence as lora_torch.
    """
    from accelerate.hooks import remove_hook_from_module

    table = model.model.language_model.embed_tokens_per_layer
    remove_hook_from_module(table)
    table.to("cpu")
    fwd = table.forward
    table.forward = lambda ids, _f=fwd: _f(ids.to("cpu")).to("cuda")


def load_gemma4_cpu_embeddings(model_path: str, quantize: str | None, device: str):
    """Load a Gemma 4 E-series model with its embedding tables on the CPU.

    bitsandbytes refuses to mix a 4-bit quantiser with any CPU-dispatched module,
    so for these models the body is loaded in bf16 instead of 4-bit: ~3.9GB on the
    GPU for E2B, against ~4.7GB of tables kept off it. Falls back to the ordinary
    single-device load when there is no CUDA or the model is not a PLE one.
    """
    cfg = AutoConfig.from_pretrained(model_path)
    if not torch.cuda.is_available() or not is_gemma4_per_layer(cfg):
        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map=_device_map(quantize, device),
            quantization_config=quantization_config(quantize),
        )
    device_map = gemma4_cpu_module_map(cfg)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 device_map=device_map)
    patch_cpu_embedding_lookup(model)
    model.config.use_cache = False
    return model


@dataclass(frozen=True)
class OptionScore:
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float

    def to_dict(self) -> dict:
        return asdict(self)


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def _cache_tensors(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot a Cache as plain (keys, values) tensors, one pair per layer."""
    return [(layer.keys, layer.values) for layer in past_key_values.layers]


def _repeat_kv(kv: list[tuple[torch.Tensor, torch.Tensor]], n: int) -> DynamicCache:
    """Build a fresh Cache with the prefix KV replicated n times batch-wise.

    A new Cache per chunk is required: the model appends the option tokens to
    whatever cache it is handed, so reusing one would corrupt the prefix.
    """
    return DynamicCache(
        ddp_cache_data=[
            (k.repeat_interleave(n, dim=0), v.repeat_interleave(n, dim=0))
            for k, v in kv
        ]
    )


def _device_map(quantize: str | None, device: str = "auto") -> str | dict:
    """Placement for from_pretrained.

    An explicit --device wins. Otherwise: bitsandbytes cannot run a quantised
    module that accelerate parked on the CPU, and "auto" keeps a safety margin
    that makes it offload on a small card -- on a 4GB GPU a 4-bit Gemma 3 4B
    needs ~3.1GB and fits, but only if it is pinned. Pin the whole model to GPU
    0 when quantising; fall back to "auto" without CUDA or without
    quantisation.
    """
    if device and device != "auto":
        return device if ":" in device or device == "cpu" else {"": device}
    if quantize and quantize != "none" and torch.cuda.is_available():
        return {"": 0}
    return "auto"


def _resolve_device(device: str = "auto") -> str:
    """The device the weights actually ended up on, for /health and errors."""
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class OptionScorer:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        batch_size: int = 8,
        chat: bool = False,
        sep: str = "",
        quantize: str | None = None,
        adapter_path: str | None = None,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.quantize = quantize or "none"
        self.backend = "torch"
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = load_gemma4_cpu_embeddings(model_path, quantize, device)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.batch_size = max(1, batch_size)
        self.chat = chat
        self.sep = sep
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else 0
        )
        self.bos_id = self.tokenizer.bos_token_id
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------ text
    def context_ids(self, context: str, chat: bool | None = None, sep: str | None = None) -> list[int]:
        chat = self.chat if chat is None else chat
        sep = self.sep if sep is None else sep
        if chat:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.bos_id is not None and ids[:1] != [self.bos_id]:
                ids = [self.bos_id] + ids
            return ids
        ids = self.tokenizer.encode(context + sep, add_special_tokens=True)
        return ids

    def option_ids(self, option: str) -> list[int]:
        ids = self.tokenizer.encode(option, add_special_tokens=False)
        if not ids:
            raise ValueError(f"option tokenises to nothing: {option!r}")
        return ids

    def _uncond_prefix_ids(self) -> list[int]:
        """A one-token prefix for the unconditional (PMI) pass.

        Gemma always has a BOS; models like Qwen do not, and the forward pass
        rejects an empty sequence -- fall back to EOS, then to a newline.
        """
        for tok in (self.bos_id, self.tokenizer.eos_token_id):
            if tok is not None:
                return [tok]
        ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if not ids:
            raise ValueError("model has no usable prefix token for PMI scoring")
        return ids[:1]

    # --------------------------------------------------------------- prefill
    def _prefill(self, ids: list[int]):
        """Prefill the context, return (kv tensors, last context logits)."""
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        return _cache_tensors(outputs.past_key_values), outputs.logits[0, -1].float()

    def _score_with_prefix(
        self,
        kv: list[tuple[torch.Tensor, torch.Tensor]],
        last_logits: torch.Tensor,
        opts: list[list[int]],
    ) -> list[float]:
        """Sum of log p(option tokens | prefix) for each option, batched."""
        sums: list[float] = []
        for start in range(0, len(opts), self.batch_size):
            chunk = opts[start : start + self.batch_size]
            n = len(chunk)
            L = max(len(x) for x in chunk)

            # Build padded tensor
            arr = torch.full((n, L), self.pad_id, dtype=torch.long, device=self.device)
            mask = torch.zeros((n, L), dtype=torch.float32, device=self.device)
            for i, x in enumerate(chunk):
                arr[i, : len(x)] = torch.tensor(x, dtype=torch.long)
                mask[i, : len(x)] = 1.0

            # Expand KV cache for this batch
            expanded_cache = _repeat_kv(kv, n)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=arr,
                    past_key_values=expanded_cache,
                    use_cache=False,
                )
            logits = outputs.logits.float()  # (n, L, V)

            # Concatenate the last context logits as the "first" prediction
            first = last_logits.unsqueeze(0).unsqueeze(0).expand(n, 1, -1)
            pred = torch.cat([first, logits[:, :-1]], dim=1)  # (n, L, V)

            # Log-probs
            log_probs = F.log_softmax(pred, dim=-1)  # (n, L, V)

            # Gather log-prob of the target tokens
            target = arr.unsqueeze(-1)  # (n, L, 1)
            tgt_lp = log_probs.gather(-1, target).squeeze(-1)  # (n, L)

            # Sum over option tokens (masked)
            s = (tgt_lp * mask).sum(-1)
            sums.extend(s.cpu().tolist())
        return sums

    # ---------------------------------------------------------------- public
    def score(
        self,
        context: str,
        options: Sequence[str],
        norm: str = "mean",
        chat: bool | None = None,
        sep: str | None = None,
    ) -> list[OptionScore]:
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if len(options) < 2:
            raise ValueError("need at least two options")

        t0 = time.perf_counter()
        opts = [self.option_ids(o) for o in options]
        ctx_ids = self.context_ids(context, chat=chat, sep=sep)
        cache, last = self._prefill(ctx_ids)
        t1 = time.perf_counter()
        sums = self._score_with_prefix(cache, last, opts)
        t2 = time.perf_counter()

        uncond: list[float] | None = None
        if norm == "pmi":
            base_cache, base_last = self._prefill(self._uncond_prefix_ids())
            uncond = self._score_with_prefix(base_cache, base_last, opts)
        t3 = time.perf_counter()

        self.last_timing = {
            "prefill_s": t1 - t0,
            "options_s": t2 - t1,
            "uncond_s": t3 - t2,
            "total_s": t3 - t0,
            "context_tokens": len(ctx_ids),
            "option_tokens": sum(len(o) for o in opts),
        }

        means = [s / len(o) for s, o in zip(sums, opts)]
        if norm == "sum":
            scores = sums
        elif norm == "mean":
            scores = means
        else:
            scores = [s - u for s, u in zip(sums, uncond)]
        probs = _softmax(scores)
        return [
            OptionScore(
                option=o,
                n_tokens=len(ids),
                logprob_sum=s,
                logprob_mean=m,
                logprob_uncond=(uncond[i] if uncond is not None else None),
                score=sc,
                probability=p,
            )
            for i, (o, ids, s, m, sc, p) in enumerate(
                zip(options, opts, sums, means, scores, probs)
            )
        ]

    def score_naive(self, context: str, options: Sequence[str]) -> list[float]:
        """Reference: re-encode context + option from scratch per option."""
        ctx = self.context_ids(context)
        out = []
        for o in options:
            oid = self.option_ids(o)
            ids = ctx + oid
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
            logits = outputs.logits[0].float()
            pred = logits[len(ctx) - 1 : len(ctx) - 1 + len(oid)]
            lp = F.log_softmax(pred, dim=-1)
            target = torch.tensor(oid, dtype=torch.long, device=self.device).unsqueeze(-1)
            s = lp.gather(-1, target).sum()
            out.append(s.cpu().item())
        return out


def iter_jsonl(path: str) -> Iterable[dict]:
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    scorer = OptionScorer("google/gemma-3-4b-it")
    res = scorer.score(
        "The capital of France is",
        [" Paris", " Berlin", " Madrid"],
    )
    print("=" * 60)
    for r in res:
        print(f"{r.option!r:12}  score={r.score:+.4f}  prob={r.probability:.4f}  logprob_sum={r.logprob_sum:+.4f}")
    print("=" * 60)
    print(f"best: {res[[r.score for r in res].index(max(r.score for r in res))].option!r}")
    print(f"timing: {scorer.last_timing}")
