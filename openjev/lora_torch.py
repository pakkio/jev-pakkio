"""LoRA fine-tuning of the scorer itself (Route C, PyTorch + peft).

Unlike the head (Route A), this adapts Gemma so that its own option
log-likelihoods rank the labelled option first. Each training row scores every
option as sum log p(option | context + sep) -- the zero-shot ``--norm sum``
score -- and the loss is listwise cross-entropy over those scores. So the
tuned model is still a plain option scorer, and the zero-shot baseline is the
same model with the adapter disabled.

Weights load 4-bit (QLoRA). Gemma 4 E-series models carry a per-layer
embedding table of ~2.8B unquantisable params; it is only ever indexed, so it
stays on the CPU and just its looked-up rows move to the GPU.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .scorer_torch import iter_jsonl

TARGETS = r".*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
CPU_MODULES = ("audio_tower", "vision_tower", "embed_vision", "embed_audio")


def _is_gemma4_ple(cfg) -> bool:
    text = getattr(cfg, "text_config", cfg)
    return bool(getattr(text, "hidden_size_per_layer_input", 0))


def load_model(model_path: str, quantize: str = "4bit"):
    """Load (model, tokenizer) for LoRA training, fitting an 8 GB card."""
    tok = AutoTokenizer.from_pretrained(model_path)
    cfg = AutoConfig.from_pretrained(model_path)
    q = None
    if quantize in ("4bit", "8bit"):
        q = BitsAndBytesConfig(
            load_in_4bit=quantize == "4bit",
            load_in_8bit=quantize == "8bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=[*CPU_MODULES, "lm_head"],
        )
    device_map: str | dict = {"": 0} if torch.cuda.is_available() else "cpu"
    ple = _is_gemma4_ple(cfg) and torch.cuda.is_available()
    if ple:
        lm = "model.language_model"
        device_map = {f"{lm}.{k}": 0 for k in ("embed_tokens", "layers", "norm", "rotary_emb",
                                               "per_layer_model_projection", "per_layer_projection_norm")}
        device_map["lm_head"] = 0
        device_map[f"{lm}.embed_tokens_per_layer"] = "cpu"
        device_map.update({f"model.{m}": "cpu" for m in CPU_MODULES})
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device_map,
                                                 quantization_config=q)
    if ple:
        # accelerate would stream the whole 5 GB table to the GPU on every call;
        # run the lookup on the CPU and move only the rows.
        from accelerate.hooks import remove_hook_from_module

        table = model.model.language_model.embed_tokens_per_layer
        remove_hook_from_module(table)
        table.to("cpu")
        fwd = table.forward
        table.forward = lambda ids, _f=fwd: _f(ids.to("cpu")).to("cuda")
    model.config.use_cache = False
    return model, tok


def attach_adapter(model, adapter: str):
    """Wrap `model` with the saved LoRA in `adapter`, in place.

    PeftModel.from_pretrained would re-dispatch the model through accelerate,
    which undoes the CPU placement load_model set up; inject the layers and
    load the weights instead.
    """
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    peft_model = get_peft_model(model, LoraConfig.from_pretrained(adapter))
    full = Path(adapter) / "adapter_model.safetensors"
    weights = load_file(str(full)) if full.exists() else load_int8(adapter)
    set_peft_model_state_dict(peft_model, weights)
    return peft_model


INT8_FILE = "adapter_model.int8.safetensors"


def quantize_int8(adapter: str, out: str) -> dict:
    """Store an adapter as int8 with one float16 scale per row (~4x smaller than float32).

    LoRA matrices are small and well-conditioned; symmetric per-row absmax
    quantisation keeps the error around 0.4% of each row's largest weight.
    """
    import shutil

    from safetensors.torch import load_file, save_file

    src = load_file(str(Path(adapter) / "adapter_model.safetensors"))
    packed = {}
    for name, w in src.items():
        w = w.float()
        scale = (w.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-12)
        packed[name + ".q"] = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        packed[name + ".scale"] = scale.to(torch.float16)
    Path(out).mkdir(parents=True, exist_ok=True)
    save_file(packed, str(Path(out) / INT8_FILE))
    shutil.copy(Path(adapter) / "adapter_config.json", Path(out) / "adapter_config.json")
    before = (Path(adapter) / "adapter_model.safetensors").stat().st_size
    after = (Path(out) / INT8_FILE).stat().st_size
    return {"from": adapter, "to": out, "bytes_before": before, "bytes_after": after, "tensors": len(src)}


def load_int8(adapter: str) -> dict:
    """Dequantise an int8 adapter written by quantize_int8 back to float32 tensors."""
    from safetensors.torch import load_file

    packed = load_file(str(Path(adapter) / INT8_FILE))
    return {n[:-2]: packed[n].float() * packed[n[:-2] + ".scale"].float() for n in packed if n.endswith(".q")}


def chat_ids(tok, context: str, options: list[str]) -> list[int]:
    """The chat format: context plus the option list as a user turn, options scored as the reply.

    Instruction-tuned Gemma is far stronger here than on plain `context + sep`
    (noul zero-shot 0.51 -> 1.00, news 0.23 -> 0.99 in our runs).
    """
    msg = context + "\nOptions: " + "; ".join(options) + "\nAnswer with one option exactly."
    text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


class Rows:
    """Tokenised {context, options, label} rows, in the plain (context + sep) or chat format."""

    def __init__(self, path: str, tok, sep: str, limit: int = 0, chat: bool = False) -> None:
        rows = list(iter_jsonl(path))
        if limit:
            rows = rows[:limit]
        if chat:
            self.ctx = [chat_ids(tok, r["context"], r["options"]) for r in rows]
        else:
            self.ctx = [tok.encode(r["context"] + sep, add_special_tokens=True) for r in rows]
        self.opts = [[tok.encode(o, add_special_tokens=False) for o in r["options"]] for r in rows]
        self.labels = [int(r["label"]) for r in rows]

    def __len__(self) -> int:
        return len(self.labels)


def option_scores(model, ctx: list[int], opts: list[list[int]], pad_id: int, device) -> torch.Tensor:
    """(N,) sum log p(option | ctx), one batched forward over ctx+option rows."""
    lc, lo = len(ctx), max(len(o) for o in opts)
    ids = torch.full((len(opts), lc + lo), pad_id, dtype=torch.long)
    mask = torch.zeros((len(opts), lo))
    tgt = torch.zeros((len(opts), lo), dtype=torch.long)
    for i, o in enumerate(opts):
        ids[i, :lc] = torch.tensor(ctx)
        ids[i, lc : lc + len(o)] = torch.tensor(o)
        tgt[i, : len(o)] = torch.tensor(o)
        mask[i, : len(o)] = 1
    ids, mask, tgt = ids.to(device), mask.to(device), tgt.to(device)
    # Right padding: the last lo + 1 positions hold every option-predicting logit.
    logits = model(input_ids=ids, logits_to_keep=lo + 1).logits[:, :-1].float()  # (N, lo, V)
    logp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


@torch.no_grad()
def evaluate(model, rows: Rows, pad_id: int, device, shuffle_context: bool = False) -> dict:
    """top1, top3, 10-bin ECE on the max softmax probability over option scores."""
    model.eval()
    top1 = top3 = 0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    n = len(rows)
    for i in range(n):
        ctx = rows.ctx[(i + 1) % n] if shuffle_context else rows.ctx[i]
        p = torch.softmax(option_scores(model, ctx, rows.opts[i], pad_id, device), -1)
        order = torch.argsort(-p).tolist()
        y = rows.labels[i]
        hit = order[0] == y
        top1 += hit
        top3 += y in order[:3]
        pm = p.max().item()
        b = bins[min(9, int(pm * 10))]
        b[0] += 1
        b[1] += pm
        b[2] += hit
    ece = sum(abs(sc - sh) for c, sc, sh in bins if c) / n
    return {"top1": top1 / n, "top3": top3 / n, "ece": ece, "examples": n}


def train(model_path: str, train_path: str, val_path: str, out: str, test_path: str | None = None,
          sep: str = "\nChoice: ", quantize: str = "4bit", rank: int = 16, alpha: int = 32,
          epochs: int = 1, lr: float = 2e-4, accum: int = 8, warmup: int = 20, limit: int = 0,
          eval_every: int = 0, seed: int = 7, resume: str | None = None, max_rows: int = 0,
          chat: bool = False) -> dict:
    from peft import LoraConfig, get_peft_model

    random.seed(seed)
    torch.manual_seed(seed)
    t = time.perf_counter()
    model, tok = load_model(model_path, quantize)
    print(f"loaded {model_path} in {time.perf_counter() - t:.1f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB on GPU", file=sys.stderr, flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va = Rows(train_path, tok, sep, limit, chat), Rows(val_path, tok, sep, chat=chat)
    te = Rows(test_path, tok, sep, chat=chat) if test_path else None

    # Recompute activations in backward: without this, 42 layers of dequantised
    # 4-bit weights and activations overflow an 8 GB card.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if resume:  # continue from a saved adapter (fresh optimiser state)
        model = attach_adapter(model, resume)
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                                 target_modules=TARGETS, bias="none"))
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"train {len(tr)} / val {len(va)} rows, LoRA r={rank} alpha={alpha}, "
          f"{n_params / 1e6:.2f}M trainable params", flush=True)

    baseline = {}
    with model.disable_adapter():
        baseline["val"] = evaluate(model, va, pad_id, device)
        if te:
            baseline["test"] = evaluate(model, te, pad_id, device)
    print(json.dumps({"zero_shot": baseline}), flush=True)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    budget = min(max_rows, epochs * len(tr)) if max_rows else epochs * len(tr)
    total_steps = max(1, budget // accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / total_steps))
    eval_every = eval_every or len(tr)
    best, history, seen, t = -1.0, [], 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(tr)))
        random.shuffle(order)
        running = 0.0
        if max_rows:
            order = order[: max(0, budget - seen)]
        for k, i in enumerate(order, 1):
            model.train()
            s = option_scores(model, tr.ctx[i], tr.opts[i], pad_id, device)
            loss = nn.functional.cross_entropy(s.unsqueeze(0), torch.tensor([tr.labels[i]], device=device))
            (loss / accum).backward()
            running += loss.item()
            seen += 1
            if k % accum == 0 or k == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if k % 100 == 0:
                print(f"  epoch {epoch} {k}/{len(tr)} loss {running / 100:.4f} "
                      f"{(time.perf_counter() - t) / seen * 1000:.0f} ms/row "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                running = 0.0
            if seen % eval_every == 0 or k == len(order):
                val = evaluate(model, va, pad_id, device)
                rec = {"epoch": epoch, "rows_seen": seen, **{f"val_{a}": b for a, b in val.items() if a != "examples"},
                       "seconds": round(time.perf_counter() - t, 1)}
                history.append(rec)
                print(json.dumps(rec), flush=True)
                if val["top1"] > best:
                    best = val["top1"]
                    model.save_pretrained(out)
    result = {"model": model_path, "train": train_path, "validation": val_path, "sep": sep, "chat": chat,
              "quantize": quantize, "rank": rank, "alpha": alpha, "epochs": epochs, "lr": lr,
              "accum": accum, "seed": seed, "best_val_top1": best, "zero_shot": baseline, "history": history}
    if te:
        model = attach_adapter(model.unload(), out)  # reload the best checkpoint
        result["test"] = evaluate(model, te, pad_id, device)
        result["test_shuffled_context"] = evaluate(model, te, pad_id, device, shuffle_context=True)
        print(json.dumps({"test": result["test"], "test_shuffled_context": result["test_shuffled_context"]}), flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "openjev_lora.json").write_text(json.dumps(result, indent=2))
    return result


def eval_adapter(model_path: str, adapter: str, data: str, sep: str = "\nChoice: ", quantize: str = "4bit",
                 zero_shot: bool = True, chat: bool = False) -> dict:
    """Test a saved adapter: tuned, shuffled-context control, and (optionally) the same model zero-shot."""
    model, tok = load_model(model_path, quantize)
    model = attach_adapter(model, adapter)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = Rows(data, tok, sep, chat=chat)
    out = {"lora": evaluate(model, rows, pad_id, device),
           "lora_shuffled_context": evaluate(model, rows, pad_id, device, shuffle_context=True)}
    if zero_shot:
        with model.disable_adapter():
            out["zero_shot"] = evaluate(model, rows, pad_id, device)
    return out
