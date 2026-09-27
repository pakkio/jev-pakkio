"""Serve LoRA adapters: one 4-bit base model, any number of named adapters, switched per request.

Used by the HTTP server in LoRA mode (``openjev serve --backend torch --lora NAME=PATH``).
An adapter dir may hold the float32 ``adapter_model.safetensors`` or the int8 file written
by ``openjev lora-quantize``.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .lora_torch import INT8_FILE, chat_ids, load_int8, load_model, option_scores


class LoraEngine:
    def __init__(self, model_path: str, adapters: dict[str, str], quantize: str = "4bit") -> None:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        self.model_path = model_path
        self.model, self.tok = load_model(model_path, quantize)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        self.adapters: dict[str, str] = {}
        for name, path in adapters.items():
            cfg = LoraConfig.from_pretrained(path)
            if not self.adapters:
                self.model = get_peft_model(self.model, cfg, adapter_name=name)
            else:
                self.model.add_adapter(name, cfg)
            full = Path(path) / "adapter_model.safetensors"
            weights = load_file(str(full)) if full.exists() else load_int8(path)
            set_peft_model_state_dict(self.model, weights, adapter_name=name)
            self.adapters[name] = path
        self.model.eval()
        self.last_timing: dict[str, float] = {}

    @torch.no_grad()
    def score(self, context: str, options: list[str], adapter: str | None = None, chat: bool = True,
              sep: str = "\nChoice: ") -> tuple[list[float], list[float]]:
        """(sum log-probs, softmax probabilities) over options; adapter None = zero-shot."""
        if adapter is not None and adapter not in self.adapters:
            raise ValueError(f"unknown adapter {adapter!r}; loaded: {sorted(self.adapters)}")
        if len(options) < 2:
            raise ValueError("need at least two options")
        t = time.perf_counter()
        ctx = chat_ids(self.tok, context, options) if chat else self.tok.encode(context + sep, add_special_tokens=True)
        opts = [self.tok.encode(o, add_special_tokens=False) for o in options]
        if any(not o for o in opts):
            raise ValueError("an option tokenises to nothing")
        if adapter is None:
            if self.adapters:
                with self.model.disable_adapter():
                    s = option_scores(self.model, ctx, opts, self.pad_id, self.device)
            else:
                s = option_scores(self.model, ctx, opts, self.pad_id, self.device)
        else:
            self.model.set_adapter(adapter)
            s = option_scores(self.model, ctx, opts, self.pad_id, self.device)
        probs = torch.softmax(s, -1)
        self.last_timing = {"seconds": time.perf_counter() - t, "context_tokens": float(len(ctx)),
                            "option_tokens": float(sum(len(o) for o in opts))}
        return s.tolist(), probs.tolist()
