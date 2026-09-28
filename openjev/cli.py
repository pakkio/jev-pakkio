"""Command line entry points: score, eval, bench, check, serve, features, train, eval-head."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time

# NOTE: backend modules are imported lazily in _get_backend_modules so that
# `--help` and `--backend torch` work on machines without MLX (e.g. Linux).
DEFAULT_MODEL = "models/gemma-3-4b-it"
NORMS = ("mean", "sum", "pmi")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps (torch backend)")
    p.add_argument("--model", default=None, help="local dir or HF repo id")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir to load on top of --model (e.g. adapters/chess-lora)")
    p.add_argument("--batch-size", type=int, default=8, help="options per forward pass")
    p.add_argument("--norm", choices=NORMS, default="mean",
                   help="mean: per-token log-prob; sum: total; pmi: sum minus unconditional")
    p.add_argument("--chat", action="store_true",
                   help="wrap context in Gemma's chat template; options score as the reply")
    p.add_argument("--sep", default="", help="string inserted between context and option")
    p.add_argument("--backend", choices=["mlx", "torch"], default="mlx",
                   help="scoring backend: mlx (default, Apple silicon) or torch (PyTorch)")
    p.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none",
                   help="torch backend only: load the weights quantised via bitsandbytes")


def _get_backend_modules(backend: str):
    """Load backend modules based on the backend name.

    Returns (scorer, features, train, head) modules. The torch modules are
    imported lazily so the MLX path has no torch dependency.
    """
    if backend == "torch":
        from . import features_torch
        from . import head_torch
        from . import scorer_torch
        from . import train_torch
        return scorer_torch, features_torch, train_torch, head_torch
    from . import features
    from . import head
    from . import scorer
    from . import train
    return scorer, features, train, head


def _scorer(args: argparse.Namespace):
    t = time.perf_counter()
    scorer_mod, _, _, _ = _get_backend_modules(args.backend)
    kwargs = {}
    quantize = getattr(args, "quantize", "none")
    if args.backend == "torch":
        kwargs["quantize"] = quantize
    elif quantize != "none":
        raise SystemExit("--quantize requires --backend torch")
    if getattr(args, "adapter", None):
        kwargs["adapter_path"] = args.adapter
    model_path = args.model or scorer_mod.DEFAULT_MODEL
    s = scorer_mod.OptionScorer(model_path, batch_size=args.batch_size, chat=args.chat, sep=args.sep, **kwargs)
    print(f"loaded {model_path} in {time.perf_counter() - t:.1f}s", file=sys.stderr)
    return s


def _read_options(path: str) -> list[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def cmd_score(args: argparse.Namespace) -> None:
    if args.options_file:
        args.option = _read_options(args.options_file) + args.option
    if len(args.option) < 2:
        sys.exit("need at least two options (--option ... or --options-file)")
    scorer = _scorer(args)
    result = scorer.score(args.context, args.option, norm=args.norm)
    scorer.score(args.context, args.option, norm=args.norm)  # warm second run for timing
    payload = {
        "context": args.context,
        "norm": args.norm,
        "timing": scorer.last_timing,
        "options": [r.to_dict() for r in result],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    ranked = sorted(result, key=lambda r: -r.score)
    for r in ranked:
        print(f"{r.probability:7.3%}  score={r.score:8.3f}  sum={r.logprob_sum:8.3f}  "
              f"n={r.n_tokens:3d}  {r.option!r}")
    t = scorer.last_timing
    print(f"[{t['context_tokens']} ctx tok, {t['option_tokens']} opt tok] "
          f"prefill {t['prefill_s']*1000:.0f} ms, options {t['options_s']*1000:.0f} ms, "
          f"total {t['total_s']*1000:.0f} ms", file=sys.stderr)


def cmd_eval(args: argparse.Namespace) -> None:
    scorer = _scorer(args)
    scorer_mod, _, _, _ = _get_backend_modules(args.backend)
    rows = list(scorer_mod.iter_jsonl(args.data))
    if args.limit:
        rows = rows[: args.limit]
    fixed = _read_options(args.fixed_options) if args.fixed_options else None
    top1 = top3 = labelled = 0
    lat: list[float] = []
    certainties: list[float] = []
    for i, row in enumerate(rows):
        options = row.get("options") or fixed
        if not options:
            sys.exit(f"row {i} has no 'options' and no --fixed-options given")
        res = scorer.score(row["context"], options, norm=args.norm)
        lat.append(scorer.last_timing["total_s"])
        order = sorted(range(len(res)), key=lambda j: -res[j].score)
        label = row.get("label")
        labels = row.get("labels") or ([label] if label is not None else None)

        # Certainty index: probability margin between top-2
        probs = [r.probability for r in res]
        sorted_probs = sorted(probs, reverse=True)
        if len(sorted_probs) >= 2:
            certainty = sorted_probs[0] - sorted_probs[1]
        else:
            certainty = 1.0
        certainties.append(certainty)

        if labels is not None:
            labelled += 1
            if len(labels) == 1:
                top1 += order[0] in labels
                top3 += any(l in order[:3] for l in labels)
            else:
                # Multi-label: top-1 must be one of the valid labels
                top1 += order[0] in labels
                top3 += sum(1 for l in labels if l in order[:3]) / len(labels)
        if args.verbose or label is None:
            print(json.dumps({"i": i, "label": labels, "pred": order[0], "pred_option": options[order[0]],
                              "probs": [round(p, 4) for p in probs], "certainty": round(certainty, 4)}))
    n = len(rows)
    print(json.dumps({
        "examples": n,
        "labelled": labelled,
        "top1": top1 / labelled if labelled else None,
        "top3": top3 / labelled if labelled else None,
        "median_certainty": statistics.median(certainties) if certainties else None,
        "low_certainty_count": sum(1 for c in certainties if c < 0.3),
        "norm": args.norm,
        "median_latency_s": statistics.median(lat),
        "mean_latency_s": statistics.fmean(lat),
    }, indent=2))


def _synthetic(scorer, ctx_tokens: int, n_opts: int, opt_tokens: int, seed: int):
    rng = random.Random(seed)
    words = ("river stone cloud engine quiet market ledger signal orbit velvet "
             "harbour lantern cipher meadow granite").split()
    def text(n):  # roughly n tokens of plain words
        return " ".join(rng.choice(words) for _ in range(n))
    context = text(ctx_tokens)
    options = [text(opt_tokens) for _ in range(n_opts)]
    return context, options


def cmd_bench(args: argparse.Namespace) -> None:
    scorer = _scorer(args)
    context, options = _synthetic(scorer, args.context_tokens, args.options, args.option_tokens, 0)
    ctx_n = len(scorer.context_ids(context))
    opt_n = sum(len(scorer.option_ids(o)) for o in options)
    scorer.score(context, options, norm="mean")  # warm-up
    scorer.score_naive(context, options[:1])

    cached = []
    for _ in range(args.repeat):
        scorer.score(context, options, norm="mean")
        cached.append(scorer.last_timing["total_s"])
    naive = []
    for _ in range(args.repeat):
        t = time.perf_counter()
        scorer.score_naive(context, options)
        naive.append(time.perf_counter() - t)
    print(json.dumps({
        "context_tokens": ctx_n,
        "options": len(options),
        "option_tokens_total": opt_n,
        "batch_size": scorer.batch_size,
        "prefix_cached_batched_s": {"median": statistics.median(cached), "min": min(cached)},
        "naive_per_option_s": {"median": statistics.median(naive), "min": min(naive)},
        "speedup": statistics.median(naive) / statistics.median(cached),
    }, indent=2))


def cmd_check(args: argparse.Namespace) -> None:
    """Verify the prefix-shared batched path against naive full re-encoding."""
    scorer = _scorer(args)
    cases = [
        ("The capital of France is", [" Paris", " Berlin", " a city in Europe", " Lyon"]),
        ("Q: What is 2 + 2?\nA:", [" 4", " 5", " four", " twenty-two"]),
    ]
    context, options = _synthetic(scorer, args.context_tokens, args.options, args.option_tokens, 1)
    cases.append((context, options))
    worst = 0.0
    for ctx, opts in cases:
        fast = [r.logprob_sum for r in scorer.score(ctx, opts, norm="sum")]
        slow = scorer.score_naive(ctx, opts)
        diffs = [abs(a - b) for a, b in zip(fast, slow)]
        worst = max(worst, max(diffs))
        rel = max(d / max(1e-6, abs(b)) for d, b in zip(diffs, slow))
        print(f"ctx_tokens={len(scorer.context_ids(ctx)):4d} options={len(opts):2d} "
              f"max_abs_diff={max(diffs):.4f} max_rel_diff={rel:.4%}")
        for o, a, b in zip(opts, fast, slow):
            print(f"    cached={a:9.3f} naive={b:9.3f}  {o[:40]!r}")
    ok = worst < args.tol
    print("OK" if ok else f"MISMATCH (worst abs diff {worst:.4f} > tol {args.tol})")
    sys.exit(0 if ok else 1)


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    heads = {t: p for t, p in (("choice", args.head_choice), ("score", args.head_score),
                               ("noul", args.head_noul)) if p}
    lora = {}
    for spec in args.lora:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--lora expects NAME=PATH, got {spec!r}")
        lora[name] = path
    serve(args.host, args.port, args.model, args.batch_size, args.backend, args.quantize, heads or None, lora or None)


def cmd_features(args: argparse.Namespace) -> None:
    _, features_mod, _, _ = _get_backend_modules(args.backend)
    scorer = _scorer(args)
    meta = features_mod.extract_dataset(scorer, args.data, args.out, limit=args.limit, chat=args.chat, sep=args.sep,
                                       contextual=args.contextual)
    print(json.dumps(meta))


def cmd_train(args: argparse.Namespace) -> None:
    _, _, train_mod, _ = _get_backend_modules(args.backend)
    print(json.dumps(train_mod.train(args.train, args.validation, args.out, rank=args.rank, epochs=args.epochs,
                                     batch_size=args.batch_size, lr=args.learning_rate, seed=args.seed)))


def cmd_eval_head(args: argparse.Namespace) -> None:
    _, features_mod, train_mod, head_mod = _get_backend_modules(args.backend)
    head_obj, cfg = head_mod.AttentionHead.load(args.checkpoint)
    fs = features_mod.FeatureSet(args.features)
    print(json.dumps({"model": train_mod.evaluate(head_obj, fs),
                      "shuffled_context": train_mod.evaluate(head_obj, fs, shuffle_context=True),
                      "checkpoint": args.checkpoint, "rank": cfg["rank"]}, indent=2))


def cmd_lora(args: argparse.Namespace) -> None:
    from . import lora_torch

    r = lora_torch.train(args.model, args.train, args.validation, args.out, test_path=args.test, sep=args.sep,
                         quantize=args.quantize, rank=args.rank, alpha=args.alpha, epochs=args.epochs,
                         lr=args.learning_rate, accum=args.accum, limit=args.limit,
                         eval_every=args.eval_every, seed=args.seed,
                         resume=args.resume, max_rows=args.max_rows, chat=args.chat)
    print(json.dumps({k: r[k] for k in ("best_val_top1", "zero_shot", "test", "test_shuffled_context") if k in r}))


def cmd_lora_eval(args: argparse.Namespace) -> None:
    from . import lora_torch

    print(json.dumps(lora_torch.eval_adapter(args.model, args.adapter, args.data, sep=args.sep,
                                             quantize=args.quantize, zero_shot=not args.no_zero_shot,
                                             chat=args.chat), indent=2))


def cmd_lora_quantize(args: argparse.Namespace) -> None:
    from . import lora_torch

    print(json.dumps(lora_torch.quantize_int8(args.adapter, args.out or args.adapter.rstrip("/") + "-int8")))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="openjev", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score options for one context")
    _add_common(s)
    s.add_argument("--context", required=True)
    s.add_argument("--option", action="append", default=[], help="repeat for each option")
    s.add_argument("--options-file", help="text file with one predefined option per line")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_score)

    e = sub.add_parser("eval", help="top-k accuracy on jevlike-style JSONL {context, options, label}")
    _add_common(e)
    e.add_argument("data")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--fixed-options", help="text file of options used for every row lacking an 'options' field")
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(fn=cmd_eval)

    for name, fn, helptext in (
        ("bench", cmd_bench, "latency of prefix-cached batched scoring vs naive re-encoding"),
        ("check", cmd_check, "verify cached batched scores match naive full-sequence scores"),
    ):
        b = sub.add_parser(name, help=helptext)
        _add_common(b)
        b.add_argument("--context-tokens", type=int, default=200)
        b.add_argument("--options", type=int, default=8)
        b.add_argument("--option-tokens", type=int, default=30)
        b.add_argument("--repeat", type=int, default=5)
        b.add_argument("--tol", type=float, default=0.5, help="check: max abs log-prob diff allowed")
        b.set_defaults(fn=fn)

    f = sub.add_parser("features", help="cache frozen-Gemma features for a JSONL file into an .npz")
    _add_common(f)
    f.add_argument("data")
    f.add_argument("--out", required=True)
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--contextual", action="store_true",
                   help="encode options as continuations of the context (leaks the match into option features)")
    f.set_defaults(fn=cmd_features)

    tr = sub.add_parser("train", help="train the attention head on cached features")
    tr.add_argument("train")
    tr.add_argument("--validation", required=True)
    tr.add_argument("--out", default="runs/head.safetensors")
    tr.add_argument("--rank", type=int, default=256)
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--learning-rate", type=float, default=5e-4)
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--backend", choices=["mlx", "torch"], default="mlx",
                    help="scoring backend: mlx (default, Apple silicon) or torch (PyTorch)")
    tr.set_defaults(fn=cmd_train)

    eh = sub.add_parser("eval-head", help="top-k, ECE and shuffled-context control for a trained head")
    eh.add_argument("checkpoint")
    eh.add_argument("features")
    eh.add_argument("--backend", choices=["mlx", "torch"], default="mlx",
                    help="scoring backend: mlx (default, Apple silicon) or torch (PyTorch)")
    eh.set_defaults(fn=cmd_eval_head)

    lo = sub.add_parser("lora", help="LoRA-tune the scorer (torch + peft, QLoRA) on JSONL rows")
    lo.add_argument("train")
    lo.add_argument("--validation", required=True)
    lo.add_argument("--test", default=None, help="also report zero-shot vs tuned on this split")
    lo.add_argument("--out", default="runs/lora")
    lo.add_argument("--model", default=DEFAULT_MODEL)
    lo.add_argument("--sep", default="\nChoice: ")
    lo.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="4bit")
    lo.add_argument("--rank", type=int, default=16)
    lo.add_argument("--alpha", type=int, default=32)
    lo.add_argument("--epochs", type=int, default=1)
    lo.add_argument("--learning-rate", type=float, default=2e-4)
    lo.add_argument("--accum", type=int, default=8, help="rows per optimiser step")
    lo.add_argument("--limit", type=int, default=0, help="use only the first N training rows")
    lo.add_argument("--eval-every", type=int, default=0, help="validate every N rows (default: once per epoch)")
    lo.add_argument("--seed", type=int, default=7)
    lo.add_argument("--resume", default=None, help="continue training from this saved adapter dir")
    lo.add_argument("--chat", action="store_true",
                    help="chat format: context + option list as a user turn (much stronger on instruction-tuned models)")
    lo.add_argument("--max-rows", type=int, default=0, help="stop after this many training rows")
    lo.set_defaults(fn=cmd_lora)

    le = sub.add_parser("lora-eval", help="top-k, ECE and shuffled-context control for a LoRA adapter vs zero-shot")
    le.add_argument("adapter")
    le.add_argument("data")
    le.add_argument("--model", default=DEFAULT_MODEL)
    le.add_argument("--sep", default="\nChoice: ")
    le.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="4bit")
    le.add_argument("--chat", action="store_true",
                    help="chat format: context + option list as a user turn (much stronger on instruction-tuned models)")
    le.add_argument("--no-zero-shot", action="store_true", help="skip the adapter-disabled baseline")
    le.set_defaults(fn=cmd_lora_eval)

    lq = sub.add_parser("lora-quantize", help="store a LoRA adapter as int8 + per-row scales (~4x smaller)")
    lq.add_argument("adapter")
    lq.add_argument("--out", default=None, help="output dir (default: ADAPTER-int8)")
    lq.set_defaults(fn=cmd_lora_quantize)

    v = sub.add_parser("serve", help="HTTP server with the model loaded once (POST /score, /v1/systemone)")
    v.add_argument("--model", default=None)
    v.add_argument("--batch-size", type=int, default=8)
    v.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    v.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--backend", choices=["mlx", "torch"], default="mlx",
                   help="scoring backend: mlx (default, Apple silicon) or torch (PyTorch)")
    v.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none",
                     help="torch backend only: load the weights quantised via bitsandbytes")
    v.add_argument("--head-choice", default=None, help="Route-A choice head checkpoint for /score and choice questions")
    v.add_argument("--head-score", default=None, help="Route-A score head checkpoint for score questions")
    v.add_argument("--head-noul", default=None, help="Route-A noul head checkpoint for noul questions")
    v.add_argument("--lora", action="append", default=[], metavar="NAME=PATH",
                   help="LoRA mode (torch): load an adapter under NAME; repeatable. NAME choice/score/noul "
                        "also serves that /v1/systemone question type")
    v.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    if len(getattr(args, "option", []) or []) == 1:
        p.error("--option must be given at least twice")
    args.fn(args)


if __name__ == "__main__":
    main()
