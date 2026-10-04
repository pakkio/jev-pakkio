"""Compare jev_pakkio against TypeSafe's hosted Jev on the same rows.

Each row is one System One `choice` question: a `state` and a set of options.
Both engines get identical input; we report top-1 agreement, per-engine
accuracy against the row's label, and median latency. Jev's responses are
cached under runs/jev/ so a rerun only pays for new rows.

The TypeSafe key is read from TYPESAFE_API_KEY in the environment or ../.env /
./.env. It is never printed, logged or written to disk.

    python compare_jev.py                       # built-in smoke set
    python compare_jev.py data.jsonl            # {"state","options","label"} per row
    python compare_jev.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "runs" / "jev" / "compare.jsonl"

# A few rows the hosted Jev can be asked straight up: unambiguous, so a
# disagreement means one engine is wrong rather than the label being arguable.
SMOKE = [
    {"state": "The capital of France is", "options": [" Paris", " Berlin", " Lyon", " Marseille"],
     "label": 0},
    {"state": "Water freezes at zero degrees on the Celsius scale.", "options": [" True", " False"],
     "label": 0},
    {"state": "A patient with a fever and a stiff neck may have meningitis, which needs urgent care.",
     "options": [" Seek emergency care now", " Wait a week and see", " Take aspirin and rest" ], "label": 0},
    {"state": "The user wrote: my package says delivered but nothing arrived.",
     "options": [" Shipping", " Billing", " Returns", " Account security"], "label": 0},
    {"state": "Which team won the 1966 World Cup?", "options": [" England", " West Germany", " Brazil"],
     "label": 0},
    {"state": "A recipe calls for 250 grams of flour.", "options": [" About 2 cups", " About 1 cup",
                                                                   " About 4 cups"], "label": 0},
    {"state": "The mitochondrion is the organelle responsible for producing ATP in the cell.",
     "options": [" Mitochondrion", " Ribosome", " Golgi apparatus", " Lysosome"], "label": 0},
    {"state": "Ticket prices went up and the service got worse; the customer is unhappy.",
     "options": [" Moderate", " Severe", " None"], "label": 1},
]


def load_key() -> str:
    """TYPESAFE_API_KEY from the environment, then ../.env, then ./.env."""
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    for path in (ROOT.parent / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return ""


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out: dict[str, dict] = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["state"]] = row
    return out


def append_cache(row: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def ask_jev(state: str, options: list[str], key: str) -> dict:
    """One System One choice request against TypeSafe's hosted Jev."""
    from typesafe_sdk import TypeSafeClient  # imported late: only needed for Jev rows

    client = TypeSafeClient(api_key=key)
    resp = client.system_one(
        state=state,
        model="jev-latest",
        questions={
            "pick": {
                "type": "choice",
                "instructions": "Which option answers the question or completes the statement?",
                "criteria": {opt: opt for opt in options},
            }
        },
    )
    answer = resp.answers["pick"]
    return {
        "choice": answer.choice,
        "confidence": getattr(answer, "confidence", None),
        "probabilities": getattr(answer, "probabilities", None),
    }


def local_answer(scorer, state: str, options: list[str], chat: bool) -> dict:
    t = time.perf_counter()
    results = scorer.score(state, options, norm="sum")
    ms = (time.perf_counter() - t) * 1000
    best = max(results, key=lambda r: r.probability)
    return {
        "choice": best.option,
        "best_index": best.index if hasattr(best, "index") else options.index(best.option),
        "confidence": best.probability,
        "probabilities": [r.probability for r in results],
        "ms": ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", help="JSONL of {state, options, label}; default: a smoke set")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows")
    ap.add_argument("--model", default=None, help="override the model (default: chosen by VRAM)")
    ap.add_argument("--quantize", default="4bit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-chat", action="store_true", help="score as plain text, not a chat turn")
    ap.add_argument("--out", default=None, help="write the full per-row report here")
    args = ap.parse_args()

    if args.data:
        rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        rows = SMOKE
    if args.limit:
        rows = rows[: args.limit]

    key = load_key()
    if not key:
        print("TYPESAFE_API_KEY not found (env, ../.env or ./.env) -- cannot query Jev.", file=sys.stderr)
        return 2

    from jev_pakkio.scorer_torch import OptionScorer, auto_model

    model = args.model or auto_model()
    print(f"local model: {model} ({args.quantize}, device {args.device})", file=sys.stderr)
    t0 = time.perf_counter()
    scorer = OptionScorer(model, quantize=args.quantize, device=args.device, chat=not args.no_chat)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    cache = load_cache()
    report, jev_ms, local_ms = [], [], []
    agree = correct_local = correct_jev = graded = 0

    for i, row in enumerate(rows, 1):
        state, options = row["state"], row["options"]
        label = row.get("label")

        local = local_answer(scorer, state, options, chat=not args.no_chat)
        local_ms.append(local["ms"])

        cached = cache.get(state)
        if cached:
            jev, from_cache = cached, True
        else:
            t = time.perf_counter()
            try:
                jev = ask_jev(state, options, key)
            except Exception as exc:  # network/auth/quota: keep going, report it
                jev = {"error": f"{type(exc).__name__}: {exc}"}
            jev["ms"] = (time.perf_counter() - t) * 1000
            append_cache({"state": state, "options": options, **jev})
            cached, from_cache = jev, False
        if jev.get("ms") is not None:
            jev_ms.append(jev["ms"])

        same = jev.get("choice") == local["choice"]
        agree += 1 if same else 0
        if label is not None:
            graded += 1
            correct_local += 1 if local["best_index"] == label else 0
            correct_jev += 1 if jev.get("choice") == options[label] else 0

        report.append({
            "state": state, "options": options, "label": label,
            "local_choice": local["choice"], "local_p": round(local["confidence"], 4),
            "local_ms": round(local["ms"], 1),
            "jev_choice": jev.get("choice"), "jev_p": jev.get("confidence"),
            "jev_ms": round(jev["ms"], 1) if jev.get("ms") is not None else None,
            "cached": from_cache, "agree": same, "error": jev.get("error"),
        })
        print(f"[{i}/{len(rows)}] agree={same}  local={local['choice']!r} jev={jev.get('choice')!r}",
              file=sys.stderr)

    n = len(report)
    print()
    print(f"rows            {n}" + (f" (graded {graded})" if graded else ""))
    print(f"agreement       {agree}/{n}" + (f" ({agree / n:.1%})" if n else ""))
    if graded:
        print(f"local top-1     {correct_local}/{graded}")
        print(f"jev   top-1     {correct_jev}/{graded}")
    if local_ms:
        print(f"local median    {statistics.median(local_ms):.0f} ms")
    if jev_ms:
        print(f"jev   median    {statistics.median(jev_ms):.0f} ms (network)")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nper-row report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())