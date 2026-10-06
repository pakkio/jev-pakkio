# Jev-Mem on openjev (arXiv 2609.23986)

Goal: `openjev/mem/` package implementing the paper's System-One-controlled agentic memory, using this repo's
local `/v1/systemone` (Noul / Choice questions) as the controller. System Two = pluggable callable (default: none / echo evidence).

## Design (from paper)
- Memory plane: nodes {id, content, timestamp, entities, type scores, embedding}; edge views G = semantic|temporal|causal|entity; vector + lexical (BM25-ish) index.
- Write path (one batched call each): typing (4 Nouls: episodic/semantic/procedural/preference) -> candidates (<=10, vector+lexical+entity+time)
  -> relations (Nouls semantic, caused_by, causes per candidate; edge if >=0.60). Timestamp order -> temporal edge, shared entity ids -> entity edge deterministically.
- Consolidation every 20 writes (redundancy/contradiction/obsolescence/link-usefulness Nouls + representation Choice); records decisions only.
- Read path: route (6 Nouls: semantic/temporal/causal/entity need, multi_hop_need, recency_importance) -> budget split (B=80, min 1 for p>=0.10, gamma=1, largest remainder)
  -> depth D=min(Dmax,max(1,ceil(Dmax*h))) -> anchors via RRF(k=60) -> loop: assess evidence (sufficiency/utility/missing/contradiction Nouls) -> stop rule (eq 21/22)
  -> expand with per-graph budgets -> score candidates (relevance, usefulness, novelty, support) -> eq 23-25 score -> top-W beam -> top-K to System Two.
- Unspecified in paper (my choices, to be flagged): theta_act=0.10, theta_suff/theta_cont, lambda weights, W, K, Dmax, embedding model.

## Tasks
- [x] 1. `openjev/mem/store.py`: nodes, typed edges, vector (numpy) + lexical index, RRF, JSON persistence
- [x] 2. `openjev/mem/control.py`: Jev client protocol (HTTP to /v1/systemone or in-process `system_one(scorer, ...)`), question builders from paper App. B
- [x] 3. `openjev/mem/write.py`: typing, candidate discovery, relation judgment, consolidation
- [x] 4. `openjev/mem/read.py`: routing, budget (eq 13-15), anchors, expansion loop, scoring (eq 23-25), stopping
- [x] 5. `tests/test_mem.py`: budget/depth math, RRF, stop rule, bicycle example from App. B using a fake controller (no model needed)
- [x] 6. `openjev mem` CLI subcommands (add/query) + docs/mem.md
- [ ] 7. Verify: unit tests pass; live smoke test against `openjev serve` if the model is available locally

## Open decisions (need user)
1. Embeddings: no embedding model exists in repo. Options: Gemma mean-pooled hidden states (no new deps) vs sentence-transformers (new dep).
2. Scope: library + tests only, or also a LoCoMo eval harness (paper's benchmark; needs dataset + LLM judge)?
3. System Two: leave pluggable (recommended) or wire to an API?

## Review
- 15 unit tests in tests/test_mem.py pass (fake controller). Remaining 3 suite errors are pre-existing (torch not installed here).
- Fixed pre-existing duplicate `--backend` in cli.py (bad merge) because it crashed every `openjev` command.
- Not done: live smoke test with real model, LoCoMo eval, implicit temporal Choice.
