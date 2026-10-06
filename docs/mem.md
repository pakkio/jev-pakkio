# Jev-Mem agentic memory

`jev_pakkio.mem` implements the architecture of *Jev-Mem: System-One-Controlled Agentic Memory*
(Jiang, Li, Li; arXiv:2609.23986) on top of the local `/v1/systemone` endpoint. Every memory-control
decision (typing, relation judgment, routing, candidate scoring, stopping) is a typed Noul/Choice
question answered by one batched forward pass. Nothing is generated. System Two (answer synthesis) is a
callable you supply.

```python
from jev_pakkio.mem import JevMem, HTTPController, SentenceTransformerEmbedder

mem = JevMem(HTTPController("http://127.0.0.1:8000"), SentenceTransformerEmbedder(),
             system_two=lambda query, nodes: my_llm(query, [n.content for n in nodes]))
mem.add("Mira: My old bicycle broke.", timestamp=1715680800, entities=["Mira", "bicycle"])
mem.add("Mira: I bought a new bicycle yesterday because my old one broke.", timestamp=1715853600)
res = mem.query("When did Mira buy a new bicycle, and why?")
res.evidence, res.trace, res.answer
```

CLI (needs `uv sync --extra mem` and `jev_pakkio serve` running):

```sh
jev_pakkio mem --store jevmem.json add "Mira: My old bicycle broke." --timestamp 2024-05-14T10:00
jev_pakkio mem --store jevmem.json query "When did Mira buy a new bicycle, and why?"
```

## What follows the paper

- **Write** (one typing call, one batched relation call over at most 10 candidates): four overlapping type
  scores; deterministic temporal and entity edges; semantic and directed causal edges when P >= 0.60;
  every non-empty observation is kept; consolidation every 20 writes records decisions without deleting.
- **Read**: six routing Nouls, budget split (eq 12-15, B=80), RRF anchors (kappa=60), per-round evidence
  assessment, stop rules (eq 21-22), candidate scoring (eq 23) with recency adjustment (eq 24-25), top-K to System Two.

## Choices the paper does not specify

Defaults live in `WriteConfig` / `ReadConfig` and are guesses, not the authors' values: candidate score
weights, `theta_suff`/`theta_cont` (0.70/0.30), lambda weights, beam width, K, `d_max`, anchor count, the
embedder (MiniLM), the heuristic entity extractor, and the wording of the prompts not reproduced in the
paper's appendix. Implicit temporal ordering (the paper's before/after/overlaps Choice) is not implemented;
only timestamp-derived temporal edges are.

Not validated against LoCoMo. The unit tests use a keyword fake controller and check the mechanics only.

## Engines and MCP

`EngineController(get_engine("jev"))` runs Jev-Mem in-process on any registered engine (`jev`, `laya`, `4g`, `8g`),
with no HTTP server. `jev_pakkio mcp-engines` exposes it as tools: `memory_add(text, store, engine, timestamp,
entities)`, `memory_query(query, store, engine, top_k)` and `memory_stores()`. Stores are JSON files under
`JEVMEM_DIR` (default `./jevmem`); the engine is only the controller and can differ between calls on one store.
Needs the `mem` and `mcp` extras. Measured so far on the 10-question set: `jev` as controller gives recall@3 0.900
against 0.883 for flat top-3; `laya` and the local `4g`/`8g` engines have not been benchmarked as controllers.
