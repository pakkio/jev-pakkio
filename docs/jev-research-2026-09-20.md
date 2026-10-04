# Jev, jevlike, and jev_pakkio: research report

Date: 2026-09-20. Scope: TypeSafe AI's commercial Jev model and "System One" API, the open-source jevlike starter that copies its input/output shape, the small ecosystem that grew around jevlike in the past week, and how jev_pakkio (this repo) sits relative to both. Sources are listed at the end of each part.

## Executive summary

- **Jev is TypeSafe AI's first "System One" model**, announced 2026-09-15 by founder Diogo Almeida. It takes unstructured `state` plus a map of typed `questions` and returns typed answers with calibrated probabilities in one non-autoregressive pass. Public model id `jev-latest` currently resolves to `jev-1.13.0`, served at `POST https://api.typesafe.ai/v1/systemone`.
- **Disclosed internals are thin.** TypeSafe names a new architecture, a "parallel sampler", and a training method called Reinforcement Learning for Calibrated Decisions (RLCD). It caps `choice` questions at 255 options and `score` questions at 10 levels, and promises schema-guaranteed output with structurally zero hallucination. Parameter count, backbone, tokenizer, and training data are all undisclosed. The CEO said only that it is "neither small nor an LLM" and that all training data is made in-house.
- **Claimed numbers:** 70 to 500 ms end to end, $0.042 per million input tokens with free output tokens, and homepage figures of 193.6x faster and 444.6x cheaper than frontier LLMs on TypeSafe's own workflow evals. Those evals use an average of GPT-6 Astra and Fable 5.1 as the reference answer, so they should be read as vendor benchmarks.
- **jevlike (vinnylarouge, "Minimal Labs") is an independent MIT starter**, not a reproduction. It is a three-commit, one-author repo created 2026-09-16 that hit about 1,065 stars and a 166-point Hacker News thread in four days. Its core is a single cross-attention head: each option becomes a query over context tokens, a dot product gives one logit per option, and a softmax across options gives probabilities. The default byte-level model has 41,280 parameters and truncates context to 192 bytes and options to 32 bytes.
- **The ecosystem is one week old.** jevbetter adds hashed n-gram encoding, rival-aware attention, a gated head, and temperature scaling and reports 0.916 versus 0.873 top-1 against the jevlike baseline. jevlike-esp32 exports float32 weights and reimplements inference in C for a microcontroller. JevLikeDiffusionGemma shares only the name. omo-jevlike-router is an archived skill-routing experiment.
- **jev_pakkio takes the opposite bet:** keep a pretrained 4B backbone (Gemma 3) and score options with one cached forward pass, either zero-shot from token log-probs or with jevlike's head retrained on frozen Gemma features. Zero-shot reaches 1.000 top-1 on the synthetic menus at about 90 ms per request, versus well under 10 ms for jevlike's tiny model. jev_pakkio also serves TypeSafe's System One request contract locally.

---

# Part 1: TypeSafe's Jev


## 1. What Jev is and the problem framing

TypeSafe AI (founder Diogo Almeida, ex-OpenAI, co-inventor of RLHF; two years in stealth) announced its first "System One Model", **Jev**, on Sep 15, 2026, in early access behind a waitlist. The pitch: chat models are superhuman at talking but have produced little automation because software cannot depend on free-form strings. Jev is framed as "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out." It takes a `state` (text or JSON) and a map of typed `questions`, and returns, per question, a value plus a probability distribution and a confidence. It generates no strings at all.

The naming comes from Kahneman's fast/intuitive System 1 vs. deliberate System 2 (TypeSafe claims System One models can be made *more* reliable than LLMs, reasons deferred), and from William Stanley Jevons: every order-of-magnitude drop in the cost of intelligence should unlock orders of magnitude more use cases (the Jevons paradox). Target uses: "smart if-statements" (classify, route, score, branch), map-reduce over petabytes, sub-100 ms real-time apps, and guardrails/judges.

## 2. What is disclosed about internals

- **Three primitives, mapped to code constructs.** `noul` (short for Bernoulli) = if-statement, returns P(yes); `choice` = match statement, returns a distribution over user-supplied options; `score` = sorting/ordinal rubric, returns a distribution over 2-10 ordered levels plus a probability-weighted mean. `confidence` (Choice and Score only; noul has none) is derived from the distribution, flatter means lower; the docs' demo formula is `(n x max_p - 1)/(n - 1)`. Question IDs are not sent to the model; option names and descriptions are.
- **Parallel sampler, no string generation.** Jev "outputs all probabilities in parallel instead of autoregressively generating by token." Almeida on HN: strings "and all sequential data structures" are disallowed as outputs, which is what lets all outputs be computed in a single pass with no output-token cost. Every question is evaluated independently against the same state; adding questions barely changes latency. Docs: "Jev ingests the `state` once and evaluates every question against it in parallel."
- **RLCD (Reinforcement Learning for Calibrated Decisions).** Positioned as a third post-training path next to RLHF and RLVR. Objective: return decisions and probabilities such that "higher probability should correspond to a greater chance that the answer is correct" (a 0.8 should be right about 80% of the time). RLHF is criticised for rewarding sycophancy and causing mode dropping. No algorithmic detail beyond this.
- **Schema-guaranteed outputs.** Outputs are typed values from a predefined set, so type errors are "mathematically impossible"; the hallucination chart plots 0% for Jev by construction. Almeida concedes Jev can still be confidently wrong within the valid set ("Would you say a linear classifier hallucinates?").
- **Cardinality.** Choice accepts at most **255 options**; Score at most **10 levels**. For high-cardinality choices TypeSafe runs "a 2 stage-system of scoring independently then making an explicit choice," which causes occasional slowdowns.
- **Limits.** Context 64k tokens per request; 32k for `state` plus the longest single question. Text only (no images/audio/video); English primary, CJK accepted but weaker. Same weights serve every account; no customer fine-tuning or LoRA; not trained on customer traffic. Documented weaknesses ("jaggedness" page): literal reading, no counting or arithmetic, multi-hop indirection, context rot from large irrelevant state, adversarial content can steer answers, and complementary nouls need not sum to 1 (one example summed to 1.19).
- **Not a language model.** "Jev is neither small nor an LLM"; Almeida endorses "Large Classification Model" and "zero-shot classifier over raw/structured text" as accurate descriptions. He also argues OpenAI-style constrained decoding "make[s] models dumber": mass on an invalid token means the model is confused and should error instead.

## 3. What is not disclosed

Parameter count ("neither small nor an LLM" is all), backbone/encoder type, whether it is transformer, diffusion, or something else, tokenizer, training data (all made in-house: "We make all the data ourselves"; "we'd have to hire you"), the RLCD reward, any calibration metrics (ECE etc.), the exact confidence formula, and any public benchmark results (deliberately withheld per their "antibenchmaxxing" stance; the ToS reportedly bars users from publishing benchmarks). Architecture is "close to the chest for now"; a paper is contemplated.

## 4. API shape

Single endpoint `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <API_KEY>`; the OpenAPI spec is public at `https://api.typesafe.ai/openapi.json` (FastAPI, version 0.2.0). `state` is `str | object | array`; `questions` is a non-empty map discriminated on `type`; `noul.criteria` optionally describes `true`/`false`; `choice.criteria` maps option name to description (or null); `score.criteria` is an ordered list whose index is the level. Models: `jev-latest` and `jev-preview` (both currently `jev-1.13.0`); `GET /v1/models` lists them. Errors: 401, 422 (validation), 429 (rate limit), 529 (overloaded). Official SDKs: `typesafe-sdk` (Python >= 3.10, v0.7.0, default timeout 10 s, retries on 408/429/5xx honoring `retry-after`) and `@typesafe-ai/sdk` (TypeScript, v0.6.0); Vercel AI SDK exposes it as `experimental_evaluate()`. `instructions` and `criteria` may be strings or nested objects/arrays. Request/response (abridged from the quickstart):

```json
// request
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days ... Please help ASAP.",
  "model": "jev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this",
      "criteria": {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems", "sales": "Pricing or account questions"}},
    "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
      "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}
  }
}
// response
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {"type": "choice", "choice": "technical", "confidence": 0.78,
      "probabilities": {"technical": 0.85, "sales": 0.0, "billing": 0.15}},
    "frustration": {"type": "score", "score": 1.0, "confidence": 1.0,
      "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"},
      "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}},
    "is_urgent": {"type": "noul", "noul": 1.0}
  },
  "usage": {"input_tokens": 392, "output_tokens": 65}
}
```

The open-source `system-one-adapter-python` (194 stars) reproduces this exact interface on top of OpenAI/Anthropic chat models with one chat call per request that asks the LLM to emit the whole answer map as JSON (`llm_answer_mode="probabilities"` asks for the distribution directly; `"discrete"` yields one-hot), using native structured outputs where available, renormalising invalid distributions and retrying malformed output. It uses no logprobs or repeated sampling, and computes Choice confidence as `(max_p - 1/n)/(1 - 1/n)` and Score as the probability-weighted level mean. It is the "System One LLM wrapper" used for the baselines in the evals. Other org repos: `typesafe-sdk-python`, `typesafe-sdk-js`, `skills` (1,053 stars), and forks of **vLLM** and **LLaDA** (masked-diffusion LM); the forks are unexplained.

## 5. Numbers

| Metric | Jev | LLM comparison (TypeSafe's figures) |
|---|---|---|
| Input price | $0.042/MTok ($42/BTok) | $0.20-$10/MTok |
| Output price | Free | ~5x input |
| End-to-end latency | 70-500 ms (evals: 0.3-0.5 s/case) | 3-329 s frontier; 6-241 s/case in workflow evals |
| Rate limits | 250,000 tok/s; 1,200 req/min (dynamic) | - |
| Context | 64k/request; 32k state + longest question | - |
| Headline claim | 193.6x faster, 444.6x cheaper (from workflow evals; "higher end of real world gains") | - |
| Workflow-eval accuracy (mean of 4) | 67.8%, $0.0004, 0.4 s/case | Opus 5 73.1% / $0.176 / 37.8 s; GPT-5.6 Sol 74.1% / $0.084 / 23.3 s; Terra 67.9% / $0.030 / 10.1 s; Sonnet 5 67.8% / $0.117 / 78.1 s; DeepSeek V4 Flash 64.4% / $0.006 / 51.9 s |
| Doom demo | 10 queries/s, ~$7/hour | - |

Eval methodology: four code-defined workflows (security incidents 240 cases, agent-trace observability 117, invoice processing 150, customer service 204; 711 total, 14-191 questions each), every model runs the same graph, and "accuracy" is agreement with the averaged probabilities of GPT-6 Astra and Claude Fable 5.1 at high thinking, not ground truth. Caveats TypeSafe states: workflows were built by its own team; the reference biases toward OpenAI/Anthropic; LLM hallucination rates come from OpenRouter; results were run from laptops on the West Coast. Jev is weakest on invoice processing (61.8% vs Sol 79.1%).

## 6. Demos

- **Side-by-side**: a short customer paragraph with human-readable question keys, Jev vs GPT-5.6 Terra; the only disagreement was "Churn likelihood level".
- **Doom**: a bot driven by structured text game state (not pixels, "not on images (yet...)"), ~10 decisions/s; TypeSafe admits a scripted bot would play better, the point is instruction-following on state.
- **Wikiracing**: choosing among hundreds to thousands of links per hop, exercising the 255-option cap and the two-stage scoring path; LLM opponents ran in non-reasoning modes, so speedups were smaller and Jev finished in fewer steps.
- Docs also ship a smart-home demo and cookbooks (confidence-routed classification, "autoresearch" feature discovery).

## 7. Informed speculation on how it likely works (not confirmed by TypeSafe)

- **Joint encoding with per-option heads.** The most consistent reading of "parallel sampler", no output tokens, per-question independence, and a hard 255-option cap is a single non-autoregressive forward pass over `state` plus each question's options, with a scalar logit per option and a softmax per question (Bernoulli sigmoid for noul, ordinal softmax for score). The 255 cap smells like a `uint8` option index or a fixed-width label slot in the sampler. HN's largest camp (quotemstr, txhwind, Vetch, several GLiNER/DeBERTa proponents) argues for an encoder-only or cross-encoder transformer with classification/regression heads; the "two-stage scoring then explicit choice" for large option sets is exactly what a late-interaction/bi-encoder prefilter plus a cross-encoder rerank would look like.
- **Size.** paraschopra estimates ~3B parameters from the $42/BTok price; Almeida only says "not small". The 70 ms floor is consistent with a few-billion-parameter encoder on a single GPU without decoding.
- **Diffusion hypothesis.** The org's LLaDA and vLLM forks, and a community vLLM PR turning DiffusionGemma into a Jev clone, fuel a masked-diffusion-LM theory (bigglebear, brausepulver, mmastrac). Counter-evidence from cmrdporcupine: a plain KV-shared autoregressive Gemma answered a first question in ~170 ms and each extra question in ~33 ms, faster than the diffusion clone, so diffusion is not needed to explain the latency.
- **Single-token logprob baseline.** Several open replications (TheoLeeCJ/jev_pakkio, ekzhang/jev_pakkio-sglang) emit one token per question and read logprobs; paraschopra bets Jev's probabilities correlate with LLM logprobs. lhk931122 notes constrained decoding renormalises over allowed tokens only, which is why raw logprob classifiers are poorly calibrated and why a dedicated calibration objective (RLCD) would matter.
- **RLCD guess.** Likely RL or distillation against strictly proper scoring rules (log/Brier) on synthetic decision data, possibly with logprob distillation from a larger teacher (paraschopra); Laya's arXiv 2510.01237 describes a similar ModernBERT policy with proper-scoring-rule rewards. Nobody outside TypeSafe knows what RLCD actually is.
- **Consensus on novelty.** Even skeptics (ramon156, jacobgold, zmmmmm on the Astra/Fable reference) mostly agree the product novelty is zero-shot, zero-training, calibrated probabilities over a fixed decision set at $42/BTok, i.e. a "generalised classifier" rather than a generalised token predictor.

## 8. Sources

- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/ and https://docs.typesafe.ai/llms.txt
- https://docs.typesafe.ai/introduction/quickstart.md
- https://docs.typesafe.ai/api.md
- https://docs.typesafe.ai/models.md
- https://docs.typesafe.ai/concepts/state.md
- https://docs.typesafe.ai/confidence.md
- https://docs.typesafe.ai/primitives/choice.md, https://docs.typesafe.ai/primitives/score.md, https://docs.typesafe.ai/primitives/noul.md, https://docs.typesafe.ai/primitives/advanced.md
- https://docs.typesafe.ai/introduction/machine-learning-primer.md
- https://evals.typesafe.ai/ (plus per-workflow pages security_incidents, agent_trace_observability, invoice_processing, customer_service and their `*-cases.js` data)
- https://github.com/typesafe-ai/system-one-adapter-python (README plus `src/system_one_adapter/{_client,_schema,_response}.py`, `_utils/confidence_metrics.py`, `providers/{openai,anthropic}.py`)
- https://github.com/typesafe-ai/typesafe-sdk-python (`src/typesafe_sdk/_core/*`, `_schemas/models.py`) and https://github.com/typesafe-ai/skills (`skills/typesafe-ai/SKILL.md`)
- https://api.typesafe.ai/openapi.json
- https://docs.typesafe.ai/model-jaggedness/jev-1.13.md and https://docs.typesafe.ai/llms-full.txt
- https://api.github.com/orgs/typesafe-ai/repos
- https://hn.algolia.com/api/v1/search?query=typesafe%20jev
- https://news.ycombinator.com/item?id=49717558 (launch thread, CEO replies as CompleteSkeptic)
- https://news.ycombinator.com/item?id=49752041 (OpenJev thread)
- https://news.ycombinator.com/item?id=49765348 (Laya / non-autoregressive decision models thread)
- https://vercel.com/kb/guide/typesafe-jev-and-ai-sdk
- https://mikulskibartosz.name/typesafe-jev-guess-what-i-drew


---

# Part 2: jevlike and its ecosystem


Research snapshot compiled 2026-09-20 from the GitHub API, raw source files, and the Hacker News Algolia API.

## 1. What jevlike is

`vinnylarouge/jevlike` (MIT, Python, 1,065 stars, 92 forks, 5 open issues/PRs, created 2026-09-16T10:26Z) is a "research starter" that trains a small model to "choose among a changing list of text options": a context string plus N option strings go in, one probability per option comes out, in a single forward pass rather than token-by-token decoding.

Provenance is stated plainly in the README: Jev "is TypeSafe's commercial model for this kind of task. TypeSafe has not published its design. This repository is an independent starter model with the same input and output shape." It adds: "We did not show equal quality with Jev or reproduce TypeSafe's private training method" and "This is a research starter, not a copy of Jev." The only relationship to Jev is the I/O contract (context + options -> calibrated distribution). Package metadata: `jevlike` 0.1.0, author "Minimal Labs", `requires-python >=3.10`, deps `numpy>=1.26`, `torch>=2.2`; extras `transformers` (`transformers>=4.45`), `dev` (`pytest>=8`), `games` (`chess`, `imageio`, `imageio-ffmpeg`, `pillow`, `vizdoom>=1.2.4`). Commits are co-authored with OpenAI Codex and Claude.

## 2. Model internals

The whole text model lives in `jevlike/model.py` (142 lines): `TinyScorer` (byte encoder) and `FrozenTransformerScorer` (HF path) share one `AttentionHead`.

**Byte embedding.** `data.py` maps text to `byte + 1` over UTF-8 (`errors="replace"`), so ids are 1..256 and 0 is padding. `nn.Embedding(257, width, padding_idx=0)`. Context is truncated to `--context-tokens` (default **192** bytes) and each option to `--option-tokens` (default **32** bytes); batches pad to the batch max, not the fixed length. Context gets a learned absolute position table `nn.Embedding(192, width)`; option bytes get no positions and are masked-mean-pooled into one vector per option.

**AttentionHead(input_width, rank)**: two `LayerNorm`s (context, option) and three bias-free `Linear(width -> rank)` projections Q, K, V.

```
context_ids (B,Lc<=192) --Emb(257,d)+Pos(192,d)--> (B,Lc,d) --LN--> K,V = Linear -> (B,Lc,r)
option_ids  (B,O,Lo<=32) --Emb(257,d)--> (B,O,Lo,d) --masked mean over Lo--> (B,O,d) --LN--> Q -> (B,O,r)
scores   = einsum("bnr,blr->bnl", Q, K) / sqrt(r)       (B,O,Lc)   pad-context masked to -inf
attended = softmax(scores, -1) @ V                       (B,O,r)    one context vector per option
logits   = sum(Q * attended, -1) / sqrt(r)               (B,O)      pad-option masked to -inf
train: F.cross_entropy(logits, label)   eval/predict: logits.softmax(-1) over the O options
```

Scoring is a plain dot product between each option's query and its own attended context vector, scaled 1/sqrt(rank) a second time; there is no bilinear matrix, bias, dropout, FFN, second layer, or label smoothing. The shuffled-context control is built into the head: `context.roll(1, dims=0)` pairs each menu with the previous example's context.

**Defaults and parameter count.** `--width 64 --rank 64 --context-tokens 192`: embedding 257x64 = 16,448; positions 192x64 = 12,288; 2 LayerNorms = 256; Q,K,V 3x64x64 = 12,288; **total 41,280 parameters**. The smoke-test config (32/32/128) is 15,520.

**Training** (`jevlike-train`): `AdamW(lr=2e-3, weight_decay=1e-4)`, `--epochs 8`, `--batch-size 64`, `--seed 7`, `clip_grad_norm_(1.0)`, no scheduler, no early stopping; the state at lowest validation NLL is kept. Checkpoint = `torch.save({"config": {encoder, hf_model, width, rank, context_tokens, option_tokens}, "state_dict": <trainable params only>})`. `load_checkpoint` uses `torch.load(weights_only=False)` (issue #3) and `strict=False` so frozen `encoder.*` keys may be missing.

**HF frozen-encoder path** (`--encoder hf`, default `Qwen/Qwen2.5-0.5B`, hidden 896): `AutoModel` is `requires_grad_(False).eval()` under `no_grad`; context uses all `last_hidden_state` tokens as K/V, options are masked-mean-pooled; only `AttentionHead(896, rank)` trains. README recipe `--rank 256 --batch-size 8` gives 691,712 trainable params; the checkpoint stores the head plus the encoder name, not encoder weights.

**Vision** (`jevlike/vision.py`, `DoomScorerV2(actions, width=32, rank=32, reads=1)`): a 4-channel 120x160 frame (RGB + motion channel) goes through `Conv2d(4,16,5,s2)+GroupNorm+SiLU`, `Conv2d(16,w,3,s2)+GroupNorm+SiLU`, then a 4x4/stride-4 patch conv into an 8x10 grid of 80 patch tokens. Options are a learned 12-row `nn.Embedding` table (Doom ids 0-6, chess 7-11) instead of text. The same `AttentionHead` scores them, with fixed 2-D sinusoidal positions added to the keys after projection. A value head (`LayerNorm -> Linear(w,1)`) supports PPO. About 26.3k parameters at width 32; shipped checkpoints are ~126 KB.

## 3. Data format, recipe, and metrics

One JSON object per line: `{"context": "The customer needs a refund.", "options": ["refund", "sales", "technical support"], "label": 0}`. `label` is the zero-based correct index; rows may have different option counts, minimum two. Validation errors: "options must contain at least two non-empty strings", "label must be an option index".

`jevlike-data synthetic` builds menus of 2-8 badges from 8 colours x 8 animals (`--train 2000 --validation 400 --test 400 --seed 17`). `jevlike-data wikispeedia` turns `paths_finished.tsv` into one next-click step per path (`max_options=64`: clicked link + up to 63 others), context `"Target article: {t}\nCurrent article: {c}\n{body[:2048]}"`, with a target-disjoint split by `sha256(target)`.

`jevlike-eval` reports per split: **top-1**, **top-3** (`logits.topk(3)`), **ECE** (10 equal-width confidence bins on max softmax probability, `sum(frac_bin * |acc_bin - conf_bin|)`), `examples`, and a full second pass with `shuffle_context=True` as the **shuffled-context control**. NLL is printed during training only. Issue #1 notes the control leaks the target on Wikispeedia (39% of shuffled pairs share it).

## 4. Benchmark numbers

| Setting | Result | Source |
|---|---|---|
| Synthetic menus, tiny byte model | ~98% accuracy | README |
| Wikispeedia, frozen Qwen2.5-0.5B + head | 26% top-1 vs ~8% shuffled/random-encoder controls | README |
| Wikispeedia, from-scratch on 40,000 clicks | 29% top-1 | README |
| 8 options, one pass vs small decoder writing 400 tokens | ~100x faster | README |
| Doom `deadly_corridor`, joint 12-option checkpoint, 10 episodes | 0.60 mean kills, -97.50 reward (random baseline 0.4 kills) | README / scripts |
| Chess vs random mover, 50 games | 4 W / 46 D / 0 L, 4% wasted presses, 11.8 keys/move, CPL 288 | chess README |
| Chess vs Stockfish level 0 | 0 W / 2 D / 48 L, 13.0 keys/move, CPL 155 | chess README |
| Chess vs Stockfish level 3 | 0 W / 0 D / 50 L, 13.3 keys/move, CPL 144 | chess README |
| Chess inference | ~0.7 ms per key press on Apple MPS | chess README |

The README stresses these "describe local experiments, not this quickstart run", and the chess docs say the model "learned the controller but not chess" (cursor-square reading 99.95%, next-key accuracy ~65% unmarked).

## 5. CLI reference

| Command | Purpose | Key flags |
|---|---|---|
| `jevlike-data synthetic --output DIR` | generate JSONL splits | `--train 2000 --validation 400 --test 400 --seed 17` |
| `jevlike-data wikispeedia --root R --output O` | build next-click dataset | after `scripts/get_wikispeedia.sh` |
| `jevlike-train TRAIN --validation VAL --output runs/model.pt` | train | `--encoder {tiny,hf}`, `--hf-model Qwen/Qwen2.5-0.5B`, `--width 64`, `--rank 64`, `--context-tokens 192`, `--option-tokens 32`, `--epochs 8`, `--batch-size 64`, `--learning-rate 2e-3`, `--device {auto,cpu,mps,cuda}`, `--seed 7` |
| `jevlike-eval CKPT DATA` | metrics JSON (model + shuffled_context) | `--batch-size 64`, `--device` |
| `jevlike-predict CKPT --context STR --option A --option B ...` | JSON list of `{option, probability}` | needs at least two `--option` |

Doom/chess scripts (`examples/doom/{train_imitation,train_dagger,train_ppo,train_joint,train_joint_ppo,play,audit}.py`, `examples/chess/{positions,train,dagger,eval,play}.py`) are separate entry points; `examples/film/make-film.sh TRACE OUTPUT [SECONDS]` renders traces with Playwright + ffmpeg.

## 6. Ecosystem

### jevbetter (olanotolu/jevbetter)
13 stars, 3 forks, MIT, created 2026-09-16T20:04Z, 24 commits in 72 minutes by Ola Adu, not a GitHub fork. "Same JSONL data format as jevlike" but a different architecture: `NgramScorer(width=128, buckets=65536, heads=4, layers=2)`. **Hashed n-grams**: character 3-6-grams of `"<"+text.lower()+">"`, id = `zlib.crc32(gram) % 65535 + 1`, up to 512 features per context and 64 per option, in one shared `Embedding(65536,128)` (~8.4M of ~8.9M params). Context runs through a 2-layer `nn.TransformerEncoder` (FFN 256, dropout 0). **Rival-aware attention**: `nn.MultiheadAttention` self-attention across the option axis (`LayerNorm(options + mixed)`) before cross-attention into context. **Gated head**: `interaction = rival_aware * attended`; `gated = interaction * sigmoid(Linear(128,128)(interaction))`; MLP `Linear(128,128) -> SiLU -> Linear(128,1)`. **Temperature scaling**: post-hoc grid search over `logspace(0.05, 20, 80)` on validation NLL, stored as a float and applied as `logits / T`. Training adds warmup 5% + cosine, early stopping (patience 3), optional label smoothing, 12 epochs. Benchmark (800 held-out "hard" synthetic menus, CPU, 8 epochs, same seed):

| model | top-1 | top-3 | MRR | ECE | menus/sec |
|---|---|---|---|---|---|
| jevlike | 0.873 | 0.995 | n/a | 0.0367 | 4608 |
| jevbetter | 0.916 | 0.999 | 0.955 | 0.0182 | 40 |

Shuffled-context control 0.335 top-1. No Wikispeedia or other real-data numbers; throughput measurement is asymmetric between the two.

### jevlike-esp32 (david-cermak/jevlike-esp32)
1 star, MIT, 5 commits on 2026-09-18, "Jevlike edge router on ESP32": a 3-way router (`command` / `weather` / `complex`, 41 training rows). `export_firmware.py` loads a tiny-encoder `.pt` via `jevlike.model.load_checkpoint` and writes a custom `weights.bin`: header `"JEVLIKE1"` + 6 uint32 (width, rank, context_tokens, option_tokens, 257, n_options), then **float32** tensors (embedding, position, both LayerNorms, Q, K, V) and NUL-terminated option names, embedded with ESP-IDF `EMBED_FILES`. No quantization. `jevlike_scorer.c` (251 lines, plain C, `-lm`) reimplements byte tokenizing, mean pooling, LayerNorm (eps 1e-5), the QKV projections, per-option softmax attention and the final softmax; limits `context_tokens <= 192`, `option_tokens <= 32`, `JEV_MAX_OPTIONS = 4`. Target is plain `esp32` (4 MB flash, 16 KB main stack, UART 115200). `host_check.c` verifies C vs PyTorch within 2e-5. Routing threshold 0.45: `weather` -> local client, `command` -> GPIO light, else forward to a cloud LLM. No latency or RAM figures are published.

### JevLikeDiffusionGemma (pahndev/JevLikeDiffusionGemma)
0 stars, 1 fork, no license, 1 commit 2026-09-20 ("upload for quick poc"). Despite the name it does not use the jevlike code. It ports the structured-read idea from vLLM PR #57250 to MLX: `mlx-community/diffusiongemma-26B-A4B-it-4bit` (~16.6 GB) is used read-only; a 32-token canvas `answer: <label>` is built so all labels differ at one slot, that slot is denoised 1-48 passes, and label probabilities are the re-softmaxed log-probs of label tokens at that slot. Question types `noul` (yes/no), `choice` (2-26 letters), `score` (expected level). FastAPI server `POST /v1/evaluate`; warm ~0.62 s per question on Apple Silicon (32 GB recommended). README: "This is not TypeSafe Jev ... The project name indicates a similar decision interface, not affiliation."

### omo-jevlike-router (islee23520/omo-jevlike-router, archived)
0 stars, MIT, 3 commits on 2026-09-18, archived without a note. A skill router for OmO (`code-yeongyu/oh-my-openagent`): on every agent start a frozen `Qwen/Qwen2.5-0.5B` + `jevlike.model.AttentionHead(rank=256)` scores the whole skill catalog (options = `"{name}: {description[:200]}"`, context = last 1500 chars of the user prompt) and a JS extension shrinks the `<available_skills>` block to top-K (default 24, fail-open on error, keeps full catalog if top-K mass < 0.5). Dataset mined from session logs: 1,414 turns, 141 skills, 132 test turns. Honest metrics after fixing an eval leak: top-1 27.3% (majority-class ~29%, shuffled 19.7%), top-3 47.0%, recall@12 74.2%, recall@24 84.1%, recall@32 90.2%, ECE ~0.10, ~50-70 ms warm.

### Related (HN)
CUA-S1 (`trycua/cua`, 88 points) says it was "built from ideas and code in jevlike": 706k params, 2.8 MB, 7-9 ms local vs 260-280 ms per hosted Jev call. Other Jev-like projects seen on HN: `jaredpalmer/kev`, `daseinlabs/open-jev` (Gemma 3 4B), `r-ms/mini-jev`, `browser-use/jev-ultrafast`.

## 7. Timeline

| Date (UTC) | Event |
|---|---|
| 2026-09-15 | TypeSafe posts "Introducing System One Models and Jev" |
| 2026-09-16 10:25 | commit "Build Jev-like scorer starter"; repo created 10:26 |
| 2026-09-16 10:29 | "docs: centre the sum glyph in the architecture figures" |
| 2026-09-16 18:12 | "Add shared Doom and chess training examples" (last push) |
| 2026-09-16 18:49 | HN "Reverse-engineered Jev-like model": 166 points, 24 comments |
| 2026-09-16 20:04-21:24 | jevbetter created and finished |
| 2026-09-17 | issue #1 (shuffled control leak), PR #2 (cache pooled options) |
| 2026-09-18 | jevlike-esp32 (07:31-11:38), omo-jevlike-router (14:43-15:31) |
| 2026-09-19 | issue #3 (`weights_only=False`), PR #4 (reproducible synthetic), CUA-S1 on HN |
| 2026-09-20 | issue #5 (Laya/BERT baselines), JevLikeDiffusionGemma created |

Three commits, one author, no releases or tags, no maintainer replies on 3 issues and 2 PRs; the earliest fork is 2026-09-16T18:06Z, the latest 2026-09-20T17:50Z. HN reception treats the README as clearer than TypeSafe's own announcement ("still had no idea what it was" until reading it), while skepticism targets Jev itself ("nothing breakthrough", no paper, undisclosed weights).

## 8. Sources

- https://api.github.com/search/repositories?q=jevlike
- https://api.github.com/repos/vinnylarouge/jevlike
- https://api.github.com/repos/vinnylarouge/jevlike/commits?per_page=50
- https://api.github.com/repos/vinnylarouge/jevlike/git/trees/main?recursive=1
- https://api.github.com/repos/vinnylarouge/jevlike/issues?state=all&per_page=30
- https://api.github.com/repos/vinnylarouge/jevlike/pulls?state=all&per_page=30
- https://api.github.com/repos/vinnylarouge/jevlike/forks?per_page=30&sort=stargazers
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/README.md
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/AGENTS.md
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/pyproject.toml
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/jevlike/model.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/jevlike/data.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/jevlike/train.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/jevlike/eval.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/jevlike/predict.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/jevlike/vision.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/tests/test_smoke.py
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/examples/doom/README.md
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/examples/chess/README.md
- https://raw.githubusercontent.com/vinnylarouge/jevlike/main/examples/doom/environment.py (and other examples/doom/*.py, examples/chess/*.py)
- https://github.com/olanotolu/jevbetter and https://raw.githubusercontent.com/olanotolu/jevbetter/main/{README.md,jevbetter/model.py,jevbetter/featurize.py,jevbetter/train.py,jevbetter/benchmark.py,pyproject.toml}
- https://api.github.com/repos/olanotolu/jevbetter and .../commits?per_page=50
- https://github.com/david-cermak/jevlike-esp32 and https://raw.githubusercontent.com/david-cermak/jevlike-esp32/main/{README.md,esp32_demo/export_firmware.py,esp32_demo/fake_esp32.py,firmware/main/jevlike_scorer.c,firmware/main/main.c,firmware/sdkconfig.defaults}
- https://api.github.com/repos/david-cermak/jevlike-esp32 and .../commits?per_page=50
- https://github.com/pahndev/JevLikeDiffusionGemma and https://raw.githubusercontent.com/pahndev/JevLikeDiffusionGemma/main/{README.md,pyproject.toml,src/diffusiongemma_decisions/_mlx_backend.py,verification/model-smoke.json}
- https://api.github.com/repos/pahndev/JevLikeDiffusionGemma and .../commits?per_page=50
- https://github.com/islee23520/omo-jevlike-router and https://raw.githubusercontent.com/islee23520/omo-jevlike-router/main/{README.md,Concept.md,server/train_router.py,server/serve_router.py,extract/extract_dataset.py,extension/jevlike-router.js}
- https://api.github.com/repos/islee23520/omo-jevlike-router and .../commits?per_page=50
- https://hn.algolia.com/api/v1/search?query=jevlike
- https://hn.algolia.com/api/v1/search?query=typesafe%20jev
- https://typesafe.ai/blog/introducing-system-one-models-and-jev (cited by the README)


---

# Part 3: jev_pakkio


jev_pakkio (daseinlabs/open-jev) targets the same task on a different substrate. Instead of a small scorer trained from scratch, it runs `google/gemma-3-4b-it` in bf16 through mlx-lm on Apple silicon. The context is prefilled once, the KV cache is expanded across the option batch, and every option is scored in one padded forward pass with no decoding. The score is the log-probability of the option tokens given the context; a softmax over option scores gives a per-option probability, exactly the output shape of `jevlike-predict` and of Jev's `choice` question.

### What it replicates from jevlike

| Piece | jevlike | jev_pakkio |
|---|---|---|
| Data format | JSONL `{context, options, label}` | identical, shares `jevlike-data synthetic` output |
| CLI | `jevlike-data / train / eval / predict` | `jev_pakkio score / eval / bench / check / serve / features / train / eval-head` |
| Eval battery | top-1, top-3, 10-bin ECE, shuffled-context control | same four numbers |
| Scorer head | `AttentionHead` (cross-attention, one query per option) | ported to MLX in `jev_pakkio/head.py`, trained on frozen Gemma features |
| Encoder | byte embeddings from scratch, or frozen Qwen2.5-0.5B | frozen Gemma 3 4B hidden states (or zero-shot log-prob, no head at all) |

### Two routes

- **Route A: train a head on frozen Gemma features.** `jev_pakkio features` stops Gemma before the LM head and keeps every context token's final hidden state plus a masked mean of each option's tokens (options encoded independently, as in jevlike, so the head must do the matching). Training: AdamW at 5e-4 (2e-3 diverges; Gemma feature norms are around 115), weight decay 1e-4, grad clip 1.0, 8 epochs, batch 64, listwise cross-entropy.
- **Route B: Gemma zero-shot.** No training. Normalisation modes: `mean` (default), `sum`, and `pmi` (subtract the option's unconditional log-prob at the cost of one extra batched pass).

### Measured numbers (synthetic split 2000/400/400, 2026-09-17)

| Scorer | top-1 | top-3 | ECE |
|---|---|---|---|
| Head on frozen Gemma features | 0.970 | 1.000 | 0.027 |
| Shuffled-context control | 0.258 | 0.698 | 0.724 |
| jevlike tiny byte encoder + head | 0.998 | 1.000 | 0.008 |
| Gemma zero-shot, `--norm sum` | 1.000 | 1.000 | n/a |

Latency on an M5 Pro (64 GB), 202-token context, 8 options: 0.17 s median with the cached batched path versus 0.68 s re-encoding each option. The HTTP server answers a scoring request in about 90 ms. jevlike's tiny byte model answers in well under 10 ms, which is the gap a 4B backbone pays for zero-shot generality.

### System One compatibility

The FastAPI server exposes `/health`, `/score`, and `/v1/systemone`, the last one accepting TypeSafe's request contract (`state` plus a `questions` map with `choice`, `score`, and `noul` types) with optional bearer auth via `OPENJEV_API_KEY`. On the docs quick-start request, Jev returns `department.choice = technical` at p=0.84 with confidence 0.60, `frustration.score = 1.04`, and `is_urgent.noul = 0.999`. jev_pakkio with Gemma 3 4B returns technical at p=1.00 with confidence 1.00, frustration 2.00, and is_urgent 0.005. Same routing decision, but the zero-shot backbone is over-confident and disagrees on the judgement calls. jev_pakkio computes `confidence` as one minus normalised entropy; TypeSafe does not publish its formula.

### Where the repo currently stands

Uncommitted work on `main` touches the CLI, scorer, server, Makefile, and a new `finetune/` directory holding a LoRA pipeline (a chess fine-tune of Gemma, roughly 15 minutes to train). The Doom demo under `demo/doom/` mirrors jevlike's: ViZDoom headless, each frame described as one line of text, the action menu ranked by a single `/score` call or one System One `choice` question.
