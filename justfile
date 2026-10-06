# jev_pakkio: one-pass option scoring with a local Gemma 4 via MLX (Apple silicon)
# or PyTorch (NVIDIA/CPU). `just run` starts the HTTP server, which is the
# usual entry point; everything else is a helper around it.

set dotenv-load := false

venv     := ".venv"
bin      := venv / "bin"
host     := "127.0.0.1"
port     := "8000"
model    := env_var_or_default("MODEL", "auto")
quantize := env_var_or_default("QUANTIZE", "4bit")
device   := env_var_or_default("DEVICE", "auto")
backend  := env_var_or_default("BACKEND", "auto")
norm     := env_var_or_default("NORM", "mean")
data     := env_var_or_default("DATA", "data/synthetic/test.jsonl")
data_dir := env_var_or_default("DATA_DIR", "data/synthetic")
feats    := env_var_or_default("FEATS", "runs/feats")
head     := env_var_or_default("HEAD", "runs/head.safetensors")
lora_data := env_var_or_default("LORA_DATA", "data/synthetic/movies_hard2k")
lora_model := env_var_or_default("LORA_MODEL", "google/gemma-4-E2B-it")
lora     := env_var_or_default("LORA", "runs/lora")
adapter  := env_var_or_default("ADAPTER", "adapters/chess-lora")
scenario := env_var_or_default("SCENARIO", "defend_the_center")
api      := env_var_or_default("API", "score")
play     := env_var_or_default("PLAY", "white")
mcp_port := env_var_or_default("MCP_PORT", "5001")

# List the recipes.
default:
    @just --list

# Create the venv (uv sync) and install the project.
setup:
    uv sync {{ if os() == "macos" { "--extra mlx --extra torch" } else { "--extra torch" } }}

# Start the HTTP server: POST /score, POST /v1/systemone, GET /health.
# Loads the model once. MODEL=auto (the default) picks by VRAM: Gemma 4 E2B on
# an 8GB card, Qwen 2B below that. Overrides: HOST, PORT, MODEL, QUANTIZE, DEVICE.
run:
    #!/usr/bin/env bash
    set -euo pipefail
    model="{{ model }}"
    if [ "$model" = "auto" ]; then
      model=$({{ bin }}/python -c "from jev_pakkio.scorer_torch import auto_model; print(auto_model())")
      echo "auto-selected model: $model" >&2
    fi
    exec {{ bin }}/jev_pakkio serve --backend {{ backend }} --quantize {{ quantize }} \
      --device {{ device }} --host {{ host }} --port {{ port }} --model "$model"

# The MCP server (stdio): classify, rate, noul, jev, ask, compare, list_engines,
# memory_*, train_lora, status. Engines load on first use. Same MODEL=auto VRAM
# sizing as `run`. `just mcp --http` serves streamable-http on MCP_PORT instead,
# with Bearer "pakkio 62" (override: JEV_MCP_PASSWORD) as the port's token.
mcp *args:
    #!/usr/bin/env bash
    set -euo pipefail
    model="{{ model }}"
    if [ "$model" = "auto" ]; then
      model=$({{ bin }}/python -c "from jev_pakkio.scorer_torch import auto_model; print(auto_model())")
      echo "auto-selected model: $model" >&2
    fi
    exec {{ bin }}/jev_pakkio mcp --backend {{ backend }} --quantize {{ quantize }} \
      --device {{ device }} --host {{ host }} --port {{ mcp_port }} --model "$model" {{ args }}

# Curl the running server's health endpoint.
health:
    @curl -s {{ host }}:{{ port }}/health; echo

# Example scoring request against the running server.
request:
    @curl -s {{ host }}:{{ port }}/score -H 'content-type: application/json' -d '{"context": "Customer: my order arrived broken. Agent:", "options": [" I am sorry to hear that, I will send a replacement today.", " Please read our returns policy.", " Have you tried turning it off and on again?"], "norm": "{{ norm }}"}'; echo

# TypeSafe System One quickstart (choice + score + noul) against the running server.
systemone:
    @curl -s {{ host }}:{{ port }}/v1/systemone -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d @examples/systemone-quickstart.json; echo

# One-off CLI scoring, no server:
#   just score "The capital of France is" --option " Paris" --option " Berlin"
score context *options:
    {{ bin }}/jev_pakkio score --backend {{ backend }} --quantize {{ quantize }} --device {{ device }} --model {{ model }} --norm {{ norm }} --context "{{ context }}" {{ options }}

# Verify the prefix-cached batched path matches naive re-encoding.
check:
    {{ bin }}/jev_pakkio check --backend {{ backend }} --quantize {{ quantize }} --device {{ device }} --model {{ model }}

# Latency: cached+batched vs naive, 200-token context x 8 options.
bench:
    {{ bin }}/jev_pakkio bench --backend {{ backend }} --quantize {{ quantize }} --device {{ device }} --model {{ model }}

# Zero-shot top-k accuracy on jevlike-style JSONL (DATA=...).
eval:
    {{ bin }}/jev_pakkio eval {{ data }} --backend {{ backend }} --quantize {{ quantize }} --device {{ device }} --model {{ model }} --norm {{ norm }}

# Play Doom in the terminal; the running server picks every action.
doom:
    {{ bin }}/python demo/doom/play.py --scenario {{ scenario }} --api {{ api }} --url http://{{ host }}:{{ port }} --api-key local

# --- Route A: frozen features + attention head ---

# Cache frozen-Gemma features for {train,validation,test}.jsonl into FEATS/.
features:
    @for split in train validation test; do \
      {{ bin }}/jev_pakkio features {{ data_dir }}/$split.jsonl --out {{ feats }}/$split.npz --backend {{ backend }} --quantize {{ quantize }} --device {{ device }} --model {{ model }} --sep $'\nChoice: '; \
    done

# Train the attention head on cached features -> HEAD.
train:
    {{ bin }}/jev_pakkio train {{ feats }}/train.npz --validation {{ feats }}/validation.npz --out {{ head }}

# Top-k, ECE and the shuffled-context control for HEAD.
eval-head:
    {{ bin }}/jev_pakkio eval-head {{ head }} {{ feats }}/test.npz

# --- Route C: QLoRA fine-tuning (torch + CUDA) ---

# QLoRA-tune the scorer on {train,validation,test}.jsonl -> LORA/.
lora:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{ bin }}/jev_pakkio lora {{ lora_data }}/train.jsonl \
      --validation {{ lora_data }}/validation.jsonl --test {{ lora_data }}/test.jsonl \
      --model {{ lora_model }} --epochs 2 --eval-every 500 --out {{ lora }}

# Serve one or more adapters: just serve-lora "claims=path/to/claims,movies=path/to/movies"
# Each NAME=PATH becomes its own --lora flag.
serve-lora names:
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    IFS=',' read -ra parts <<< "{{ names }}"
    for p in "${parts[@]}"; do
      [ -n "$p" ] && args+=(--lora "$p")
    done
    {{ bin }}/jev_pakkio serve --backend torch --quantize {{ quantize }} --device {{ device }} \
      --model {{ lora_model }} --host {{ host }} --port {{ port }} "${args[@]}"

# --- chess next-move fine-tune (see finetune/README.md) ---

# Sample Lichess puzzles into data/chess/{train,valid,test}.jsonl.
chess-data:
    {{ bin }}/python finetune/prepare_chess_data.py --puzzles 3000 --out data/chess

# Rank legal moves on data/chess/test.jsonl, base model vs ADAPTER.
chess-eval:
    {{ bin }}/python finetune/eval_chess.py --model {{ model }}
    {{ bin }}/python finetune/eval_chess.py --model {{ model }} --adapter {{ adapter }}

# Play chess against ADAPTER in the terminal.
chess-play:
    {{ bin }}/python finetune/play_chess.py --model {{ model }} --adapter {{ adapter }} --play {{ play }}

# --- language identification (20 languages, papluca/language-identification) ---

# Sample data/langid/{train,validation,test}.jsonl (20 shuffled language-name options per row).
langid-data:
    {{ bin }}/python finetune/prepare_langid_data.py --out data/langid

# QLoRA-tune language-id -> runs/lora-langid. Chat format; compares against zero-shot on test.
langid-lora:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{ bin }}/jev_pakkio lora data/langid/train.jsonl \
      --validation data/langid/validation.jsonl --test data/langid/test.jsonl \
      --model {{ lora_model }} --chat --epochs 1 --eval-every 500 --out runs/lora-langid

# Remove caches and run artefacts (keeps the venv and downloaded models).
clean:
    rm -rf runs __pycache__ jev_pakkio/__pycache__
# Print the auto-selected model for this machine's VRAM.
model-info:
    @{{ bin }}/python -c "from jev_pakkio.scorer_torch import auto_model; import torch; vram=(torch.cuda.get_device_properties(0).total_memory//(1024**3)) if torch.cuda.is_available() else 0; print(f'vram={vram}GiB model={auto_model()}')"

# Compare jev_pakkio against TypeSafe's hosted Jev on the same rows.
# Reads TYPESAFE_API_KEY from ../.env (never printed). Pass a JSONL of
# {state, options, label} rows, or omit it for a small built-in set.
compare data="":
    #!/usr/bin/env bash
    set -euo pipefail
    model=$({{ bin }}/python -c "from jev_pakkio.scorer_torch import auto_model; print(auto_model())")
    if [ -n "{{ data }}" ]; then
      {{ bin }}/python compare_jev.py "{{ data }}" --model "$model" --out runs/compare-jev.json
    else
      {{ bin }}/python compare_jev.py --model "$model" --out runs/compare-jev.json
    fi
