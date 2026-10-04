# jev_pakkio: one-pass option scoring with Gemma 3 4B via MLX. Native macOS only (Metal).
VENV   := .venv
BIN    := $(VENV)/bin
MODEL  ?= models/gemma-3-4b-it
HF_REPO ?= google/gemma-3-4b-it
HOST   ?= 127.0.0.1
PORT   ?= 8000
NORM   ?= mean
DATA   ?= data/synthetic/test.jsonl
API_KEY ?= local
SCENARIO ?= defend_the_center
API    ?= score
DATA_DIR ?= data/synthetic
FEATS  ?= runs/feats
HEAD   ?= runs/head.safetensors
LORA_DATA ?= data/synthetic/movies_hard2k
LORA_MODEL ?= google/gemma-4-E4B-it
LORA   ?= runs/lora
SEP    ?= \nChoice: 

.PHONY: help setup venv model serve health request doom systemone score check bench eval features train eval-head lora clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: venv model ## create the venv and download the model

# One backend extra per platform. mlx must not be installed off Apple silicon:
# it resolves on Linux without its libmlx.so, and transformers then dies
# importing it, which breaks the torch backend too.
venv: ## uv sync with the backend extra for this platform (mlx on macOS, torch elsewhere)
	uv sync $(if $(filter Darwin,$(shell uname -s)),--extra mlx --extra torch,--extra torch)

model: ## download $(HF_REPO) into $(MODEL) if missing
	@test -f $(MODEL)/config.json || hf download $(HF_REPO) --local-dir $(MODEL)

serve: ## run the HTTP server (loads the model once; POST /score, POST /v1/systemone, GET /health)
	$(BIN)/jev_pakkio serve --host $(HOST) --port $(PORT) --model $(MODEL)

health: ## curl the running server's health endpoint
	@curl -s $(HOST):$(PORT)/health; echo

request: ## example scoring request against the running server
	@curl -s $(HOST):$(PORT)/score -H 'content-type: application/json' -d '{"context": "Customer: my order arrived broken. Agent:", "options": [" I am sorry to hear that, I will send a replacement today.", " Please read our returns policy.", " Have you tried turning it off and on again?"], "norm": "$(NORM)"}'; echo

doom: ## play Doom in the terminal, the running server picks every action (SCENARIO=defend_the_center API=score)
	$(BIN)/python demo/doom/play.py --scenario $(SCENARIO) --api $(API) --url http://$(HOST):$(PORT) --api-key $(API_KEY)

systemone: ## TypeSafe quickstart example (choice + score + noul) against the running server
	@curl -s $(HOST):$(PORT)/v1/systemone -H 'Authorization: Bearer $(API_KEY)' -H 'Content-Type: application/json' -d @examples/systemone-quickstart.json; echo

score: ## one-off CLI scoring, e.g. make score CONTEXT="..." OPTIONS="--option a --option b"
	$(BIN)/jev_pakkio score --model $(MODEL) --norm $(NORM) --context "$(CONTEXT)" $(OPTIONS)

check: ## verify prefix-cached batched scores match naive re-encoding
	$(BIN)/jev_pakkio check --model $(MODEL)

bench: ## latency: cached+batched vs naive, 200-token context x 8 options
	$(BIN)/jev_pakkio bench --model $(MODEL)

eval: ## zero-shot top-k accuracy on jevlike-style JSONL ($(DATA))
	$(BIN)/jev_pakkio eval $(DATA) --model $(MODEL) --norm $(NORM)

features: ## cache frozen-Gemma features for $(DATA_DIR)/{train,validation,test}.jsonl into $(FEATS)/
	@for split in train validation test; do \
	  $(BIN)/jev_pakkio features $(DATA_DIR)/$$split.jsonl --out $(FEATS)/$$split.npz --model $(MODEL) --sep "$$(printf '$(SEP)')"; \
	done

train: ## train the attention head on $(FEATS)/train.npz, validate on $(FEATS)/validation.npz -> $(HEAD)
	$(BIN)/jev_pakkio train $(FEATS)/train.npz --validation $(FEATS)/validation.npz --out $(HEAD)

eval-head: ## top-k, ECE and shuffled-context control of $(HEAD) on $(FEATS)/test.npz
	$(BIN)/jev_pakkio eval-head $(HEAD) $(FEATS)/test.npz

lora: ## QLoRA-tune $(LORA_MODEL) as a scorer on $(LORA_DATA)/{train,validation,test}.jsonl -> $(LORA)/ (torch + CUDA)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(BIN)/jev_pakkio lora $(LORA_DATA)/train.jsonl \
	  --validation $(LORA_DATA)/validation.jsonl --test $(LORA_DATA)/test.jsonl \
	  --model $(LORA_MODEL) --epochs 2 --eval-every 500 --out $(LORA)

clean: ## remove caches and run artefacts (keeps the venv and model)
	rm -rf runs __pycache__ jev_pakkio/__pycache__

# --- chess next-move fine-tune (see finetune/README.md) ---
ADAPTER ?= adapters/chess-lora

chess-data: ## sample Lichess puzzles from the Hub into data/chess/{train,valid,test}.jsonl
	$(BIN)/python finetune/prepare_chess_data.py --puzzles 3000 --out data/chess

chess-train: ## LoRA-finetune $(MODEL) on data/chess with mlx_lm.lora -> $(ADAPTER)
	$(BIN)/python -m mlx_lm.lora --model $(MODEL) --train --data data/chess \
	  --fine-tune-type lora --num-layers 16 --batch-size 4 --iters 1000 \
	  --steps-per-report 50 --steps-per-eval 200 --val-batches 50 \
	  --learning-rate 1e-4 --max-seq-length 512 --mask-prompt --save-every 200 \
	  --adapter-path $(ADAPTER) --seed 0

chess-eval: ## rank legal moves on data/chess/test.jsonl, base model vs $(ADAPTER)
	$(BIN)/python finetune/eval_chess.py --model $(MODEL)
	$(BIN)/python finetune/eval_chess.py --model $(MODEL) --adapter $(ADAPTER)

chess-play: ## play chess against $(ADAPTER) in the terminal (PLAY=white|black)
	$(BIN)/python finetune/play_chess.py --model $(MODEL) --adapter $(ADAPTER) --play $(or $(PLAY),white)
