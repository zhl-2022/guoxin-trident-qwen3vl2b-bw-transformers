#!/usr/bin/env bash
# vLLM embedding configuration matches the session004 offline GPU pilot.
export MODEL_ID="openbmb/Trident-Qwen3VL-2B"
export FRAMEWORK="vllm"
export PRECISION="bf16"
export TRIDENT_MODEL_PATH="${TRIDENT_MODEL_PATH:-/root/private_data/guoxin-candidates/models/trident}"
export TRIDENT_SERVED_NAME="${TRIDENT_SERVED_NAME:-trident-qwen3vl-2b}"
export TRIDENT_PORT="${TRIDENT_PORT:-30119}"
export TRIDENT_MEM_FRAC="${TRIDENT_MEM_FRAC:-0.45}"
export TRIDENT_LOAD_TIMEOUT="${TRIDENT_LOAD_TIMEOUT:-600}"
export TRIDENT_MAX_RUNTIME_SECONDS="${TRIDENT_MAX_RUNTIME_SECONDS:-900}"
export PYTHON_BIN="${TRIDENT_PYTHON_BIN:-/root/private_data/guoxin-candidates/envs/trident-001/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
