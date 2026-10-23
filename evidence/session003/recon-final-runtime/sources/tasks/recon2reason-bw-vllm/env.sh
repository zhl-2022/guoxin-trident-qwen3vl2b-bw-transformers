#!/usr/bin/env bash
# Task-scoped overrides prevent another sourced model's generic variables leaking in.
# Use RECON2REASON_* in the invoking shell. Never put credentials here.
export MODEL_ID="BAAI/Recon2Reason-Reasoning-4B"
export MODEL_PATH="${RECON2REASON_MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B}"
export SERVED_NAME="${RECON2REASON_SERVED_NAME:-recon2reason-reasoning-4b}"
# This display name is known; it is not a complete registry address/tag.
export IMAGE="${RECON2REASON_IMAGE:-}"
export IMAGE_CANDIDATE_NAME="${RECON2REASON_IMAGE_CANDIDATE_NAME:-jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423}"
export FRAMEWORK="vllm"
export CARD="${RECON2REASON_CARD:-BW}"
export PRECISION="bf16"
export PORT="${RECON2REASON_PORT:-30000}"
export TP="${RECON2REASON_TP:-1}"
export MAX_LEN="${RECON2REASON_MAX_LEN:-32768}"
export MEM_FRAC="${RECON2REASON_MEM_FRAC:-0.85}"
export LOAD_TIMEOUT="${RECON2REASON_LOAD_TIMEOUT:-900}"
export MAX_RUNTIME_SECONDS="${RECON2REASON_MAX_RUNTIME_SECONDS:-4500}"
export STOP_GRACE_SECONDS="${RECON2REASON_STOP_GRACE_SECONDS:-30}"
# Match vllm-state-001.json. A replacement array describes a separate experiment.
if [[ -v RECON2REASON_EXTRA_ARGS_JSON ]]; then
  export EXTRA_ARGS_JSON="$RECON2REASON_EXTRA_ARGS_JSON"
else
  export EXTRA_ARGS_JSON='["--enforce-eager","--max-num-seqs","4","--limit-mm-per-prompt","{\"image\":2,\"video\":0}"]'
fi
export PYTHON_BIN="${RECON2REASON_PYTHON_BIN:-python3}"
# Do not inherit another task's evaluator, baseline, data or image resolution.
export OFFICIAL_VERIFY_SCRIPT="${RECON2REASON_OFFICIAL_VERIFY_SCRIPT:-}"
export OFFICIAL_BENCH_SCRIPT="${RECON2REASON_OFFICIAL_BENCH_SCRIPT:-}"
export OFFICIAL_BASELINE="${RECON2REASON_OFFICIAL_BASELINE:-}"
export OFFICIAL_DATA_DIR="${RECON2REASON_OFFICIAL_DATA_DIR:-}"
export IMAGE_RES="${RECON2REASON_IMAGE_RES:-}"
export OFFICIAL_VERIFY_ARGS_JSON="${RECON2REASON_OFFICIAL_VERIFY_ARGS_JSON:-[]}"
export OFFICIAL_BENCH_ARGS_JSON="${RECON2REASON_OFFICIAL_BENCH_ARGS_JSON:-[]}"
export OFFICIAL_SELFCHECK_ARGS_JSON="${RECON2REASON_OFFICIAL_SELFCHECK_ARGS_JSON:-[\"--selfcheck\"]}"
export OFFICIAL_SELFCHECK_TIMEOUT_SECONDS="${RECON2REASON_OFFICIAL_SELFCHECK_TIMEOUT_SECONDS:-120}"
export OFFICIAL_TIMEOUT_SECONDS="${RECON2REASON_OFFICIAL_TIMEOUT_SECONDS:-7200}"
# Optional explicit output contracts; absent variables retain the template paths.
unset OFFICIAL_VERIFY_OUTPUTS_JSON OFFICIAL_BENCH_OUTPUTS_JSON
if [[ -v RECON2REASON_OFFICIAL_VERIFY_OUTPUTS_JSON ]]; then
  export OFFICIAL_VERIFY_OUTPUTS_JSON="$RECON2REASON_OFFICIAL_VERIFY_OUTPUTS_JSON"
fi
if [[ -v RECON2REASON_OFFICIAL_BENCH_OUTPUTS_JSON ]]; then
  export OFFICIAL_BENCH_OUTPUTS_JSON="$RECON2REASON_OFFICIAL_BENCH_OUTPUTS_JSON"
fi
