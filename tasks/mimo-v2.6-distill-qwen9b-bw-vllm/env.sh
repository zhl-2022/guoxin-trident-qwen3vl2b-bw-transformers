#!/usr/bin/env bash
# MiMo-only overrides. Preserve checkpoint thinking/sampling defaults.
export MODEL_ID="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
export MODEL_PATH="${MIMO_MODEL_PATH:-/root/private_data/guoxin-candidates/models/mimo9b}"
export SERVED_NAME="${MIMO_SERVED_NAME:-mimo-v2.6-distill-qwen-9b}"
export FRAMEWORK="vllm"
export CARD="${MIMO_CARD:-BW}"
export PRECISION="bf16"
export IMAGE="${MIMO_IMAGE:-}"
export IMAGE_CANDIDATE_NAME="${MIMO_IMAGE_CANDIDATE_NAME:-jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423}"
export PYTHON_BIN="${MIMO_PYTHON_BIN:-/usr/bin/python3}"
export PORT="${MIMO_PORT:-30109}"
export TP="${MIMO_TP:-1}"
export MAX_LEN="${MIMO_MAX_LEN:-32768}"
export MEM_FRAC="${MIMO_MEM_FRAC:-0.80}"
export LOAD_TIMEOUT="${MIMO_LOAD_TIMEOUT:-900}"
export MAX_RUNTIME_SECONDS="${MIMO_MAX_RUNTIME_SECONDS:-1800}"
export STOP_GRACE_SECONDS="${MIMO_STOP_GRACE_SECONDS:-30}"
# T01 text service: reproduce session004 successful image=0/video=0 profile.
# This limits accepted modalities and skips their profiling; checkpoint intact.
if [[ -n "${MIMO_EXTRA_ARGS_JSON:-}" ]]; then
  export EXTRA_ARGS_JSON="$MIMO_EXTRA_ARGS_JSON"
else
  export EXTRA_ARGS_JSON='["--enforce-eager","--max-num-seqs","4","--max-num-batched-tokens","2048","--limit-mm-per-prompt","{\"image\":0,\"video\":0}"]'
fi
export OFFICIAL_VERIFY_SCRIPT="${MIMO_OFFICIAL_VERIFY_SCRIPT:-}"
export OFFICIAL_BENCH_SCRIPT="${MIMO_OFFICIAL_BENCH_SCRIPT:-}"
export OFFICIAL_BASELINE="${MIMO_OFFICIAL_BASELINE:-}"
export OFFICIAL_DATA_DIR="${MIMO_OFFICIAL_DATA_DIR:-}"
export OFFICIAL_VERIFY_ARGS_JSON="${MIMO_OFFICIAL_VERIFY_ARGS_JSON:-[]}"
export OFFICIAL_BENCH_ARGS_JSON="${MIMO_OFFICIAL_BENCH_ARGS_JSON:-[]}"
export OFFICIAL_SELFCHECK_ARGS_JSON="${MIMO_OFFICIAL_SELFCHECK_ARGS_JSON:-[\"--selfcheck\"]}"
export OFFICIAL_SELFCHECK_TIMEOUT_SECONDS="${MIMO_OFFICIAL_SELFCHECK_TIMEOUT_SECONDS:-120}"
export OFFICIAL_TIMEOUT_SECONDS="${MIMO_OFFICIAL_TIMEOUT_SECONDS:-7200}"
unset IMAGE_RES BENCH_MODE OFFICIAL_VERIFY_OUTPUTS_JSON OFFICIAL_BENCH_OUTPUTS_JSON
if [[ -v MIMO_OFFICIAL_VERIFY_OUTPUTS_JSON ]]; then
  export OFFICIAL_VERIFY_OUTPUTS_JSON="$MIMO_OFFICIAL_VERIFY_OUTPUTS_JSON"
fi
if [[ -v MIMO_OFFICIAL_BENCH_OUTPUTS_JSON ]]; then
  export OFFICIAL_BENCH_OUTPUTS_JSON="$MIMO_OFFICIAL_BENCH_OUTPUTS_JSON"
fi
