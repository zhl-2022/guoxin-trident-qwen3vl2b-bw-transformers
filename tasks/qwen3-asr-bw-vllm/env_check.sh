#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QWEN_ASR_PYTHON_BIN="${QWEN_ASR_PYTHON_BIN:-$TASK_DIR/runtime/audio-venv/bin/python}"
source "$TASK_DIR/env.sh"
"$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_runtime.py" env --task-dir "$TASK_DIR" "$@" | tee "$TASK_DIR/env.log"
