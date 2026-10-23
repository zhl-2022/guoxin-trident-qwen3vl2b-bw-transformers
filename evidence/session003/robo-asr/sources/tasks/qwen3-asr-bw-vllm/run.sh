#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$TASK_DIR/../.." && pwd)"
# The verified TF5.5 image requires this explicit task-local compatibility plugin.
export QWEN_ASR_PYTHON_BIN="${QWEN_ASR_PYTHON_BIN:-$TASK_DIR/runtime/audio-venv/bin/python}"
source "$TASK_DIR/env.sh"
export PYTHONPATH="$PROJECT_DIR/scripts:$TASK_DIR/compat${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS="hcu,qwen_asr_tf55_compat"
DRY_RUN=0
for ARG in "$@"; do
  if [[ "$ARG" == "--dry-run" ]]; then DRY_RUN=1; fi
done
if [[ "$DRY_RUN" == 1 ]]; then
  # Expand launch arguments locally before a Linux venv exists. No import of vLLM.
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "Prepared Python absent: $PYTHON_BIN; using base Python for argument preview only." >&2
    PYTHON_BIN="${SCNET_BASE_PYTHON:-python3}"
  fi
else
  VENV_DIR="$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
  "${SCNET_BASE_PYTHON:-python3}" -B "$PROJECT_DIR/scripts/prepare_qwen_asr_task.py" \
    --task-dir "$TASK_DIR" --check-existing "$VENV_DIR"
fi
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_runtime.py" launch --task-dir "$TASK_DIR" "$@"
