#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
HAS_AUDIO=0
for ARG in "$@"; do
  case "$ARG" in --audio|--audio=*) HAS_AUDIO=1 ;; esac
done
DEFAULT_INPUTS=()
if [[ "$HAS_AUDIO" == 0 ]]; then
  # Publisher-derived short WAV, for API smoke only. No reference is submitted.
  DEFAULT_INPUTS=(--audio "$TASK_DIR/../../fixtures/smoke-session004/asr_zh.wav")
fi
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/qwen3_asr_smoke.py" \
  --task "$TASK_DIR/task.json" --base-url "http://127.0.0.1:$PORT" --model "$SERVED_NAME" \
  --log "$TASK_DIR/smoke.log" "${DEFAULT_INPUTS[@]}" "$@"
