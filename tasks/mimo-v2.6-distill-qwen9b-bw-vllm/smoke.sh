#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/mimo_text_smoke.py" \
  --base-url "http://127.0.0.1:$PORT" --model "$SERVED_NAME" --log "$TASK_DIR/smoke.log" "$@"
