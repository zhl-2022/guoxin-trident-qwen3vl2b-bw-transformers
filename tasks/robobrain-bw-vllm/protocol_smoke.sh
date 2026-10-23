#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/robobrain_protocol_smoke.py" \
  --task "$TASK_DIR/task.json" --base-url "http://127.0.0.1:$PORT" --model "$SERVED_NAME" "$@"
