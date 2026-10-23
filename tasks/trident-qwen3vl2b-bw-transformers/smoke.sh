#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/trident_embedding_smoke_004.py" --base-url "http://127.0.0.1:$TRIDENT_PORT" --model "$TRIDENT_SERVED_NAME" --image "$TASK_DIR/fixtures/red-rectangle.png" --log "$TASK_DIR/smoke.log" "$@"
