#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
"$PYTHON_BIN" "$TASK_DIR/../../scripts/trident_embedding_runtime_004.py" env --task-dir "$TASK_DIR" "$@" | tee "$TASK_DIR/env.log"
