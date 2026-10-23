#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/trident_official_bridge_005.py" bench --task-dir "$TASK_DIR" "$@"
