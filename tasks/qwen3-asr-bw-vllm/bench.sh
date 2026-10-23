#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
# T04 requires an explicit external audio benchmark and output contract.
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_runtime.py" bench --task-dir "$TASK_DIR" "$@"
