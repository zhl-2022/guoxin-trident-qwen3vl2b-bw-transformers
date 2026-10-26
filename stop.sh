#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY_ROOT/env.sh"
if [[ $# -eq 0 ]]; then echo "Supply --run-id from launch output" >&2; exit 2; fi
exec "$PYTHON_BIN" "$PROJECT/scripts/trident_embedding_runtime_004.py" stop --task-dir "$TASK" "$@"
