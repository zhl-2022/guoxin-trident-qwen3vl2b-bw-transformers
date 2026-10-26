#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY_ROOT/env.sh"
exec "$PYTHON_BIN" "$PROJECT/scripts/trident_embedding_runtime_004.py" launch --task-dir "$TASK" "$@"
