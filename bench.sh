#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY_ROOT/env.sh"
# The frozen evaluator runs full accuracy plus both performance rounds.
exec "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/submission009/evaluate_trident.py" --data "$DATA_DIR" --project "$PROJECT" --output "${BENCH_OUTPUT:-$REPOSITORY_ROOT/bench-replay-$(date -u +%Y%m%dT%H%M%SZ)}" "$@"
