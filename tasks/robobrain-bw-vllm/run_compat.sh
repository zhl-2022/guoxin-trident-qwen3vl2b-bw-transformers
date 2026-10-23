#!/usr/bin/env bash
# Compatibility alias; run.sh is the single default candidate entry point.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TASK_DIR/run.sh" "$@"
