#!/usr/bin/env bash
# Explicit native framework comparison. No compatibility worker is injected.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ROBOBRAIN_EXTRA_ARGS_JSON:-}" ]]; then
  export ROBOBRAIN_EXTRA_ARGS_JSON='["--enforce-eager","--max-num-seqs","4","--limit-mm-per-prompt","{\"image\":2,\"video\":0}"]'
fi
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_runtime.py" launch --task-dir "$TASK_DIR" "$@"
