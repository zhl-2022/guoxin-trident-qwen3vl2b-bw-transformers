#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
# Fixed synthetic fixtures test the API path only; they are not accuracy data.
HAS_IMAGE=0
for ARG in "$@"; do
  case "$ARG" in --image|--image=*) HAS_IMAGE=1 ;; esac
done
DEFAULT_INPUTS=()
if [[ "$HAS_IMAGE" == 0 ]]; then
  DEFAULT_INPUTS=(--image "$TASK_DIR/../../fixtures/smoke-session004/recon_red.png"
                  --image "$TASK_DIR/../../fixtures/smoke-session004/recon_blue.png")
fi
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_smoke.py" \
  --base-url "http://127.0.0.1:$PORT" --model "$SERVED_NAME" \
  --log "$TASK_DIR/smoke.log" "${DEFAULT_INPUTS[@]}" "$@"
