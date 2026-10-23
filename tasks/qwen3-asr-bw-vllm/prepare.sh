#!/usr/bin/env bash
# Default is a local plan, with no filesystem writes, downloads or GPU checks.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCNET_BASE_PYTHON:-python3}" -B "$TASK_DIR/../../scripts/prepare_qwen_asr_task.py" --task-dir "$TASK_DIR" "$@"
