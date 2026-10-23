#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
PROJECT_DIR="$(cd -- "$TASK_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
# Keep the guarded compatibility candidate on the default path. Baseline
# reproduction is explicit in run_baseline.sh and never calls this script.
EXTRA_ARGS_JSON="$("$PYTHON_BIN" - <<'PY'
import json
import os

extra = json.loads(os.environ["EXTRA_ARGS_JSON"])
if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
    raise SystemExit("ROBOBRAIN_EXTRA_ARGS_JSON must be a JSON array of strings")
reserved = {"--max-num-seqs", "--limit-mm-per-prompt", "--worker-extension-cls"}
if any(item.split("=", 1)[0] in reserved for item in extra):
    raise SystemExit("Use ROBOBRAIN_MAX_NUM_SEQS / ROBOBRAIN_PROFILE_WIDTH / "
                     "ROBOBRAIN_PROFILE_HEIGHT; the default worker extension is fixed. "
                     "Use run_baseline.sh for native framework comparisons.")

def positive_int(name):
    value = os.environ[name]
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise SystemExit(name + " must be a positive integer")
    return int(value)

profile = {"image": {"count": 2,
                      "width": positive_int("ROBOBRAIN_PROFILE_WIDTH"),
                      "height": positive_int("ROBOBRAIN_PROFILE_HEIGHT")}, "video": 0}
extra += ["--max-num-seqs", str(positive_int("ROBOBRAIN_MAX_NUM_SEQS")),
          "--limit-mm-per-prompt", json.dumps(profile, separators=(",", ":")),
          "--worker-extension-cls", "robobrain_vision_math_compat.VisionMathCompat"]
print(json.dumps(extra, separators=(",", ":")))
PY
)"
export EXTRA_ARGS_JSON
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_runtime.py" launch --task-dir "$TASK_DIR" "$@"
