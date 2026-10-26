#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY_ROOT/env.sh"
"$PYTHON_BIN" -c 'import json,torch,importlib.metadata as m; print(json.dumps({"torch":torch.__version__,"HIP":torch.version.hip,"vllm":m.version("vllm"),"transformers":m.version("transformers"),"GPU_available":torch.cuda.is_available(),"Device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))' | tee "${ENV_LOG:-$REPOSITORY_ROOT/env-replay.log}"
