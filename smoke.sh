#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY_ROOT/env.sh"
exec bash "$TASK/smoke.sh" "$@"
