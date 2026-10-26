#!/usr/bin/env bash
_TASK_REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_TASK_REPOSITORY_ROOT/tasks/trident-qwen3vl2b-bw-transformers/env.sh"
unset _TASK_REPOSITORY_ROOT
