#!/usr/bin/env bash
set -euo pipefail
# Never send an audio task to the shared BENCH_MODE=image six-load benchmark.
printf '%s\n' 'T04 audio benchmark materials/integration are pending; no benchmark started.' >&2
exit 2
