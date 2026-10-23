#!/usr/bin/env bash
set -euo pipefail
# The shared official bridge currently hardcodes vision checks; do not misroute T04.
printf '%s\n' 'T04 official evaluator integration is pending; no ASR accuracy evaluation started.' >&2
exit 2
