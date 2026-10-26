#!/usr/bin/env python3
"""Transparent shell verifier; no evaluation result synthesis."""
import os
from pathlib import Path
import sys

entry = Path(__file__).resolve().parent / 'tasks/trident-qwen3vl2b-bw-transformers/verify005.sh'
try:
    os.execvp('bash', ['bash', str(entry), *sys.argv[1:]])
except OSError as exc:
    print(f'Cannot execute canonical entry: {exc}', file=sys.stderr)
    raise SystemExit(2)
