#!/usr/bin/env python3
"""Replay the frozen publisher evaluator; do not overwrite the archived result."""
from pathlib import Path
import datetime
import os
import subprocess
import sys
root=Path(__file__).resolve().parent
if '--selfcheck' in sys.argv[1:]:
    raise SystemExit(subprocess.call([sys.executable,str(root/'tests/test_trident_evaluator009.py')],cwd=root))
image=Path('/opt/guoxin-trident009')
args=[str(image/'venv/bin/python'),str(root/'scripts/submission009/evaluate_trident.py'),'--data',str(image/'data')]
if 'trident'=='trident':args+=['--project',str(image/'project')]
extra=sys.argv[1:]
if '--output' not in extra:args+=['--output',str(root/('verify-replay-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))]
os.execv(args[0],args+extra)
