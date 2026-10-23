import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

args = argparse.ArgumentParser()
args.add_argument('--run', action='store_true')
run = args.parse_args().run
root = Path('/root/private_data/guoxin-hackathon-20261020-session003')
original = root / 'evidence/session003/recon-matrix/run.json'
d = json.loads(original.read_text())
missing = [r['name'] for r in d['rows'] if r.get('status') != 'complete']
driver = root / 'scripts/t02_development_bench.py'
assert hashlib.sha256(driver.read_bytes()).hexdigest() == 'bc09711bb58fb5a7ae4676a840956a4da1caf49f6e6034b8015c5dd9a5bae14e'
hard_end = datetime.datetime(2026, 10, 19, 18, 57, 40, tzinfo=datetime.timezone.utc)
remaining = min(560, int((hard_end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
output = root / 'evidence/session003/recon-matrix-retry'
plan = {'original_status': d.get('status'), 'missing_rows': missing,
        'retry_timeout_seconds': remaining, 'output_dir': str(output),
        'scope': 'Independent development retry; original run remains unchanged', 'run_requested': run}
print(json.dumps(plan))
if not run:
    raise SystemExit(0)
if d.get('status') == 'complete':
    print('NO_RETRY_NEEDED')
    raise SystemExit(0)
assert d.get('status') == 'incomplete_or_failed', 'Original driver is not finished'
assert missing == ['round2_16384_1024_c64'], 'Only the final row can be retried within this bounded window'
assert remaining >= 500, 'Insufficient remaining budget; preserve partial results instead'
assert not output.exists(), 'Retry output already exists'
old_process = Path('/proc/5801/cmdline')
assert not old_process.exists() or not old_process.read_bytes(), 'Original driver still active'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:30000/metrics', timeout=3) as response:
    metrics = response.read().decode()
active = [float(line.split()[-1]) for line in metrics.splitlines()
          if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
assert len(active) >= 2 and not any(active), 'Wait for original requests to finish/abort before retry'
prefix = d['rows'][0]['command']
prefix = prefix[:prefix.index('bench')]
argv = [sys.executable, '-B', str(driver), '--run', '--model-path', d['model_path'],
        '--tokenizer-path', d['tokenizer_path'], '--served-model-name', d['served_model_name'],
        '--image-size', str(d['image_size'][0]), '--vllm-command-json', json.dumps(prefix),
        '--output-dir', str(output), '--load', '16384:64', '--round', '2',
        '--timeout-seconds', str(remaining), '--total-timeout-seconds', str(remaining)]
log = Path('/root/hg-recon-matrix-retry-003.txt')
with log.open('xb') as stream:
    process = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
record = dict(plan, argv=argv, pid=process.pid,
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              original_report_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
              driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest())
record_path = root / 'evidence/session003/recon-matrix-retry-launch.json'
with record_path.open('x') as stream:
    json.dump(record, stream, indent=2)
print('RETRY_STARTED', process.pid)
