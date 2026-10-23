#!/usr/bin/env python3
"""Wait for this run to settle, verify owned cleanup, then export. No GPU calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

PROJECT = Path('/root/private_data/guoxin-hackathon-20261023-session004')
EVIDENCE = PROJECT / 'evidence/session004'
DEADLINE = datetime(2026, 10, 23, 12, 26, tzinfo=timezone.utc).timestamp()


def token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def main():
    rid = (EVIDENCE / 'recon-matrix-run-id.txt').read_text().strip()
    runtime = PROJECT / 'tasks/recon2reason-bw-vllm/runtime'
    while time.time() < DEADLINE:
        state = json.loads((runtime / 'state.json').read_bytes())
        if state['run_id'] != rid:
            raise ValueError('Run changed; no cleanup/export attempted')
        if state['status'] in ('stopped', 'expired', 'failed'):
            break
        time.sleep(3)
    else:
        raise TimeoutError('Not settled before export deadline; manual review needed')
    for _ in range(15):
        if (EVIDENCE / 'recon-server-metrics-001/collector-final.json').is_file():
            break
        time.sleep(1)
    check = {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': rid,
             'state': state, 'observed_process_tokens': {}, 'no_inference_performed': True}
    live = False
    for kind in ('supervisor', 'server'):
        observed = token(state.get(kind + '_pid'))
        check['observed_process_tokens'][kind] = observed
        live |= observed is not None and observed == str(state.get(kind + '_token'))
    with socket.socket() as connection:
        connection.settimeout(1)
        port_open = connection.connect_ex(('127.0.0.1', 30000)) == 0
    check.update(owned_processes_stopped_verified=not live, port_30000_open=port_open,
                 cleanup_verified=not live and not port_open)
    destination = EVIDENCE / 'recon-final-cleanup-001.json'
    with destination.open('x') as output:
        json.dump(check, output, ensure_ascii=False, indent=2)
    if not check['cleanup_verified']:
        raise ValueError('Owned process/port cleanup not closed')
    scripts = ['/root/export_session004_results_004b.py', '/root/t02_sglang_bench_004g.py',
               '/root/repair_sglang_client_004e.py', '/root/collect_vllm_metrics_004.py',
               '/root/private_data/guoxin-candidate-tools-20261023/scripts/trident_vllm_pooling_probe_20261023.py',
               str(Path(__file__).absolute())]
    args = ['/usr/bin/python', '-B', scripts[0], '--output', '/root/hg-session004-results-final-001.zip']
    for name in scripts:
        args += ['--script-file', name]
    completed = subprocess.run(args, check=False, text=True, capture_output=True)
    log = EVIDENCE / 'export-final-001-result.txt'
    with log.open('x') as output:
        output.write(completed.stdout + completed.stderr)
    print(completed.stdout + completed.stderr, flush=True)
    if completed.returncode:
        raise RuntimeError(f'Export exit {completed.returncode}')
    archive = Path('/root/hg-session004-results-final-001.zip')
    print('LOCAL_UI_DOWNLOAD_EXPECTED_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
