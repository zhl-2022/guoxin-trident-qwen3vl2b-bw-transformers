#!/usr/bin/env python3
"""Seal the pre-build field snapshot; later delivery closure stays external."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'docs/stage004-report-fields.json'
    data = json.loads(target.read_bytes())
    data.update(kind='verified_runtime_and_prebuild_delivery_field_snapshot',
        snapshot_utc=datetime.now(timezone.utc).isoformat(),
        snapshot_boundary='Local-stage and push fields describe this prebuild snapshot. Actual build, ZIP, commits and remote readback are recorded externally after freezing; this file is not rewritten inside frozen payloads.',
        final_platform_checkpoint='evidence/platform-checkpoint-20261023-004.json',
        official_accuracy_verified=False, official_performance_verified=False, contest_form_submitted=False)
    for row in data['reports']:
        row['report_sha256_at_draft'] = hashlib.sha256((ROOT / row['report_source']).read_bytes()).hexdigest()
        row['report_sha256_scope'] = 'Current frozen report source before repository build; legacy key name retained'
    with target.open('w', encoding='utf-8', newline='\n') as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
