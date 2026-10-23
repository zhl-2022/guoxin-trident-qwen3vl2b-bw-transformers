#!/usr/bin/env python3
"""Record UI observations already captured by the operator; no platform calls."""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / 'evidence/session004/screenshots'


def evidence(name):
    path = SCREEN / name
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'captured_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def main():
    output = ROOT / 'evidence/platform-checkpoint-20261023-004.json'
    stopped = evidence('all-notebooks-stopped-20261023.png')
    record = {'schema_version': 1, 'session': 'session004',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'Fresh SCNet visible UI; screenshot timestamps are capture times, not backend stop timestamps',
        'all_instances_stopped': True, 'all_stopped_evidence': stopped,
        'intermediate_platform': evidence('B-stopped-A-running-20261023.png'),
        'automatic_stop_configuration': evidence('notebooks-autooff-20261023.png'),
        'instances': [
            {'id': '2112186448839770114', 'scope': 'Recon', 'state': 'stopped',
             'allocated_start_observed_utc': '2026-10-23T09:38:49+00:00',
             'configured_auto_off_utc': '2026-10-23T12:30:00+00:00',
             'manual_stop_confirmed': True, 'save_environment_selected': True,
             'observed_stopped_by_utc': stopped['captured_utc'], 'round_budget_cardhours': 3},
            {'id': '2112237945233408001', 'scope': 'Robo/ASR/MiMo/Forced/Trident', 'state': 'stopped',
             'allocated_start_observed_utc': '2026-10-23T09:36:59+00:00',
             'configured_auto_off_utc': '2026-10-23T11:35:00+00:00',
             'manual_stop_confirmed': False, 'stop_method': 'platform_automatic_deadline',
             'observed_stopped_by_utc': evidence('B-stopped-A-running-20261023.png')['captured_utc'],
             'round_budget_cardhours': 2},
            {'id': '2097876209970565121', 'scope': 'historical reference', 'state': 'stopped',
             'started_in_session004': False}],
        'resource': {'region': '华中一区【A区】', 'group': '113 / hx1hgbwnormal',
                     'card': 'BW 64GB', 'cards_per_active_instance': 1, 'cpu_cores': 15, 'memory_gb': 59},
        'image': {'id': '2071293281204580354', 'version_id': '2082674417745739777',
                  'tag': 'jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423',
                  'full_actual_registry_reference': None,
                  'access': 'Account image is selectable in SCNet Notebook; reviewer access not independently verified'},
        'balance': {'before_visible_cardhours': '492.6924999999',
                    'after_visible_cardhours': '488.0238888888',
                    'visible_delta_cardhours': str(Decimal('492.6924999999')-Decimal('488.0238888888')),
                    'after_evidence': evidence('remaining-cardhours-20261023.png'),
                    'final_settled_bill': False,
                    'read_only_start_dialog_cancelled': True, 'notebook_restarted_to_read_balance': False},
        'recharged_or_purchased': False, 'contest_form_submitted': False,
        'scope_boundary': 'Platform stop observation is separate from service-process cleanup and official acceptance'}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
