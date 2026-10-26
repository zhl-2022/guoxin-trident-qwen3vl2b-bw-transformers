"""Recompute all downloaded ChartQA ranks and verify original byte identities."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import statistics
import tarfile
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/session009/trident-chartqa'
RESULT = ROOT / 'evidence/session009/trident002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def main():
    manifest = read(ROOT / 'evidence/session009/image-preparation002/opt/guoxin-submission009/IMAGE-CONTENTS.json')
    group = next(g for g in manifest['copied_groups'] if g['label'] == 'trident002')
    expected = {f['path']: f for f in group['files']}
    actual = {p.relative_to(RESULT).as_posix(): p for p in RESULT.rglob('*') if p.is_file()}
    assert set(actual) == set(expected)
    for name, path in actual.items():
        assert path.stat().st_size == expected[name]['bytes'] and sha(path) == expected[name]['sha256'], name
    queries = rows(DATA / 'queries.jsonl')
    documents = rows(DATA / 'origin_corpus.jsonl')
    qrels = {r['query_id']: set(r['relevant_docs']) for r in rows(DATA / 'qrels.jsonl')}
    ev = RESULT / 'evaluation'
    protocol = read(ev / 'protocol.json')
    for name, digest in protocol['data_sha256'].items():
        assert sha(DATA / name) == digest
    original = rows(ev / 'raw-embeddings.jsonl')
    responses = {(r['role'], r['id']): r for r in original}
    assert len(original) == len(responses) == 2759
    jobs = [('query', r) for r in queries] + [('doc', r) for r in documents]
    assert set(responses) == {(role, r['id']) for role, r in jobs}
    with tarfile.open(DATA / 'images.tar.gz') as archive:
        image_hashes = {m.name.removeprefix('./'): hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                        for m in archive.getmembers() if m.isfile()}
    matrices = []
    norm_errors = []
    for role, items, file in [('query', queries, 'queries.npy'), ('doc', documents, 'documents.npy')]:
        vectors = []
        for item in items:
            row = responses[(role, item['id'])]
            assert row['status'] == 'ok'
            vector = np.asarray(row['raw_response']['data'][0]['embedding'], dtype=np.float32)
            assert vector.shape == (2048,) and np.isfinite(vector).all()
            norm_errors.append(abs(float(np.linalg.norm(vector)) - 1))
            if item.get('image'):
                assert row['source_image_sha256'] == image_hashes[item['image']]
            vectors.append(vector)
        matrix = np.asarray(vectors, dtype=np.float32)
        assert np.array_equal(matrix, np.load(ev / file, allow_pickle=False))
        matrices.append(matrix)
    assert max(norm_errors) < 0.01
    q, d = matrices
    scores = q @ d.T
    dids = [r['id'] for r in documents]
    ranks = {r['id']: r for r in rows(ev / 'rankings.jsonl')}
    assert len(ranks) == 1250 and set(ranks) == {r['id'] for r in queries}
    values = []
    for index, query in enumerate(queries):
        uid = query['id']
        order = np.argsort(-scores[index], kind='stable')[:10]
        ids = [dids[int(i)] for i in order]
        saved = ranks[uid]
        assert ids == saved['ranked_documents'], uid
        assert np.allclose(scores[index, order], saved['scores'], rtol=1e-5, atol=1e-6)
        positives = qrels[uid]
        assert positives == set(saved['relevant'])
        weights = [1 / math.log2(r + 2) for r in range(10)]
        value = sum(weights[r] for r, doc in enumerate(ids) if doc in positives) / sum(weights[:min(10, len(positives))])
        assert math.isclose(value, saved['ndcg10'], abs_tol=1e-12)
        values.append(value)
    score = 100 * math.fsum(values) / 1250
    saved_score = read(ev / 'verify_result.json')
    assert math.isclose(score, saved_score['ndcg10_percent'], abs_tol=1e-10)
    assert saved_score['numeric_comparison_pass'] == (score >= 68.19)
    selected = sorted(jobs, key=lambda j: hashlib.sha256(('trident009-bench-v1:' + j[0] + ':' + j[1]['id']).encode()).digest())[:24]
    expected_ids = [[role, item['id']] for role, item in selected]
    assert read(ev / 'bench/protocol.json')['selected'] == expected_ids
    performance = read(ev / 'bench/summary.json')
    assert len(performance['rounds']) == 4 and performance['all_successful']
    for run in performance['rounds']:
        calls = rows(ev / f"bench/c{run['concurrency']}-round{run['round']}.jsonl")
        assert len(calls) == 24 and [[r['role'], r['id']] for r in calls] == expected_ids
        assert all(r['status'] == 'ok' for r in calls)
        for row in calls:
            v = np.asarray(row['raw_response']['data'][0]['embedding'])
            assert v.shape == (2048,) and np.isfinite(v).all()
        latency = sorted(r['wall_seconds'] for r in calls)
        assert math.isclose(statistics.median(latency), run['latency_p50_seconds'], abs_tol=1e-12)
        assert math.isclose(latency[math.ceil(.95*24)-1], run['latency_p95_nearest_rank_seconds'], abs_tol=1e-12)
        assert math.isclose(24/run['wall_seconds'], run['vectors_per_second'], abs_tol=1e-12)
    final = read(RESULT / 'run-final.json')
    assert final['service_stopped'] and all(code == 0 for code in final['phases'].values())
    for name, digest in final['source_files'].items():
        assert sha(ROOT / 'scripts/submission009' / name) == digest
    report = {'UTC': dt.datetime.now(dt.timezone.utc).isoformat(), 'passed': True,
              'files_verified_against_image_manifest': len(expected), 'queries': 1250, 'documents': 1509,
              'raw_http_embeddings_checked': 2759, 'all_rankings_recomputed': True,
              'ndcg10_percent': score, 'publisher_baseline_percent': 70.19, 'threshold_percent': 68.19,
              'numeric_comparison_pass': score >= 68.19, 'max_vector_norm_error': max(norm_errors),
              'performance_requests_checked': 96, 'service_stopped': True,
              'archive_sha256': sha(ROOT / 'dist/session009-trident002-results.zip'),
              'official_acceptance': False, 'submitted': False}
    (ROOT / 'evidence/session009/trident-independent-audit002.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
