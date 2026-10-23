#!/usr/bin/env python3
"""Session005 isolated Trident HTTP run; stdlib controller, no package installation.

Copyright (c) 2026 the task author. No new repository-wide license selected.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path('/root/private_data/guoxin-hackathon-20261023-session005-trident-retry002')
EXPECTED_ENV = Path('/root/private_data/guoxin-candidates/envs/trident-001')
MODEL = '/root/private_data/guoxin-candidates/models/trident'
TASK = ROOT / 'tasks/trident-qwen3vl2b-bw-transformers'
RUN_ID = '1b62c9a782bf40ec923a16edc132e8ad'
MANIFEST = 'session005-trident-http-manifest.json'
CASES = ['query_text','doc_text','doc_negative','doc_image','doc_fused']
MODEL_NAME = 'trident-qwen3vl-2b'
PORT = 30119
PHASE_BUDGET = 2100
CLEANUP_BUDGET = 60
FLAGS = {'official_accuracy_verified':False,'official_performance_verified':False,
         'official_embedding_evaluator_bridge_implemented':False,'submission_ready':False}


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def bundle_check(root=ROOT):
    root=Path(root).resolve();manifest=read(root/MANIFEST);seen=set()
    if manifest.get('schema')!='session005-trident-http-source-v1':raise ValueError('Wrong bundle manifest schema')
    from recon2reason_runtime import checked_run_id
    checked_run_id(RUN_ID)
    if manifest.get('run_id')!=RUN_ID or manifest.get('target_project')!=EXPECTED_ROOT.as_posix():
        raise ValueError('Bundle run/project identity differs from retry002 controller')
    self_path=Path(__file__).resolve().relative_to(root).as_posix()
    if self_path!='scripts/session005_trident_http_retry002.py':
        raise ValueError('Retry002 controller must run from its manifest-bound source path')
    for item in manifest['files']:
        rel=PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:raise ValueError('Unsafe or duplicate manifest path')
        seen.add(str(rel));path=root.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Bundle path escapes root')
        raw=path.read_bytes()
        if len(raw)!=item['size'] or sha(raw)!=item['sha256']:raise ValueError('Bundle file SHA/size differs: '+str(rel))
    if self_path not in seen:raise ValueError('Controller own source is absent from manifest')
    return {'manifest_sha256':sha((root/MANIFEST).read_bytes()),'files_verified':len(seen)}


def dependency_metadata():
    expected=read(ROOT/'fixtures/session005-trident-reference/environment.json')['distributions'];actual={}
    for name,baseline in expected.items():
        dist=importlib.metadata.distribution(name)
        root=str(Path(dist.locate_file('')).resolve())
        if dist.version!=baseline['version'] or root!=baseline['root']:
            raise ValueError('Existing candidate dependency changed: '+name)
        text=dist.read_text('METADATA')
        actual[name]={'version':dist.version,'root':root,'metadata_sha256':sha(text.encode()) if text is not None else None}
    return {'python':sys.executable,'prefix':sys.prefix,'packages':actual,'metadata_only':True,
            'torch_module_already_imported':'torch' in sys.modules,'package_installation_performed':False}


def environment():
    # Preserve the caller's existing visible-device scope; do not set/expand it.
    env=os.environ.copy()
    env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
        'TRIDENT_MODEL_PATH':MODEL,'TRIDENT_SERVED_NAME':MODEL_NAME,'TRIDENT_PORT':str(PORT),
        'TRIDENT_MEM_FRAC':'0.45','TRIDENT_LOAD_TIMEOUT':'600','TRIDENT_MAX_RUNTIME_SECONDS':'1800'})
    return env


def step(argv,log,seconds,env):
    started=time.monotonic();process=None;timed_out=False
    with Path(log).open('xb') as output:
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=seconds)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            timed_out=isinstance(exc,subprocess.TimeoutExpired)
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            if not timed_out:raise
            code=124
    return {'argv':argv,'exit_code':code,'timed_out':timed_out,'child_reaped':True,
            'wall_seconds':time.monotonic()-started,'log_sha256':sha(Path(log).read_bytes())}


def compare_smoke(path):
    from trident_embedding_smoke_004 import response_check
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if not rows or rows[-1].get('functional_embedding_api_passed') is not True:raise ValueError('HTTP smoke did not pass')
    binding=read(ROOT/'fixtures/session005-trident-reference/reference-binding.json')
    if rows[0]['source_image_sha256']!=binding['fixture_sha256']:raise ValueError('HTTP source image differs from sealed fixture')
    requests=[r['case'] for r in rows if 'request' in r]
    if requests!=CASES:raise ValueError('HTTP request order/cases differ')
    vectors=[]
    for case in CASES:
        values=[r['raw_response'] for r in rows if r.get('case')==case and 'raw_response' in r]
        if len(values)!=1:raise ValueError('Missing/duplicate raw HTTP embedding response')
        value=json.loads(values[0]);response_check(value,MODEL_NAME);vectors.append(value['data'][0]['embedding'])
    reference_path=ROOT/'fixtures/session005-trident-reference/worker-result.json'
    raw=reference_path.read_bytes()
    if sha(raw)!=binding['worker_result_sha256']:raise ValueError('Offline reference bytes changed')
    reference=json.loads(raw)
    refvectors=reference['embeddings']['queries']+reference['embeddings']['documents']
    dots=[sum(a*b for a,b in zip(vectors[0],v)) for v in vectors[1:]]
    ranks=sorted(range(4),key=lambda i:dots[i],reverse=True)
    differences=[abs(a-b) for x,y in zip(vectors,refvectors) for a,b in zip(x,y)]
    return {'HTTP_calls':5,'query_vectors':1,'document_vectors':4,'dimensions':[len(v) for v in vectors],
        'cosine_dot_scores':dots,'ranked_document_indices':ranks,
        'offline_reference_rank':binding['offline_rank'],'rank_matches_offline_reference':ranks==binding['offline_rank'],
        'max_abs_vector_difference_from_offline_development_reference':max(differences),
        'reference_worker_result_sha256':binding['worker_result_sha256'],
        'reference_is_publisher_TF_or_official_baseline':False,'scope':'Fixed synthetic development case only',**FLAGS}


def cleanup(output,env):
    from recon2reason_runtime import process_token
    state_path=TASK/'runtime/state.json';state=read(state_path) if state_path.exists() else None
    result={'utc':stamp(),'owned_run_id':RUN_ID,'state_existed':state is not None,'stop_called':False}
    if state is None:
        config_exists=(TASK/f'runtime/config-{RUN_ID}.json').exists()
        result.update(cleanup_verified=not config_exists,owned_config_existed=config_exists,
            scope='No state; no signal attempted. A created config without state leaves launch cleanup unverified.')
        return result
    if state.get('run_id')!=RUN_ID:
        result.update(cleanup_verified=False,reason='Different run identity; no stop/signal attempted')
        return result
    result['stop_called']=True
    result['stop_step']=step([sys.executable,str(ROOT/'scripts/trident_embedding_runtime_004.py'),'stop',
        '--task-dir',str(TASK),'--run-id',RUN_ID],output/'stop.log',45,env)
    state=read(state_path);result['state_after']=state
    result['observed_process_tokens']={key:process_token(state.get(key+'_pid',0)) for key in ('supervisor','server')}
    with socket.socket() as sock:
        sock.settimeout(2);result['port_open']=sock.connect_ex(('127.0.0.1',PORT))==0
    result['cleanup_verified']=(result['stop_step']['exit_code']==0 and
        all(token is None for token in result['observed_process_tokens'].values()) and not result['port_open'])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','run'])
    parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args(argv);started=time.monotonic();seal=bundle_check()
    plan={'root':str(ROOT),'expected_root':str(EXPECTED_ROOT),'python':str(EXPECTED_ENV/'bin/python'),
        'model':MODEL,'run_id':RUN_ID,'port':PORT,'service_runtime_seconds':1800,
        'controller_phase_budget_seconds':PHASE_BUDGET,'cleanup_budget_seconds':CLEANUP_BUDGET,
        'dry_run':a.dry_run,'bundle':seal,'native_packages_will_be_installed':False,**FLAGS}
    if a.dry_run:print(json.dumps(plan,indent=2));return 0
    if sys.platform!='linux' or ROOT!=EXPECTED_ROOT or Path(sys.prefix).resolve()!=EXPECTED_ENV:
        raise ValueError('Actual execution requires the isolated session005 root and existing trident-001 interpreter')
    output=ROOT/'evidence/session005'/('trident-http-preflight-retry002' if a.action=='preflight' else 'trident-http-retry002')
    output.mkdir(parents=True,exist_ok=False);write(output/'run-start.json',{'utc':stamp(),**plan})
    try:
        before=dependency_metadata();write(output/'native-before.json',before)
    except Exception as exc:
        write(output/'run-final.json',{'ended_utc':stamp(),'exit_code':2,'GPU_inference_executed':False,
            'error':{'type':type(exc).__name__,'message':str(exc)},**FLAGS})
        print(json.dumps({'output':str(output),'error':str(exc),**FLAGS}));return 2
    if a.action=='preflight':
        write(output/'preflight-result.json',{'metadata_dependency_contract_passed':True,
            'qwen_vl_utils_version':before['packages']['qwen-vl-utils']['version'],
            'model_weight_hash_deferred_to_bounded_launch':True,'GPU_inference_executed':False,**FLAGS})
        print(json.dumps({'output':str(output),'metadata_dependency_contract_passed':True,'GPU_inference_executed':False}));return 0
    runtime_target=ROOT/'evidence/session005/trident-http-runtime-001'
    runtime_target.mkdir(exist_ok=False)
    # The canonical supervisor keeps its task/runtime location. This new-project
    # symlink sends every config/state/server/supervisor log into session005.
    (TASK/'runtime').symlink_to(runtime_target,target_is_directory=True)
    env=environment();result={'started_utc':stamp(),'run_id':RUN_ID,'steps':[],**FLAGS};code=2
    cleaning=False
    def interrupted(_number,_frame):
        if not cleaning:raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        launch=[sys.executable,str(ROOT/'scripts/trident_embedding_runtime_004.py'),'launch','--task-dir',str(TASK),'--run-id',RUN_ID]
        s=step(launch,output/'launch.log',min(750,PHASE_BUDGET-(time.monotonic()-started)),env);result['steps'].append(s)
        if s['exit_code']!=0:raise ValueError('Bounded launch/checkpoint preflight failed; see launch.log')
        smoke=[sys.executable,str(ROOT/'scripts/trident_embedding_smoke_004.py'),'--base-url',f'http://127.0.0.1:{PORT}',
            '--model',MODEL_NAME,'--image',str(TASK/'fixtures/red-rectangle.png'),'--log',str(output/'smoke.jsonl'),'--timeout','30']
        s=step(smoke,output/'smoke-client.log',min(600,PHASE_BUDGET-(time.monotonic()-started)),env);result['steps'].append(s)
        if s['exit_code']!=0:raise ValueError('HTTP smoke failed; raw errors retained in smoke.jsonl')
        comparison=compare_smoke(output/'smoke.jsonl');write(output/'embedding-comparison.json',comparison)
        result['functional_HTTP_passed']=True;result['rank_matches_offline_reference']=comparison['rank_matches_offline_reference']
        code=0 if comparison['rank_matches_offline_reference'] else 2
    except (Exception,KeyboardInterrupt) as exc:
        result['error']={'type':type(exc).__name__,'message':str(exc)};code=130 if isinstance(exc,KeyboardInterrupt) else 2
    finally:
        cleaning=True
        try:
            cleaned=cleanup(output,env);write(output/'cleanup.json',cleaned);result['cleanup_verified']=cleaned['cleanup_verified']
            if not cleaned['cleanup_verified']:code=2
        except Exception as exc:result['cleanup_error']=str(exc);code=2
        try:
            after=dependency_metadata();write(output/'native-after.json',after)
            result['native_metadata_unchanged']=before==after
            if before!=after:code=2
        except Exception as exc:result['native_after_error']=str(exc);code=2
        result.update(ended_utc=stamp(),wall_seconds=time.monotonic()-started,exit_code=code)
        write(output/'run-final.json',result)
        signal.signal(signal.SIGTERM,previous)
    print(json.dumps({'output':str(output),'exit_code':code,'functional_HTTP_passed':result.get('functional_HTTP_passed',False),
        'cleanup_verified':result.get('cleanup_verified',False),**FLAGS},indent=2));return code


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'error':{'type':type(exc).__name__,'message':str(exc)},**FLAGS}));raise SystemExit(2)
