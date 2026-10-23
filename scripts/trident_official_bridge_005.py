#!/usr/bin/env python3
"""Transparent external embedding evaluator execution; never interpret scores or acceptance."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import signal
import subprocess
import sys
import time

FLAGS={'official_execution_bridge_implemented':True,'official_schema_interpretation_implemented':False,
       'official_accuracy_verified':False,'official_performance_accepted':False,'submission_ready':False}
RESERVED={'bridge-inputs-before.json','bridge-start.json','bridge-final.json','bridge-stdout.log'}
MAX_FILE=256*1024*1024;MAX_TOTAL=20*1024*1024*1024;MAX_LOG=4*1024*1024

def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def unique(pairs):
    value={}
    for k,v in pairs:
        if k in value:raise ValueError('Duplicate JSON key')
        value[k]=v
    return value
def parsed(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def dump(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def no_links(p):
    p=Path(p).absolute()
    for part in (p,*p.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(),'st_file_attributes',0)&0x400):raise ValueError('Linked/reparse path refused: '+str(part))
    return p
def regular(p):
    p=no_links(p)
    if not p.is_file():raise ValueError('Declared input is not a regular file: '+str(p))
    return p
def relative(v):
    if not isinstance(v,str) or not v or '\\' in v or ':' in v:raise ValueError('Expected relative POSIX file path')
    p=PurePosixPath(v)
    if not p.parts or p.is_absolute() or '..' in p.parts or p.as_posix()!=v or any(x in ('','.','..') for x in p.parts):raise ValueError('Unsafe output/data relative path')
    return p
def required(env,key):
    v=env.get(key,'')
    if not v:raise ValueError('Missing explicit '+key)
    return v
def string_list(env,key,nonempty=False):
    value=parsed(required(env,key))
    if not isinstance(value,list) or (nonempty and not value) or len(value)>4096 or not all(isinstance(x,str) and '\0' not in x for x in value):raise ValueError('Invalid string-list contract: '+key)
    return value
def configuration(mode,selfcheck,task,env):
    task=no_links(task).resolve()
    if task.name!='trident-qwen3vl2b-bw-transformers' or parsed(regular(task/'task.json').read_bytes()).get('model_id')!='openbmb/Trident-Qwen3VL-2B':raise ValueError('Trident task identity differs')
    if selfcheck and mode!='verify':raise ValueError('Selfcheck is only defined for verify')
    phase='SELFCHECK' if selfcheck else mode.upper();prefix='OFFICIAL_TRIDENT_'+phase
    script=regular(required(env,prefix+'_SCRIPT'));args=string_list(env,prefix+'_ARGS_JSON')
    data=no_links(required(env,'OFFICIAL_TRIDENT_DATA_DIR')).resolve()
    if not data.is_dir():raise ValueError('Explicit data directory missing')
    names=string_list(env,'OFFICIAL_TRIDENT_DATA_FILES_JSON',True)
    if len(set(names))!=len(names):raise ValueError('Duplicate declared data file')
    data_files=[regular(data.joinpath(*relative(x).parts)) for x in names]
    qrels=regular(required(env,'OFFICIAL_TRIDENT_QRELS'));baseline=regular(required(env,'OFFICIAL_TRIDENT_BASELINE'))
    outputs=string_list(env,prefix+'_OUTPUTS_JSON',True)
    if len(set(outputs))!=len(outputs):raise ValueError('Duplicate output contract path')
    for name in outputs:
        relative(name)
        if name in RESERVED or any(PurePosixPath(name) in PurePosixPath(other).parents for other in outputs):raise ValueError('Reserved/conflicting output contract path')
    out=no_links(required(env,'OFFICIAL_TRIDENT_OUTPUT_DIR')).resolve()
    if out.exists():raise ValueError('Output directory must be NEW; prior evidence is never overwritten')
    if out.is_relative_to(data) or data.is_relative_to(out):raise ValueError('Output and declared data tree must be independent')
    timeout=float(required(env,'OFFICIAL_TRIDENT_TIMEOUT_SECONDS'));hash_budget=float(env.get('OFFICIAL_TRIDENT_HASH_TIMEOUT_SECONDS','30'))
    if not math.isfinite(timeout) or not 0<timeout<=7200 or not math.isfinite(hash_budget) or not 0<hash_budget<=600:raise ValueError('Invalid bounded execution/hash timeout')
    if script.suffix=='.py':
        interpreter=env.get('OFFICIAL_TRIDENT_PYTHON_BIN') or sys.executable
        if '/' in interpreter or '\\' in interpreter:interpreter=str(Path(interpreter).absolute())
        command=[interpreter,str(script),*args]
    elif script.suffix=='.sh':command=['bash',str(script),*args]
    else:
        if not os.access(script,os.X_OK):raise ValueError('Explicit evaluator is not executable')
        command=[str(script),*args]
    inputs=list(dict.fromkeys([task/'task.json',script,qrels,baseline,*data_files]))
    if any(p==out or out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):raise ValueError('Output overlaps a declared input')
    return {'phase':phase.lower(),'mode':mode,'selfcheck':selfcheck,'task':str(task),'script':str(script),'argv':command,
        'data_dir':str(data),'declared_data_files':[str(p) for p in data_files],'qrels':str(qrels),'baseline':str(baseline),
        'output_dir':str(out),'output_contract':outputs,'timeout_seconds':timeout,'hash_timeout_seconds':hash_budget,
        'declared_inputs':[str(p) for p in inputs],**FLAGS}

def fingerprints(paths,seconds):
    deadline=time.monotonic()+seconds;total=0;result=[]
    for value in paths:
        p=regular(value);before=p.stat()
        if before.st_size>MAX_FILE:raise ValueError('Declared file exceeds256MiB byte cap')
        total+=before.st_size
        if total>MAX_TOTAL:raise ValueError('Declared inputs exceed20GiB total byte cap')
        h=hashlib.sha256();count=0
        with p.open('rb') as f:
            while True:
                if time.monotonic()>=deadline:raise TimeoutError('Input/output hashing deadline exceeded')
                chunk=f.read(1024*1024)
                if not chunk:break
                h.update(chunk);count+=len(chunk)
        after=p.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or count!=before.st_size:raise ValueError('File changed during hash')
        result.append({'path':str(p),'size':count,'sha256':h.hexdigest(),'mtime_ns':before.st_mtime_ns})
    return result

def stop_group(p):
    if os.name=='posix':
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    elif p.poll() is None:p.terminate()
    try:p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name=='posix':
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        else:p.kill()
        p.wait(timeout=3)
    if os.name=='posix':
        # A reaped leader can leave descendants in its original owned session.
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass

def execute(config,env):
    before=fingerprints(config['declared_inputs'],config['hash_timeout_seconds'])
    out=Path(config['output_dir']);no_links(out);out.mkdir(parents=True,exist_ok=False)
    dump(out/'bridge-inputs-before.json',before);dump(out/'bridge-start.json',{'started_utc':utc(),'configuration':config})
    record={'started_utc':utc(),'configuration':config,'executed':False,'status':'preparing','inputs_unchanged':None,**FLAGS}
    # Freeze the known material paths before entering output cwd. Extra argv is
    # still forwarded verbatim; no attempt is made to guess its path semantics.
    bindings={'OFFICIAL_TRIDENT_DATA_DIR':config['data_dir'],'OFFICIAL_TRIDENT_QRELS':config['qrels'],
        'OFFICIAL_TRIDENT_BASELINE':config['baseline'],'OFFICIAL_TRIDENT_OUTPUT_DIR':config['output_dir'],
        'OFFICIAL_TRIDENT_'+config['phase'].upper()+'_SCRIPT':config['script']}
    child_env=dict(env);child_env.update(bindings);record['child_environment_path_bindings']=bindings
    p=None;began=time.monotonic();code=2
    try:
        with (out/'bridge-stdout.log').open('xb') as log:
            p=subprocess.Popen(config['argv'],cwd=out,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['executed']=True
            while p.poll() is None:
                if time.monotonic()-began>=config['timeout_seconds']:raise TimeoutError('External evaluator execution deadline exceeded')
                if log.tell()>MAX_LOG or (out/'bridge-stdout.log').stat().st_size>MAX_LOG:raise ValueError('External stdout byte cap exceeded')
                time.sleep(.03)
            record['external_exit_code']=p.wait();record['status']='executed_uninterpreted' if record['external_exit_code']==0 else 'executed_failed'
            if (out/'bridge-stdout.log').stat().st_size>MAX_LOG:raise ValueError('External stdout byte cap exceeded')
    except (Exception,KeyboardInterrupt) as exc:
        record.update(status='executed_timeout' if isinstance(exc,TimeoutError) else 'executed_error',error_type=type(exc).__name__,error=str(exc))
    finally:
        cleanup_previous=signal.getsignal(signal.SIGTERM);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if p is not None:
            try:stop_group(p)
            except Exception as exc:record.update(status='cleanup_unverified',cleanup_error=str(exc))
            record.update(child_reaped=p.poll() is not None,owned_session_cleanup_attempted=True,external_exit_code=p.returncode)
        record['execution_wall_seconds']=time.monotonic()-began
        try:
            after=fingerprints(config['declared_inputs'],config['hash_timeout_seconds']);record['inputs_after']=after
            record['inputs_unchanged']=after==before
            if not record['inputs_unchanged']:record.update(status='declared_inputs_changed',error='External program changed declared input bytes or metadata')
        except Exception as exc:record.update(status='input_integrity_unverified',input_after_error=str(exc),inputs_unchanged=False)
        if record['status']=='executed_uninterpreted':
            try:
                paths=[out.joinpath(*relative(name).parts) for name in config['output_contract']]
                record['outputs']=fingerprints(paths,config['hash_timeout_seconds'])
                code=0
            except Exception as exc:record.update(status='output_contract_failed',output_contract_error=str(exc))
        record.update(ended_utc=utc(),bridge_exit_code=code,external_output_content_interpreted=False,
            bridge_source_sha256=sha(Path(__file__).read_bytes()),stdout_sha256=sha((out/'bridge-stdout.log').read_bytes()),
            input_hash_scope='Only the explicitly declared file list; unlisted files in data_dir are not attested',
            execution_scope='POSIX owned process session' if os.name=='posix' else 'Windows direct child; no descendant-tree claim')
        dump(out/'bridge-final.json',record)
        signal.signal(signal.SIGTERM,cleanup_previous)
    return record,code

def main(argv=None,env=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','bench']);p.add_argument('--task-dir',type=Path,required=True)
    p.add_argument('--selfcheck',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv);env=dict(os.environ if env is None else env);execution_call_entered=False
    try:
        config=configuration(a.mode,a.selfcheck,a.task_dir,env)
        if a.dry_run:record={'status':'plan_only','executed':False,'configuration':config,**FLAGS};code=0
        else:
            previous=signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                execution_call_entered=True;record,code=execute(config,env)
            finally:signal.signal(signal.SIGTERM,previous)
    except (Exception,KeyboardInterrupt) as exc:record={'status':'execution_or_recording_failed' if execution_call_entered else 'blocked_or_preflight_failed',
        'executed':None if execution_call_entered else False,'error_type':type(exc).__name__,'error':str(exc),**FLAGS};code=2
    print(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False));return code

if __name__=='__main__':raise SystemExit(main())
