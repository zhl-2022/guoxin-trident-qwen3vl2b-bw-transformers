#!/usr/bin/env python3
"""Guarded one-attempt cleanup: compare resolved --config without editing prior state/code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

PROJECT=Path('/root/private_data/guoxin-hackathon-20261023-session005-trident-retry002')
RUN_ID='1b62c9a782bf40ec923a16edc132e8ad'
TASK_NAME='trident-qwen3vl2b-bw-transformers'
ALLOWED_PYTHON=Path('/root/private_data/guoxin-candidates/envs/trident-001/bin/python')
RUNTIME_SHA='0edc33da22c07a804c93047f918a4a39a58af90997390c49695d8b39a891d85b'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):
    raw=Path(p).read_bytes()
    if len(raw)>1024*1024:raise ValueError('Oversized state/config')
    return json.loads(raw),sha(raw)
def proc(pid):
    if type(pid) is not int or pid<=1:raise ValueError('Unsafe PID')
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        argv=[os.fsdecode(x) for x in Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
        return {'pid':pid,'ppid':int(fields[1]),'token':fields[19],'argv':argv}
    except FileNotFoundError:return None
def port_open():
    with socket.socket() as s:s.settimeout(2);return s.connect_ex(('127.0.0.1',30119))==0
def inspect(project=PROJECT):
    project=Path(project)
    for p in (project,*project.parents):
        if p.is_symlink():raise ValueError('Linked project/ancestor')
    task=project/'tasks'/TASK_NAME;runtime=task/'runtime'
    for p in (project/'scripts',project/'tasks',task):
        if p.is_symlink():raise ValueError('Unexpected linked source/task path')
    expected=project/'evidence/session005/trident-http-runtime-001'
    if not runtime.is_symlink() or runtime.resolve()!=expected.resolve():raise ValueError('Runtime alias differs from the fixed attempt')
    for p in (expected,*list(expected.parents)[:2]):
        if p.is_symlink():raise ValueError('Linked actual evidence path')
    state_path=runtime/'state.json'
    if state_path.is_symlink():raise ValueError('Linked state')
    state,state_sha=load(state_path)
    if state.get('run_id')!=RUN_ID:raise ValueError('Foreign run UUID')
    config_path=Path(state['config_path']);expected_config=expected/f'config-{RUN_ID}.json'
    if config_path.is_symlink() or config_path.resolve()!=expected_config.resolve():raise ValueError('Foreign config path')
    config,config_sha=load(config_path)
    if config.get('run_id')!=RUN_ID or Path(config['task']).resolve()!=task.resolve():raise ValueError('Foreign config identity/task')
    if config.get('url')!='http://127.0.0.1:30119' or config.get('port')!=30119:raise ValueError('Foreign service endpoint')
    supervisor=proc(state['supervisor_pid']);server=proc(state['server_pid'])
    if supervisor is None or supervisor['token']!=state.get('supervisor_token'):raise ValueError('Supervisor PID/start token differs or is gone')
    argv=supervisor['argv'];script=project/'scripts/recon2reason_runtime.py'
    if len(argv)!=5 or Path(argv[1]).resolve()!=script.resolve() or argv[2:4]!=['supervise','--config']:
        raise ValueError('Supervisor command/script differs')
    if script.is_symlink() or Path(argv[4]).resolve()!=config_path.resolve():raise ValueError('Supervisor config path does not resolve to its sealed state config')
    if Path(argv[0]).resolve()!=ALLOWED_PYTHON.resolve():raise ValueError('Unexpected supervisor interpreter')
    if sha(script.read_bytes())!=RUNTIME_SHA:raise ValueError('Frozen supervisor source SHA differs')
    if server is not None:
        if server['token']!=state.get('server_token') or server['ppid']!=supervisor['pid']:raise ValueError('Server PID/token/parent differs')
        if server['argv']!=config['command'] or state.get('command')!=config['command']:raise ValueError('Server command differs from its immutable config')
    return {'project':str(project),'run_id':RUN_ID,'state_path':str(state_path),'state_sha256_before':state_sha,
            'config_path':str(config_path),'config_sha256':config_sha,'supervisor':supervisor,'server':server,
            'logical_and_resolved_config_are_same_file':Path(argv[4]).resolve()==config_path.resolve(),
            'runtime_sha256':sha(script.read_bytes()),'port_30119_open_before':port_open()}

def run(project=PROJECT,execute=False,wait_seconds=45):
    record=inspect(project);record.update(executed=False,signal_sent=False,cleanup_verified=False)
    if not execute:return record
    # Revalidate immediately before signal; immutable config and owned process tokens
    # must still match. Do not rewrite the older state or configuration.
    current=inspect(project)
    for key in ('config_sha256','supervisor','server','runtime_sha256'):
        if current[key]!=record[key]:raise ValueError('Cleanup identity changed before signal: '+key)
    os.kill(record['supervisor']['pid'],signal.SIGTERM);record.update(executed=True,signal_sent=True)
    end=time.monotonic()+wait_seconds
    while time.monotonic()<end:
        a=proc(record['supervisor']['pid']);b=proc(record['server']['pid']) if record['server'] else None
        alive_a=a is not None and a['token']==record['supervisor']['token']
        alive_b=b is not None and b['token']==record['server']['token'] if record['server'] else False
        if not alive_a and not alive_b and not port_open():break
        time.sleep(.2)
    record.update(supervisor_alive_after=alive_a,server_alive_after=alive_b,port_30119_open_after=port_open())
    record['cleanup_verified']=not alive_a and not alive_b and not record['port_30119_open_after']
    state,_=load(record['state_path']);record['state_after']=state
    return record

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if sys.platform!='linux':raise ValueError('Linux /proc required')
    if a.output.exists() or a.output.is_symlink():raise FileExistsError('Never overwrite cleanup evidence')
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();code=2
    try:
        record=run(execute=a.execute);code=0 if (not a.execute or record['cleanup_verified']) else 2
    except Exception as exc:record={'error':str(exc),'type':type(exc).__name__,'cleanup_verified':False,'execute_requested':a.execute,'signal_sent':None}
    record.update(started_utc=start,run_id=RUN_ID,helper_sha256=sha(Path(__file__).read_bytes()),official_acceptance=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record,indent=2));return code

if __name__=='__main__':raise SystemExit(main())
