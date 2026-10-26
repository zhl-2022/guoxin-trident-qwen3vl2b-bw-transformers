"""Replay both frozen submission repositories in a fresh saved-image instance."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile

BASE=Path('/root/private_data/guoxin-session010')
PACKAGES={
 'asr':('qwen3-asr-bw-vllm','99ee93e0768198ddfcf39958e471da385274a8304ce7d75351f69ba038c52076',30002),
 'trident':('trident-qwen3vl2b-bw-transformers','c2baa3ca66c590dba2bd0f6da29af63abcf83c7a92b50e5e2f0128cedabba422',30119),
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline-utc',required=True);args=ap.parse_args()
    deadline=dt.datetime.fromisoformat(args.deadline_utc)
    assert deadline.utcoffset()==dt.timedelta(0) and Path('/dev/kfd').exists()
    out=BASE/'delivery-replay010';out.mkdir(exist_ok=False)
    summary={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'hostname':socket.gethostname(),
             'source_image_id':'2114587646006431746','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'models':{},'submitted':False}
    for name,(slug,archive_sha,port) in PACKAGES.items():
        record={'run_id':uuid.uuid4().hex,'phases':{},'archive_sha256':archive_sha}
        target=out/name;target.mkdir()
        archive=BASE/f'delivery-inputs/session009-{slug}-publisher-review002.zip'
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==archive_sha
        repo=out/(name+'-repository');repo.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for member in z.namelist():assert (repo/member).resolve().is_relative_to(repo.resolve())
            z.extractall(repo)
        manifest=json.loads((repo/'publisher-delivery-manifest009.json').read_text())
        for item in manifest['files']:
            file=repo/item['path'];assert file.stat().st_size==item['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
        record['repository_manifest_verified']=len(manifest['files'])
        image=Path('/opt')/('guoxin-asr009' if name=='asr' else 'guoxin-trident009')
        python=str(image/'venv/bin/python')
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ENV_LOG=str(target/'environment.json'))
        def step(argv,label,limit,cleanup=False):
            remaining=(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()
            seconds=90 if cleanup else min(limit,max(0,int(remaining)-120))
            if seconds<=0:raise TimeoutError('Delivery replay deadline')
            with (target/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    code=124
            record['phases'][label]=code;print(json.dumps({'model':name,'phase':label,'exit':code}),flush=True)
            return code
        try:
            for checkport in (30000,30001,30002,30119,30129,30139):
                with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',checkport))!=0
            if step(['bash','env_check.sh'],'environment',90):raise RuntimeError('environment')
            if step([python,'verify.py','--selfcheck'],'selfcheck',90):raise RuntimeError('selfcheck')
            if step(['bash','run.sh','--run-id',record['run_id']],'launch',1050):raise RuntimeError('launch')
            if step(['bash','smoke.sh'],'smoke',180):raise RuntimeError('smoke')
            if name=='asr':
                if step([python,str(BASE/'delivery-inputs/asr_stream010.py'),'--output',str(target/'stream.json')],'stream',180):raise RuntimeError('stream')
            code=step([python,'verify.py','--output',str(target/'evaluation')],'full_evaluation',1800)
            record['full_evaluation_exit']=code
            if name=='asr' and code==0:step(['bash','bench.sh','--output',str(target/'bench')],'benchmark',480)
        except Exception as error:record['error']=str(error)
        finally:
            step(['bash','stop.sh','--run-id',record['run_id']],'stop',90,cleanup=True)
            with socket.socket() as sock:record['service_stopped']=sock.connect_ex(('127.0.0.1',port))!=0
            if (target/'evaluation/verify_result.json').exists():record['accuracy_result']=json.loads((target/'evaluation/verify_result.json').read_text())
            runtime=image/'project/tasks'/slug/'runtime'
            if runtime.exists():shutil.copytree(runtime,target/'runtime',ignore=shutil.ignore_patterns('audio-venv','__pycache__'))
            record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            (target/'run-final.json').write_text(json.dumps(record,indent=2)+'\n')
            summary['models'][name]=record
            (out/'run-progress.json').write_text(json.dumps(summary,indent=2)+'\n')
            with zipfile.ZipFile(f'/root/session010-delivery-replay-{name}.zip','x',zipfile.ZIP_DEFLATED) as z:
                for p in target.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(target).as_posix())
    summary['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    (out/'run-final.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
