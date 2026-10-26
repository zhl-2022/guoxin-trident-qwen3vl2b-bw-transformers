"""Bounded native Trident full ChartQA Origin experiment with owned cleanup."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT=Path(__file__).resolve().parent
IMAGE=Path('/opt/guoxin-trident009')
TASK=IMAGE/'project/tasks/trident-qwen3vl2b-bw-transformers'
DATA=Path('/root/private_data/guoxin-session009/trident-chartqa')
OUTPUT=Path('/root/private_data/guoxin-session009/trident002')
PYTHON=IMAGE/'venv/bin/python'


def main():
    assert Path('/dev/kfd').exists()
    # Do not overlap with the ASR model or performance run on the same card.
    asr=json.loads(Path('/root/private_data/guoxin-session009/asr001/run-final.json').read_text())
    assert asr['service_stopped'] and asr['phases']['stop']==0
    OUTPUT.mkdir(parents=True,exist_ok=False)
    run_id=uuid.uuid4().hex
    env=os.environ.copy()
    env.update({'TRIDENT_MODEL_PATH':str(IMAGE/'model'),'TRIDENT_PORT':'30119','TRIDENT_MEM_FRAC':'0.45',
        'TRIDENT_LOAD_TIMEOUT':'600','TRIDENT_MAX_RUNTIME_SECONDS':'5400','PYTHONPATH':str(IMAGE/'project/scripts'),
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
    record={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'run_id':run_id,'phases':{},'submitted':False,
            'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    (OUTPUT/'run-start.json').write_text(json.dumps(record,indent=2))
    def step(argv,name,timeout):
        with (OUTPUT/(name+'.log')).open('xb') as log:
            process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=124
        print(json.dumps({'phase':name,'exit':code}),flush=True)
        return code
    driver=IMAGE/'project/scripts/trident_embedding_runtime_004.py'
    try:
        record['phases']['launch']=step([str(PYTHON),str(driver),'launch','--task-dir',str(TASK),'--run-id',run_id],'launch',700)
        if record['phases']['launch']:raise RuntimeError('Launch failed')
        record['phases']['evaluation']=step([str(PYTHON),str(ROOT/'evaluate_trident.py'),'--data',str(DATA),
            '--project',str(IMAGE/'project'),'--output',str(OUTPUT/'evaluation')],'evaluation',4200)
    except Exception as exc:record['error']=str(exc)
    finally:
        record['phases']['stop']=step([str(PYTHON),str(driver),'stop','--task-dir',str(TASK),'--run-id',run_id],'stop',90)
        try:
            with urllib.request.urlopen('http://127.0.0.1:30119/health',timeout=3):record['service_stopped']=False
        except (OSError,urllib.error.URLError):record['service_stopped']=True
        if (TASK/'runtime').is_dir():shutil.copytree(TASK/'runtime',OUTPUT/'runtime')
        record.update({'ended_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'official_acceptance':False,'submission_ready':False})
        result=OUTPUT/'evaluation/verify_result.json'
        record['accuracy_result']=json.loads(result.read_text()) if result.exists() else None
        (OUTPUT/'run-final.json').write_text(json.dumps(record,indent=2))
        with zipfile.ZipFile('/root/session009-trident002-results.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in OUTPUT.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(OUTPUT).as_posix())
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
