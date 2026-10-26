"""Complete ChartQA Origin retrieval via the adapted native vLLM HTTP endpoint."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import statistics
import time
import urllib.request


def ndcg_at_10(ranking,relevant):
    relevant=set(relevant)
    if not relevant:raise ValueError('Missing positive reference')
    if len(ranking)!=len(set(ranking)):raise ValueError('Duplicate ranked document')
    dcg=sum(1/math.log2(i+2) for i,uid in enumerate(ranking[:10]) if uid in relevant)
    ideal=sum(1/math.log2(i+2) for i in range(min(10,len(relevant))))
    return dcg/ideal


def prepared_local_image(path):
    from qwen_vl_utils.vision_process import process_vision_info
    # qwen-vl-utils treats file:// inputs by stripping the scheme, without URL
    # decoding. Pass the real filesystem path so commas/spaces remain intact.
    conv=[{'role':'user','content':[{'type':'image','image':str(Path(path).resolve()),
        'min_pixels':262144,'max_pixels':1310720}]}]
    images,_=process_vision_info(conv,image_patch_size=16)
    image=images[0];buffer=io.BytesIO();image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),list(image.size)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:30119')
    p.add_argument('--model',default='trident-qwen3vl-2b')
    p.add_argument('--concurrency',type=int,choices=(1,2,4),default=4)
    args=p.parse_args()
    if not args.endpoint.startswith('http://127.0.0.1:'):raise ValueError('Loopback only')
    import numpy as np
    sys.path.insert(0,str(args.project/'scripts'))
    from trident_embedding_smoke_004 import payload,response_check
    def read(name):return [json.loads(line) for line in (args.data/name).read_text(encoding='utf-8').splitlines()]
    queries=read('queries.jsonl'); documents=read('origin_corpus.jsonl'); qrels=read('qrels.jsonl')
    assert len(queries)==len(qrels)==1250 and len(documents)==1509
    qids=[r['id'] for r in queries]; dids=[r['id'] for r in documents]
    assert len(set(qids))==1250 and len(set(dids))==1509
    qrel={r['query_id']:r['relevant_docs'] for r in qrels}
    assert set(qrel)==set(qids) and all(set(v)<=set(dids) and v for v in qrel.values())
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'dataset':'OpenBMB/Trident-Eval/ChartQA','revision':'15c41335843f8880e24778843b462234b6427e4f',
              'scope':'complete Origin image corpus and all queries; no Mix claim','queries':1250,'documents':1509,
              'publisher_baseline_ndcg10_percent':70.19,'threshold_percent':68.19,'metric':'macro NDCG@10, binary relevance',
              'data_sha256':{n:hashlib.sha256((args.data/n).read_bytes()).hexdigest() for n in ('queries.jsonl','origin_corpus.jsonl','qrels.jsonl')},
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'endpoint':args.endpoint,'references_sent_to_model':False,'official_acceptance':False}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    jobs=[('query',r) for r in queries]+[('doc',r) for r in documents]
    def request(job):
        role,item=job
        result={'role':role,'id':item['id']}
        start=time.perf_counter()
        try:
            image=None
            if item.get('image'):
                path=(args.data/item['image']).resolve()
                if not path.is_relative_to(args.data.resolve()):raise ValueError('Image outside dataset')
                image,size=prepared_local_image(path)
                result.update({'source_image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prepared_size':size})
            request_body=payload(args.model,role,item.get('text'),image)
            raw=json.dumps(request_body).encode()
            result['request_sha256']=hashlib.sha256(raw).hexdigest()
            with urllib.request.urlopen(urllib.request.Request(args.endpoint+'/v1/embeddings',raw,{'Content-Type':'application/json'}),timeout=180) as r:
                response=r.read(1024*1024)
            value=json.loads(response)
            result['validation']=response_check(value,args.model)
            result['raw_response']=value
            result['status']='ok'
        except Exception as exc:result.update({'status':'failed','error':str(exc)})
        result['wall_seconds']=time.perf_counter()-start
        return result
    results={}; started=time.perf_counter()
    with (args.output/'raw-embeddings.jsonl').open('x',encoding='utf-8') as output, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(request,j) for j in jobs]
        for future in as_completed(futures):
            row=future.result()
            key=(row['role'],row['id'])
            if key in results:raise ValueError('Duplicate result identity')
            results[key]=row
            output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
            if len(results)%50==0:print(json.dumps({'responses':len(results),'failures':sum(r['status']!='ok' for r in results.values())}),flush=True)
    failed=[k for k,r in results.items() if r['status']!='ok']
    if failed:
        (args.output/'verify_result.json').write_text(json.dumps({'complete':False,'failed_ids':failed,'passed':False,'official_acceptance':False},indent=2))
        raise SystemExit(2)
    q=np.asarray([results[('query',uid)]['raw_response']['data'][0]['embedding'] for uid in qids],dtype=np.float32)
    d=np.asarray([results[('doc',uid)]['raw_response']['data'][0]['embedding'] for uid in dids],dtype=np.float32)
    np.save(args.output/'queries.npy',q);np.save(args.output/'documents.npy',d)
    scores=q@d.T
    values=[]
    with (args.output/'rankings.jsonl').open('x',encoding='utf-8') as output:
        for i,uid in enumerate(qids):
            order=np.argsort(-scores[i],kind='stable')[:10]
            ranking=[dids[int(j)] for j in order]
            value=ndcg_at_10(ranking,qrel[uid]);values.append(value)
            output.write(json.dumps({'id':uid,'ranked_documents':ranking,'scores':[float(scores[i,j]) for j in order],'relevant':qrel[uid],'ndcg10':value})+'\n')
    score=100*sum(values)/len(values)
    result={'complete':True,'queries':1250,'documents':1509,'ndcg10_percent':score,
            'publisher_baseline_percent':70.19,'absolute_tolerance_percentage_points':2,
            'numeric_comparison_pass':score>=68.19,'wall_seconds':time.perf_counter()-started,
            'official_acceptance':False,'submission_ready':False,'scope':protocol['scope']}
    (args.output/'verify_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    # Same fixed input set for both rounds; selection is independent of results.
    selected=sorted(jobs,key=lambda j:hashlib.sha256(('trident009-bench-v1:'+j[0]+':'+j[1]['id']).encode()).digest())[:24]
    bench=args.output/'bench';bench.mkdir()
    (bench/'protocol.json').write_text(json.dumps({'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),
        'selected':[(role,item['id']) for role,item in selected],
        'method':'24 fixed query/document requests; c1/c4; two rounds; warmed after complete evaluation'},indent=2))
    rounds=[]
    for concurrency in (1,4):
        for round_id in (1,2):
            started_bench=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:measurements=list(pool.map(request,selected))
            wall=time.perf_counter()-started_bench
            with (bench/f'c{concurrency}-round{round_id}.jsonl').open('x',encoding='utf-8') as out:
                for row in measurements:out.write(json.dumps(row,ensure_ascii=False)+'\n')
            successes=[r for r in measurements if r['status']=='ok'];latencies=sorted(r['wall_seconds'] for r in successes)
            summary={'concurrency':concurrency,'round':round_id,'requests':len(measurements),'success':len(successes),
                     'failed':len(measurements)-len(successes),'wall_seconds':wall,'vectors_per_second':len(successes)/wall,
                     'latency_p50_seconds':statistics.median(latencies) if latencies else None,
                     'latency_p95_nearest_rank_seconds':latencies[min(len(latencies)-1,math.ceil(len(latencies)*.95)-1)] if latencies else None}
            rounds.append(summary);print(json.dumps(summary),flush=True)
    (bench/'summary.json').write_text(json.dumps({'rounds':rounds,'all_successful':all(r['failed']==0 for r in rounds)},indent=2))
    raise SystemExit(0 if result['numeric_comparison_pass'] else 2)


if __name__=='__main__':main()
