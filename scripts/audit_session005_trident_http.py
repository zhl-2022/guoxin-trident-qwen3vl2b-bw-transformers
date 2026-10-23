#!/usr/bin/env python3
"""Independently recompute the authenticated session005 retry002 HTTP development result."""
import base64
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'evidence/session005/remote-trident-http-final-001'
PROJECT='/root/private_data/guoxin-hackathon-20261023-session005-trident-retry002'
RUN_ID='1b62c9a782bf40ec923a16edc132e8ad'
ARCHIVE_SHA='1865af152496f46dc7ff9d26004b6cde16fdd53a070d3d2dcb355e5923fea912'
MANIFEST_SHA='d1d63542eb0bc5489763bb0c7a34b57532a927bd630bbb8afba42fd43bbf1179'
BUNDLE_SHA='f1104acb8dd5b5c4d35bc111c20144def9567f9ad7765b15f514d6e893689536'
HELPER_SHA='8c2dffd12dc089a3b418e6aaef2d67abf1f2df9a50506901ea2b70e9132b58a1'
CASES=['query_text','doc_text','doc_negative','doc_image','doc_fused']

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def require(condition,message):
    if not condition:raise ValueError(message)
def identity(p):
    p=Path(p);raw=p.read_bytes();return {'path':str(p.relative_to(ROOT)).replace('\\','/'),'size':len(raw),'sha256':sha(raw)}

def main():
    archive=(ROOT/'dist/hg-session005-trident-http-final-001.zip').read_bytes()
    require(sha(archive)==ARCHIVE_SHA and len(archive)==121608,'Export archive identity differs')
    manifest_raw=(CAPTURE/'export-manifest.json').read_bytes();require(sha(manifest_raw)==MANIFEST_SHA,'Manifest identity differs')
    manifest=json.loads(manifest_raw);require(manifest['project']==PROJECT,'Foreign project')
    for item in manifest['files']:
        raw=(CAPTURE/item['archive_path']).read_bytes()
        require(len(raw)==item['size'] and sha(raw)==item['sha256'],'Extracted payload changed')
    before={x['source']:(x['size'],x['sha256']) for x in manifest['script_hashes_before']}
    after={x['source']:(x['size'],x['sha256']) for x in manifest['script_hashes_after']}
    require(before==after and manifest['script_hashes_unchanged'] is True,'Source seal differs')
    bundle=(ROOT/'dist/session005-trident-http-002.zip').read_bytes();require(sha(bundle)==BUNDLE_SHA,'Frozen source ZIP differs')
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        bmanifest=z.read('session005-trident-http-manifest.json')
        require(bmanifest==(CAPTURE/'project/session005-trident-http-manifest.json').read_bytes(),'Remote source manifest differs')
        bound=[]
        for name in z.namelist():
            if name.startswith('scripts/') or name.endswith('.sh'):
                raw=z.read(name);require(before[PROJECT+'/'+name]==(len(raw),sha(raw)),'Actual source seal differs from frozen bundle')
                bound.append({'source':PROJECT+'/'+name,'sha256':sha(raw),'size':len(raw)})
        binding=json.loads(z.read('fixtures/session005-trident-reference/reference-binding.json'))
        reference_raw=z.read('fixtures/session005-trident-reference/worker-result.json')
        fixture_raw=z.read('tasks/trident-qwen3vl2b-bw-transformers/fixtures/red-rectangle.png')
        require(sha(reference_raw)==binding['worker_result_sha256'] and sha(fixture_raw)==binding['fixture_sha256'],'Frozen development reference identity differs')
    helper_sources=[s for s,v in before.items() if Path(s).name=='session005_trident_guarded_cleanup_001.py']
    require(len(helper_sources)==1 and before[helper_sources[0]][1]==HELPER_SHA,'Executed helper source identity differs')
    require(sha((ROOT/'dist/session005_trident_guarded_cleanup_001.py').read_bytes())==HELPER_SHA,'Reviewed local helper differs')
    base=CAPTURE/'project/evidence/session005';run=base/'trident-http-retry002';runtime=base/'trident-http-runtime-001'
    start=read(run/'run-start.json');final=read(run/'run-final.json')
    require(start['run_id']==final['run_id']==RUN_ID,'Run identity differs')
    require(final['functional_HTTP_passed'] is True and final['exit_code']==2 and final['cleanup_verified'] is False,'Original controller state changed')
    for item,name in zip(final['steps'],['launch.log','smoke-client.log']):
        require(item['exit_code']==0 and not item['timed_out'] and item['child_reaped'],'Launch/smoke step failed')
        require(sha((run/name).read_bytes())==item['log_sha256'],'Executed step log seal differs')
    native_before=read(run/'native-before.json');native_after=read(run/'native-after.json')
    require(native_before==native_after and final['native_metadata_unchanged'] is True,'Native metadata differs')
    require(native_before['packages']['qwen-vl-utils']['version']=='0.0.14','Unexpected qwen-vl-utils')
    rows=[json.loads(x) for x in (run/'smoke.jsonl').read_text().splitlines()]
    require(rows[-1]['functional_embedding_api_passed'] is True,'Smoke summary failed')
    require(rows[0]['source_image_sha256']==binding['fixture_sha256'] and rows[0]['prepared_image_size']==[512,512],'Input image differs')
    requests=[x for x in rows if 'request' in x];require([x['case'] for x in requests]==CASES,'Request order differs')
    vectors=[];responses=[];prepared_images=[]
    for index,case in enumerate(CASES):
        request=requests[index]['request'];messages=request['messages']
        require(request['model']=='trident-qwen3vl-2b' and request['continue_final_message'] is True and request['use_activation'] is True,'Request contract differs')
        require([m['role'] for m in messages]==['system','user','assistant'] and messages[-1]['content']==[{'type':'text','text':''}],'Prompt roles differ')
        instruction='Represent the query for retrieving relevant content.' if index==0 else 'Represent the candidate content for retrieval.'
        require(messages[0]['content']==[{'type':'text','text':instruction}],'Publisher instruction differs')
        content=messages[1]['content'];expected_types=[['text'],['text'],['text'],['image_url'],['image_url','text']][index]
        require([v['type'] for v in content]==expected_types,'Image/text order differs')
        text=[v['text'] for v in content if v['type']=='text']
        require(text==(['a blue circle'] if index==2 else [] if index==3 else ['a red rectangle']),'Fixed input text differs')
        if index>=3:
            data=content[0]['image_url']['url'];require(data.startswith('data:image/png;base64,'),'Expected bounded prepared PNG')
            raw=base64.b64decode(data.split(',',1)[1],validate=True);require(raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(512,512),'Prepared PNG dimensions differ')
            prepared_images.append({'case':case,'size':len(raw),'sha256':sha(raw)})
        response_rows=[x for x in rows if x.get('case')==case and 'raw_response' in x];require(len(response_rows)==1,'Missing/duplicate response')
        raw=response_rows[0]['raw_response'].encode();value=json.loads(raw);v=value['data'][0]['embedding'];u=value['usage']
        require(value['object']=='list' and value['model']=='trident-qwen3vl-2b' and len(value['data'])==1 and value['data'][0]['index']==0,'Response model/schema differs')
        require(len(v)==2048 and all(type(n) in (int,float) and math.isfinite(n) for n in v),'Vector dimension/numbers differ')
        norm=math.sqrt(sum(n*n for n in v));require(math.isclose(norm,1,rel_tol=.005,abs_tol=.005),'Vector normalization differs')
        require(type(u['prompt_tokens']) is int and u['prompt_tokens']==u['total_tokens']>0 and u['completion_tokens']==0,'Usage differs')
        vectors.append(v);responses.append({'case':case,'raw_response_bytes':len(raw),'raw_response_sha256':sha(raw),'dimension':len(v),'l2_norm':norm,'usage':u})
    require(prepared_images[0]['sha256']==prepared_images[1]['sha256'],'Image and fused prepared PNG differ')
    reference=json.loads(reference_raw);refvectors=reference['embeddings']['queries']+reference['embeddings']['documents']
    require(len(refvectors)==5 and all(len(x)==2048 for x in refvectors),'Development reference shape differs')
    scores=[sum(a*b for a,b in zip(vectors[0],v)) for v in vectors[1:]];rank=sorted(range(4),key=lambda i:scores[i],reverse=True)
    diff=max(abs(a-b) for x,y in zip(vectors,refvectors) for a,b in zip(x,y))
    comparison=read(run/'embedding-comparison.json')
    score_diffs=[abs(a-b) for a,b in zip(scores,comparison['cosine_dot_scores'])]
    require(max(score_diffs)<=1e-12 and rank==comparison['ranked_document_indices']==binding['offline_rank'],'Recomputed rank/scores differ')
    require(diff==comparison['max_abs_vector_difference_from_offline_development_reference']==0.0,'Recomputed vector difference differs')
    log=(runtime/'server.log').read_text();http200=len(re.findall(r'"POST /v1/embeddings HTTP/1\.1" 200 OK',log));require(http200==5,'Server successful POST count differs')
    original_cleanup=read(run/'cleanup.json');require(original_cleanup['cleanup_verified'] is False and original_cleanup['stop_step']['exit_code']==2,'Original cleanup failure not preserved')
    stop_log=(run/'stop.log').read_text();require('supervisor command does not match this script and run config' in stop_log,'Original stop cause differs')
    helper=read(base/'trident-guarded-cleanup-executed-001.json');config_path=runtime/f'config-{RUN_ID}.json';config=read(config_path);state=read(runtime/'state.json')
    require(helper['helper_sha256']==HELPER_SHA and helper['run_id']==RUN_ID and helper['config_sha256']==sha(config_path.read_bytes()),'Cleanup helper/config identity differs')
    require(helper['cleanup_verified'] and helper['signal_sent'] and helper['executed'] and not helper['supervisor_alive_after'] and not helper['server_alive_after'] and not helper['port_30119_open_after'],'Independent cleanup did not pass')
    require(state==helper['state_after'] and state['status']=='stopped' and state['reason']=='stop requested','Final state differs from helper')
    for label in ['supervisor','server']:
        require(helper[label]['pid']==state[label+'_pid'] and helper[label]['token']==state[label+'_token'],'Cleanup PID token differs')
    require(helper['supervisor']['argv'][4]!=state['config_path'] and helper['logical_and_resolved_config_are_same_file'] is True,'Symlink-path diagnosis differs')
    snapshots=[read(CAPTURE/f'export-snapshots/{phase}/trident-qwen3vl2b-bw-transformers/state.json') for phase in ['before','after']]
    require(snapshots[0]==snapshots[1]==state,'Export final state snapshots differ')
    flags={k:False for k in ['official_accuracy_verified','official_performance_verified','official_embedding_evaluator_bridge_implemented','submission_ready','contest_submission_performed']}
    result={'schema_version':1,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'archive_sha256':ARCHIVE_SHA,'archive_bytes':121608,'export_manifest_sha256':MANIFEST_SHA,'project':PROJECT,'run_id':RUN_ID,
      'all_payloads_rechecked_against_authenticated_manifest':True,'payload_files':len(manifest['files']),'ZIP_entries':41,
      'script_seals_before_after_count':len(before),'actual_project_source_seals_match_frozen_bundle':bound,
      'helper_source_seal':{'source':helper_sources[0],'sha256':HELPER_SHA,'matches_reviewed_local_alias':True},
      'source_payloads_reconstructed':False,'source_hash_only_does_not_alone_prove_execution':True,
      'functional_HTTP_locally_audited_passed':True,'responses':responses,'server_log_HTTP200_embedding_POSTs':http200,
      'source_image_sha256':binding['fixture_sha256'],'prepared_images':prepared_images,
      'cosine_dot_scores':scores,'score_recomputation_abs_differences':score_diffs,'score_recomputation_tolerance':1e-12,
      'score_recomputation_boundary':'Python versions may use different summation algorithms; ranking and vector element equality are checked separately',
      'ranked_document_indices':rank,'rank_matches_offline_development_reference':True,
      'max_abs_vector_difference_from_offline_development_reference':diff,'reference_worker_result_sha256':binding['worker_result_sha256'],
      'reference_is_publisher_TF_or_official_baseline':False,'actual_HTTP_prompt_token_ids_captured':False,
      'native_metadata_before_after_equal':True,'native_packages':native_before['packages'],
      'runtime_full_checkpoint_hash_gate_in_sealed_source_precedes_successful_launch':True,
      'standalone_per_file_session005_checkpoint_hash_result_captured':False,
      'controller':{'started_utc':final['started_utc'],'ended_utc':final['ended_utc'],'wall_seconds':final['wall_seconds'],
         'launch_wall_seconds':final['steps'][0]['wall_seconds'],'smoke_wall_seconds':final['steps'][1]['wall_seconds'],
         'exit_code':final['exit_code'],'original_cleanup_verified':False,'original_stop_exit_code':2},
      'independent_guarded_cleanup':{'locally_audited_passed':True,'helper_sha256':HELPER_SHA,'started_utc':helper['started_utc'],
        'supervisor_pid':state['supervisor_pid'],'supervisor_token':state['supervisor_token'],'server_pid':state['server_pid'],'server_token':state['server_token'],
        'both_owned_processes_observed_gone':True,'port_30119_observed_closed':True,'state_ended_utc':state['ended_utc'],'state_status':state['status'],'state_reason':state['reason']},
      'evidence':[identity(p) for p in [run/'run-final.json',run/'smoke.jsonl',run/'embedding-comparison.json',run/'cleanup.json',run/'native-before.json',run/'native-after.json',runtime/'server.log',runtime/'state.json',base/'trident-guarded-cleanup-executed-001.json']],
      'limits':['Fixed five synthetic development requests, no official qrels or precision threshold',
        'No publisher Transformers vector comparison or formal embedding benchmark',
        'Native package METADATA identities are preserved; this is not a full binary tree attestation',
        manifest['capture_semantics']],**flags}
    out=ROOT/'evidence/session005/remote-trident-http-final-001-audit.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'audit':str(out),'sha256':sha(out.read_bytes()),'HTTP_pass':True,'rank':rank,'max_abs_diff':diff,'native_unchanged':True,'independent_cleanup_pass':True},indent=2))

if __name__=='__main__':main()
