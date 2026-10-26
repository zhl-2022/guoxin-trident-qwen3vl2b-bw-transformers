"""Record real SSE output chunks from a fixed audio input; no reference text sent."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import time
import urllib.request


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    audio=Path('/opt/guoxin-asr009/project/fixtures/smoke-session004/asr_zh.wav')
    body={'model':'qwen3-asr-1.7b','messages':[{'role':'user','content':[{'type':'audio_url','audio_url':{'url':'data:audio/wav;base64,'+base64.b64encode(audio.read_bytes()).decode()}}]}],
          'temperature':0,'max_tokens':1024,'stream':True,'stream_options':{'include_usage':True}}
    raw=json.dumps(body).encode();started=time.perf_counter();events=[];done=False
    request=urllib.request.Request('http://127.0.0.1:30002/v1/chat/completions',raw,{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=150) as response:
        content_type=response.headers['Content-Type'];assert content_type.startswith('text/event-stream')
        for line in response:
            line=line.decode().strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':done=True;break
            event=json.loads(data);assert 'error' not in event
            events.append({'elapsed_seconds':time.perf_counter()-started,'data':event})
    text=''.join(choice.get('delta',{}).get('content') or '' for event in events for choice in event['data'].get('choices',[]))
    finish=[choice.get('finish_reason') for event in events for choice in event['data'].get('choices',[]) if choice.get('finish_reason')]
    passed=done and len(events)>1 and '<asr_text>' in text and finish==['stop']
    record={'passed':passed,'scope':'SSE incremental output for one complete audio input; not chunked live audio ingestion',
            'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'request_sha256':hashlib.sha256(raw).hexdigest(),
            'done':done,'events':events,'content':text,'finish_reasons':finish,'content_type':content_type}
    with args.output.open('x') as f:f.write(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':passed,'events':len(events),'text':text}),flush=True)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
