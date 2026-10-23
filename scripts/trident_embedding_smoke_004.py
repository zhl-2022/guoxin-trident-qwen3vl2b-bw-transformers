#!/usr/bin/env python3
"""True embeddings HTTP smoke using vLLM's documented messages extension.

Copyright (c) 2026 the task author. No repository-wide license selected.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import ipaddress
import json
import math
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

LIMIT=2*1024*1024
INSTRUCTIONS={"query":"Represent the query for retrieving relevant content.","doc":"Represent the candidate content for retrieval."}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError("Embedding smoke redirects refused")


def base_url(value):
    p=urllib.parse.urlsplit(value)
    if (p.scheme!="http" or not p.hostname or p.username or p.password or p.query or p.fragment or p.path.rstrip("/")
        or p.hostname!="localhost" and not ipaddress.ip_address(p.hostname).is_loopback):
        raise ValueError("Use credential-free loopback HTTP server root")
    p.port
    return value.rstrip("/")


def payload(model,role,text=None,image=None):
    content=[]
    if image:content.append({"type":"image_url","image_url":{"url":image}})
    if text is not None:content.append({"type":"text","text":text})
    if not content:raise ValueError("Text or prepared image required")
    return {"model":model,"messages":[{"role":"system","content":[{"type":"text","text":INSTRUCTIONS[role]}]},
        {"role":"user","content":content},{"role":"assistant","content":[{"type":"text","text":""}]}],
        "encoding_format":"float","continue_final_message":True,"add_special_tokens":True,"use_activation":True}


def response_check(value,model):
    if not isinstance(value,dict) or value.get("error") or value.get("object")!="list" or value.get("model")!=model:
        raise ValueError("Embedding response/model mismatch")
    data=value.get("data")
    if not isinstance(data,list) or len(data)!=1 or data[0].get("index")!=0:
        raise ValueError("Expected exactly one embedding row with index0")
    vector=data[0].get("embedding")
    if not isinstance(vector,list) or len(vector)!=2048 or any(type(x) not in (float,int) or not math.isfinite(x) for x in vector):
        raise ValueError("Require finite 2048-dimensional float embedding")
    norm=math.sqrt(sum(x*x for x in vector))
    if not math.isclose(norm,1,rel_tol=.005,abs_tol=.005):raise ValueError("Embedding lacks expected L2 normalization")
    usage=value.get("usage",{})
    if (not isinstance(usage,dict) or type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"]<=0
            or type(usage.get("total_tokens")) is not int or usage["total_tokens"]!=usage["prompt_tokens"]):
        raise ValueError("Embedding raw usage missing or inconsistent")
    if type(usage.get("completion_tokens",0)) is not int or usage.get("completion_tokens",0)!=0:
        raise ValueError("Embedding output must not claim generated tokens")
    return {"dimension":len(vector),"l2_norm":norm,"usage":usage}


def body(response,timeout):
    sock=getattr(getattr(getattr(response,"fp",None),"raw",None),"_sock",None)
    if not isinstance(sock,socket.socket):raise ValueError("No deadline-controllable HTTP socket")
    expired=threading.Event()
    def expire():
        expired.set()
        try:sock.shutdown(socket.SHUT_RDWR)
        except OSError:pass
    timer=threading.Timer(timeout,expire);timer.daemon=True;timer.start()
    try:
        raw=response.read(LIMIT+1)
        if expired.is_set():raise ValueError("Embedding HTTP absolute body deadline")
        if len(raw)>LIMIT:raise ValueError("Embedding HTTP byte cap")
        return raw
    finally:timer.cancel();timer.join()


def prepared_image(path):
    # Same official image path as the successful offline pilot; no second resize.
    from qwen_vl_utils.vision_process import process_vision_info
    conv=[{"role":"user","content":[{"type":"image","image":Path(path).resolve().as_uri(),
        "min_pixels":262144,"max_pixels":1310720}]}]
    images,_=process_vision_info(conv,image_patch_size=16)
    image=images[0];buffer=io.BytesIO();image.save(buffer,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode(),list(image.size)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",default="http://127.0.0.1:30119")
    p.add_argument("--model",default="trident-qwen3vl-2b")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--log",type=Path,required=True)
    p.add_argument("--timeout",type=float,default=60)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv);base=base_url(a.base_url)
    if not math.isfinite(a.timeout) or not 0<a.timeout<=600:raise ValueError("timeout must be in(0,600]")
    plan={"api":"/v1/embeddings messages extension","roles":["query","doc"],"inputs":["text","image","image_plus_text"],
          "dry_run":a.dry_run,"http_gpu_verified":False,"official_accuracy_verified":False,"official_performance_verified":False}
    if a.dry_run:print(json.dumps(plan,indent=2));return 0
    image,size=prepared_image(a.image)
    requests=[("query_text",payload(a.model,"query","a red rectangle")),("doc_text",payload(a.model,"doc","a red rectangle")),
              ("doc_negative",payload(a.model,"doc","a blue circle")),("doc_image",payload(a.model,"doc",image=image)),
              ("doc_fused",payload(a.model,"doc","a red rectangle",image))]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with a.log.open("x",encoding="utf-8",newline="\n") as f:
        def record(value):f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
        def fetch(endpoint,request=None):
            try:
                with opener.open(request if request is not None else base+endpoint,timeout=a.timeout) as response:
                    return body(response,a.timeout)
            except urllib.error.HTTPError as exc:
                record({"endpoint":endpoint,"http_status":exc.code,"http_error":True})
                try:
                    # HTTPError.fp is the same HTTPResponse used by the success
                    # path; keep its absolute body timer and byte cap unchanged.
                    with exc:raw=body(exc.fp,a.timeout)
                    record({"endpoint":endpoint,"http_status":exc.code,"raw_error_body":raw.decode("utf-8",errors="replace"),
                            "raw_error_body_base64":base64.b64encode(raw).decode("ascii"),
                            "raw_error_body_sha256":hashlib.sha256(raw).hexdigest()})
                except Exception as capture_error:
                    record({"endpoint":endpoint,"http_status":exc.code,"error_body_capture_failed":str(capture_error)})
                raise
        record({"scope":plan,"source_image_sha256":hashlib.sha256(a.image.read_bytes()).hexdigest(),"prepared_image_size":size})
        try:
            for endpoint in ("/health","/v1/models"):
                raw=fetch(endpoint)
                record({"endpoint":endpoint,"raw_body":raw.decode()})
                if endpoint=="/v1/models" and a.model not in {x.get("id") for x in json.loads(raw).get("data",[])}:
                    raise ValueError("Served embedding model missing")
            for name,request in requests:
                record({"case":name,"request":request});started=time.monotonic()
                req=urllib.request.Request(base+"/v1/embeddings",json.dumps(request).encode(),{"Content-Type":"application/json"})
                raw=fetch("/v1/embeddings",req)
                record({"case":name,"raw_response":raw.decode()})
                validation=response_check(json.loads(raw),a.model)
                record({"case":name,"validation":validation,"wall_seconds":time.monotonic()-started})
            record({"functional_embedding_api_passed":True,"official_accuracy_verified":False,"official_performance_verified":False});return 0
        except Exception as exc:
            record({"functional_embedding_api_passed":False,"error":str(exc),"official_accuracy_verified":False});return 2


if __name__=="__main__":
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({"error":str(exc),"official_accuracy_verified":False}));raise SystemExit(2)
