#!/usr/bin/env python3
"""Trident-specific configuration for the existing bounded vLLM supervisor.

Copyright (c) 2026 the task author. Copyright scope only; no new license chosen.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import recon2reason_runtime as supervisor
from task_expansion_candidates_20261023 import verify_checkpoint, read_json
from trident_vllm_pooling_probe_20261023 import engine_kwargs, validate_model_semantics


def configuration(task):
    model_value = os.environ.get("TRIDENT_MODEL_PATH", "/root/private_data/guoxin-candidates/models/trident")
    if not model_value:
        raise ValueError("TRIDENT_MODEL_PATH must name a local checkpoint directory")
    # Resolve once in the caller's cwd. The vLLM child later runs in task cwd,
    # so both preflight and --model/--tokenizer must use this absolute identity.
    model = str(Path(model_value).expanduser().resolve())
    options = engine_kwargs(model)
    port = int(os.environ.get("TRIDENT_PORT", "30119"))
    memory = float(os.environ.get("TRIDENT_MEM_FRAC", "0.45"))
    load = int(os.environ.get("TRIDENT_LOAD_TIMEOUT", "600"))
    runtime = int(os.environ.get("TRIDENT_MAX_RUNTIME_SECONDS", "900"))
    served = os.environ.get("TRIDENT_SERVED_NAME", "trident-qwen3vl-2b")
    if not 1024 <= port <= 65535 or not 0 < memory < 1 or not 60 <= load <= runtime <= 7200:
        raise ValueError("Invalid embedding port/memory/load/runtime bounds")
    if not served or any(x in served for x in "\r\n"):
        raise ValueError("Invalid served name")
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
        "--tokenizer", model, "--served-model-name", served, "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "8192",
        "--gpu-memory-utilization", str(memory), "--max-num-seqs", "4", "--max-num-batched-tokens", "8192",
        "--runner", "pooling", "--convert", "embed", "--enforce-eager", "--no-enable-prefix-caching", "--seed", "0"]
    for key in ("hf_overrides", "pooler_config", "limit_mm_per_prompt", "mm_processor_kwargs"):
        command.extend(["--"+key.replace("_","-"), json.dumps(options[key],separators=(",",":"))])
    return {"task":str(task),"framework":"vllm","precision":"bf16","model_id":"openbmb/Trident-Qwen3VL-2B",
        "model_path":model,"served_name":served,"host":"127.0.0.1","port":port,"tp":1,"max_len":8192,
        "mem_frac":memory,"load_timeout":load,"max_runtime":runtime,"stop_grace":30,"command":command,
        "url":f"http://127.0.0.1:{port}","api":"/v1/embeddings messages extension",
        "context_scope":"8K development configuration; no generation workload inheritance",
        "http_gpu_verified":False,"official_accuracy_verified":False,"official_performance_verified":False}


def official_plan(task, action, dry_run):
    # The shared T02 bridge prescribes generative 4K/16K ->1024-token workloads.
    # Until an embedding-specific official schema is supplied, do not invoke it.
    print(json.dumps({"status":"plan_only" if dry_run else "blocked_official_embedding_contract_missing",
        "task":str(task),"action":action,"executed":False,"exit_reason":"Provide embedding-specific evaluator/selfcheck, subset/qrels, GPU baseline and output/metric schema before integration",
        "missing":["official_embedding_evaluator_schema","fixed_retrieval_subset_and_qrels","GPU_baseline","embedding_performance_workload"],
        "official_accuracy_verified":False,"official_performance_verified":False,"submission_ready":False},indent=2))
    return 0 if dry_run else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("launch","stop","env","verify","bench"))
    parser.add_argument("--task-dir",type=Path,required=True)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--selfcheck",action="store_true")
    parser.add_argument("--run-id")
    args=parser.parse_args(argv)
    task=args.task_dir.resolve()
    if task.name != "trident-qwen3vl2b-bw-transformers" or read_json(task/"task.json")["model_id"] != "openbmb/Trident-Qwen3VL-2B":
        raise ValueError("This entry is only for the recorded Trident task")
    if args.action in ("verify","bench"):
        return official_plan(task,args.action,args.dry_run)
    if args.action=="stop":
        if args.dry_run:
            print(json.dumps({"status":"plan_only","task":str(task),"action":"stop",
                "run_id":args.run_id,"executed":False,"process_signalled":False},indent=2))
            return 0
        return supervisor.stop(task,args.run_id)
    config=configuration(task)
    if args.action=="env":
        versions={}
        for name in ("torch","torchvision","vllm","vllm-hcu","transformers","qwen-vl-utils","Pillow"):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]=None
        print(json.dumps({"python":sys.executable,"packages":versions,"configuration":config,
            "metadata_only":True,"GPU_inference_tested":False,"actual_container_registry_ref":None},indent=2))
        return 0
    if not args.dry_run:
        if sys.platform!="linux":
            return supervisor.fail("Actual embedding serving requires the SCNet Linux container")
        check=verify_checkpoint(config["model_path"],read_json(task/"model-lock.json"),True)
        if not check["complete"]:
            print(json.dumps({"checkpoint_identity_failed":check},indent=2));return 2
        validate_model_semantics(config["model_path"])
    # Reuse the established launch/readiness/ownership/stop implementation.
    # Configuration specialization is confined to this dedicated CLI process;
    # the shared source file and its separate supervisor process are unchanged.
    original=supervisor.configuration
    # Reuse the exact identity checked above rather than resolving env/cwd again.
    supervisor.configuration=lambda _task:config
    try:return supervisor.launch(task,args.dry_run,args.run_id)
    finally:supervisor.configuration=original


if __name__=="__main__":
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError) as exc:
        print(json.dumps({"error":str(exc),"submission_ready":False}));raise SystemExit(2)
