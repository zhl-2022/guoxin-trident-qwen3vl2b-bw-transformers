#!/usr/bin/env python3
"""Bounded Trident embedding conversion pilot; no contest accuracy claim.

Copyright (c) 2026 the task author. No new open-source license selected.
Upstream code is not modified; publisher sources retain their original licenses.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import unicodedata

from task_expansion_candidates_20261023 import (
    digest, fixture_path, flags, json_new, read_json, validate_embeddings,
    verify_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/trident-qwen3vl2b-bw-transformers"
DEFAULT_MODEL = "/root/private_data/guoxin-candidates/models/trident"


def engine_kwargs(model_dir):
    return {
        "model": str(Path(model_dir).resolve()), "tokenizer": str(Path(model_dir).resolve()),
        "runner": "pooling", "convert": "embed", "dtype": "bfloat16",
        "tensor_parallel_size": 1, "max_model_len": 8192,
        "max_num_seqs": 4, "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.45, "enforce_eager": True,
        "enable_prefix_caching": False, "trust_remote_code": False,
        "hf_overrides": {"architectures": ["Qwen3VLForConditionalGeneration"],
                         "head_dtype": "float32", "use_cache": False},
        "pooler_config": {"seq_pooling_type": "LAST", "use_activation": True},
        "limit_mm_per_prompt": {"image": 2, "video": 0},
        "mm_processor_kwargs": {"do_resize": False}, "seed": 0,
    }


def inspect_header(path):
    with Path(path).open("rb") as handle:
        first = handle.read(8)
        if len(first) != 8:
            raise ValueError("Missing safetensors header length")
        size = struct.unpack("<Q", first)[0]
        if not 0 < size <= 2_000_000:
            raise ValueError("Unexpected safetensors header size")
        raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("Truncated safetensors header")
    header = json.loads(raw)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(tensors) != 625 or any(not k.startswith(("model.language_model.", "model.visual.")) for k in tensors):
        raise ValueError("Trident tensor names differ from researched checkpoint")
    required = {
        "model.language_model.embed_tokens.weight": [151936, 2048],
        "model.language_model.norm.weight": [2048],
        "model.visual.pos_embed.weight": [2304, 1024],
        "model.language_model.layers.27.self_attn.q_norm.weight": [128],
    }
    for name, shape in required.items():
        if tensors.get(name, {}).get("shape") != shape:
            raise ValueError("Unexpected tensor shape: " + name)
    if any(v.get("dtype") != "BF16" for v in tensors.values()):
        raise ValueError("Expected merged BF16 checkpoint")
    return {"tensor_count": len(tensors), "header_sha256": hashlib.sha256(raw).hexdigest(),
            "all_keys_match_qwen3_vl_mapper": True, "additional_projection": False,
            "lm_head_present": False, "full_weight_load_tested": False}


def validate_model_semantics(model_dir):
    base = Path(model_dir)
    config = read_json(base / "config.json")
    mmemb = read_json(base / "mmemb_model.json")
    if config.get("architectures") != ["Qwen3VLForEmbedding"] or config.get("model_type") != "qwen3_vl":
        raise ValueError("Unexpected original Trident architecture")
    if config["text_config"]["hidden_size"] != 2048 or config["vision_config"]["patch_size"] != 16:
        raise ValueError("Unexpected Trident dimensions")
    if mmemb.get("pooling") != "last_token" or mmemb.get("normalize") is not True or mmemb.get("embed_dim") is not None:
        raise ValueError("Conversion requires last_token + L2 and no projection")
    if mmemb.get("lora", {}).get("enable") or (base / "adapter_config.json").exists() or (base / "mmemb_extra.bin").exists():
        raise ValueError("Pilot only supports the researched merged checkpoint without extra weights")
    if mmemb.get("image") != {"min_pixels": 262144, "max_pixels": 1310720}:
        raise ValueError("Unexpected publisher image limits")
    return {"original_architecture": config["architectures"], "model_config_unchanged": True,
            "mmemb": mmemb, "header": inspect_header(base / "model.safetensors")}


def normalize_inputs(payload, request_file):
    clean = {}
    for group in ("queries", "documents"):
        rows = payload.get(group)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
            raise ValueError("Require 1..16 queries and documents")
        clean[group] = []
        for row in rows:
            if not isinstance(row, dict) or any(k not in {"text", "images"} for k in row):
                raise ValueError("Pilot accepts only text and local images; no silent video/instruction fallback")
            text, images = row.get("text"), row.get("images", [])
            if text is not None and not isinstance(text, str):
                raise ValueError("text must be string or null")
            if not isinstance(images, list) or len(images) > 2:
                raise ValueError("At most two local images")
            if not images and not (text and text.strip()):
                raise ValueError("Empty embedding input")
            paths = [str(fixture_path(x, request_file)) for x in images]
            clean[group].append({"text": text, "images": paths})
    return clean


def conversation(row, role, tokenizer, mmemb):
    instruction = mmemb["prompt"]["query_instruction" if role == "query" else "doc_instruction"].strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    content = [{"type": "image", "image": Path(image).as_uri(), **mmemb["image"]} for image in row["images"]]
    text = row["text"]
    if text:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > mmemb["max_text_tokens"]:
            text = tokenizer.decode(ids[:mmemb["max_text_tokens"]], skip_special_tokens=True)
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}]


def build_prompts(payload, processor, vision_fn, mmemb):
    requests, audit = [], []
    for group, role in (("queries", "query"), ("documents", "doc")):
        for index, row in enumerate(payload[group]):
            conv = conversation(row, role, processor.tokenizer, mmemb)
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            images = None
            if row["images"]:
                images, videos, video_kwargs = vision_fn(
                    conv, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
                if videos or video_kwargs.get("fps"):
                    raise ValueError("Video not part of this text/image/fused pilot")
            kwargs = {"text": [prompt], "padding": True, "return_tensors": "pt", "do_resize": False}
            if images:
                kwargs["images"] = images
            else:
                kwargs.update(truncation=True, max_length=mmemb["max_length"])
            tokens = processor(**kwargs)["input_ids"][0].tolist()
            if len(tokens) > 8192:
                raise ValueError("Pilot prompt exceeds 8192 tokens; no multimodal truncation")
            item = {"prompt": prompt}
            if images:
                item["multi_modal_data"] = {"image": images}
            requests.append(item)
            audit.append({"group": group, "index": index, "role": role, "prompt": prompt,
                          "expected_prompt_token_ids": tokens,
                          "image_sha256": [digest(p) for p in row["images"]],
                          "resized_image_sizes": [list(x.size) for x in images] if images else []})
    return requests, audit


def reference_provenance(path):
    source = Path(path).resolve()
    artifacts = {}
    for name in (source.name, "checkpoint-verification.json", "run-start.json", "environment.json",
                 "run-final.json", "prompt-audit.json"):
        item = source.parent / name
        if item.is_file():
            raw = item.read_bytes()
            if len(raw) > 10_000_000:
                raise ValueError("Reference JSON artifact exceeds pilot limit")
            artifacts[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "content": json.loads(raw)}
    checkpoint = artifacts.get("checkpoint-verification.json", {}).get("content", {})
    lock = read_json(TASK / "model-lock.json")
    if checkpoint and (checkpoint.get("model_id") != lock["model_id"] or checkpoint.get("revision") != lock["revision"]
                       or checkpoint.get("complete") is not True):
        raise ValueError("Reference checkpoint identity differs or did not pass")
    if source.name not in artifacts:
        raise ValueError("Reference artifact missing")
    return {"reference_path": str(source), "reference_sha256": artifacts[source.name]["sha256"], "artifacts": artifacts,
            "checkpoint_identity_verified": bool(checkpoint),
            "reference_input_image_bytes_at_runtime_verified": False,
            "reference_source_identity_at_runtime_verified": False,
            "unverified_reason": "Frozen publisher runner did not store per-image runtime digests or a source-verification artifact; paths alone do not prove identical image bytes",
            "expected_publisher_source_revision": "7fbf4b2033e1b3a43a99185fc125d765a549c52d"}


def compare_reference(embeddings, reference, inputs):
    if reference.get("mode") != "publisher_embedding_backend" or reference.get("inputs") != inputs:
        raise ValueError("Reference must be publisher TF result for exactly the same local inputs")
    other = reference["embeddings"]
    comparisons = []
    for group in ("queries", "documents"):
        if len(embeddings[group]) != len(other[group]):
            raise ValueError("Reference row count differs")
        for i, (left, right) in enumerate(zip(embeddings[group], other[group])):
            if len(left) != 2048 or len(right) != 2048 or any(not math.isfinite(x) for x in left + right):
                raise ValueError("Invalid reference embedding")
            cosine = sum(a*b for a,b in zip(left,right)) / math.sqrt(sum(x*x for x in left)*sum(x*x for x in right))
            comparisons.append({"group": group, "index": i, "cosine": cosine,
                                "max_abs_difference": max(abs(a-b) for a,b in zip(left,right))})
    return {"comparisons": comparisons, "engineering_threshold_cosine": 0.999,
            "engineering_agreement_passed": all(x["cosine"] >= 0.999 for x in comparisons),
            "contest_accuracy_measurement": False}


def worker(args, output):
    check = verify_checkpoint(args.model_dir, read_json(TASK / "model-lock.json"), True)
    json_new(output / "checkpoint-verification.json", check)
    if not check["complete"]:
        raise ValueError("Checkpoint identity preflight failed")
    semantics = validate_model_semantics(args.model_dir)
    json_new(output / "model-semantics.json", semantics)
    payload = normalize_inputs(read_json(args.requests), args.requests)
    provenance = reference_provenance(args.reference_result) if args.reference_result else None
    reference = None
    if provenance:
        reference = provenance["artifacts"][Path(args.reference_result).name]["content"]
        if reference.get("mode") != "publisher_embedding_backend" or reference.get("inputs") != payload:
            raise ValueError("Reference must contain publisher results for exactly the same local inputs")
        json_new(output / "reference-provenance.json", provenance)
    if importlib.metadata.version("vllm").split("+")[0] != "0.18.1":
        raise ValueError("Probe researched against vLLM 0.18.1; inspect installed source before another version")
    if importlib.metadata.version("qwen-vl-utils") != "0.0.14":
        raise ValueError("Use the locked native-first Trident environment with qwen-vl-utils 0.0.14")
    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils.vision_process import process_vision_info
    from vllm import LLM, PoolingParams
    packages = {}
    for name in ("torch", "torchvision", "vllm", "vllm-hcu", "transformers", "qwen-vl-utils", "Pillow"):
        try:
            distribution = importlib.metadata.distribution(name)
            packages[name] = {"version": distribution.version, "root": str(distribution.locate_file(""))}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    json_new(output / "environment.json", {"python": sys.executable, "distributions": packages,
                                           "package_installation_performed": False})
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    processor.tokenizer.padding_side = "right"
    prompts, audit = build_prompts(payload, processor, process_vision_info, semantics["mmemb"])
    json_new(output / "prompt-audit.json", audit)
    options = engine_kwargs(args.model_dir)
    json_new(output / "engine-options.json", options)
    started = time.monotonic()
    model = LLM(**options)
    embeddings = {"queries": [], "documents": []}
    identity = []
    # One item at a time keeps the pilot's memory and scheduling easy to audit.
    for item, record in zip(prompts, audit):
        values = model.embed([item], pooling_params=PoolingParams(use_activation=True), use_tqdm=False)
        if len(values) != 1:
            raise ValueError("Unexpected embedding output count")
        vector = list(values[0].outputs.embedding)
        embeddings[record["group"]].append(vector)
        actual = list(values[0].prompt_token_ids)
        identity.append({"group": record["group"], "index": record["index"],
                         "token_ids_identical_to_publisher_processor": actual == record["expected_prompt_token_ids"],
                         "actual_prompt_token_ids": actual})
    validation = validate_embeddings(embeddings["queries"], embeddings["documents"])
    result = {"mode": "vllm_qwen3vl_embed_conversion_pilot", "inputs": payload,
              "embeddings": embeddings, "validation": validation, "prompt_token_identity": identity,
              "wall_seconds_including_load": time.monotonic()-started,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "numerical_reference_available": bool(args.reference_result), **flags()}
    if args.reference_result:
        result["reference_comparison"] = compare_reference(embeddings, reference, payload)
        result["reference_provenance"] = provenance
    result["development_contract_passed"] = (
        all(x["token_ids_identical_to_publisher_processor"] for x in identity)
        and validation["dimension"] == 2048
        and result.get("reference_comparison", {}).get("engineering_agreement_passed", True))
    json_new(output / "worker-result.json", result)
    return 0 if result["development_contract_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--requests", default=str(TASK / "requests.example.json"))
    parser.add_argument("--output")
    parser.add_argument("--reference-result")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps({"engine_kwargs": engine_kwargs(args.model_dir),
                          "model_lock": read_json(TASK / "model-lock.json"),
                          "status": "static_researched_gpu_unverified", **flags()}, indent=2))
        return 0
    if args.action == "preflight":
        check = verify_checkpoint(args.model_dir, read_json(TASK / "model-lock.json"), args.hash_weights)
        result = {"checkpoint": check, **flags()}
        if check["complete"]:
            result["semantics"] = validate_model_semantics(args.model_dir)
        print(json.dumps(result, indent=2))
        return 0 if check["complete"] else 2
    if sys.platform != "linux" or not 60 <= args.max_seconds <= 1800:
        parser.error("run requires Linux and max-seconds 60..1800")
    if not args.output or not Path(args.output).is_absolute():
        parser.error("run requires NEW absolute output directory")
    output = Path(args.output)
    if args._worker:
        try:
            return worker(args, output)
        except Exception as exc:
            json_new(output / "worker-error.json", {"type": type(exc).__name__, "message": str(exc), **flags()})
            raise
    if output.exists() or output.is_symlink():
        parser.error("Output exists; choose a new run directory")
    output.mkdir(parents=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--_worker",
               "--model-dir", args.model_dir, "--requests", args.requests, "--output", str(output)]
    if args.reference_result:
        command.extend(["--reference-result", args.reference_result])
    json_new(output / "run-start.json", {"argv": command, "max_seconds": args.max_seconds,
        "deadline_includes_checkpoint_hash_semantics_inputs_imports_and_gpu_worker": True, **flags()})
    child, status, started = None, "failed", time.monotonic()
    try:
        with (output / "worker.log").open("xb") as log:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
            try:
                code = child.wait(timeout=args.max_seconds)
                status = "completed" if code == 0 and (output / "worker-result.json").is_file() else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    time.sleep(2)
            child.wait(timeout=10)
        json_new(output / "run-final.json", {"status": status, "wall_seconds": time.monotonic()-started,
                                           "worker_exit_code": None if child is None else child.returncode,
                                           "owned_process_group_only": True, **flags()})
    print(json.dumps({"status": status, "output": str(output), **flags()}))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
