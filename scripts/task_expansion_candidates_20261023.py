#!/usr/bin/env python3
"""Offline planning and explicitly bounded execution for three new candidates.

Copyright (c) 2026 the task author. No repository-wide open-source license chosen.
Publisher source archives retain their own LICENSE files and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SLUGS = ("qwen3-forcedaligner-bw-transformers", "mimo-v2.6-distill-qwen9b-bw-vllm", "trident-qwen3vl2b-bw-transformers")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def flags():
    return {"official_accuracy_verified": False, "official_performance_verified": False,
            "submission_ready": False, "contest_baseline_available": False}


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(x in {"", ".", ".."} for x in name.split("/")) or ":" in name:
        raise ValueError("Unsafe archive or manifest path: " + name)
    return path


def verify_checkpoint(model_dir, lock, hash_weights=False):
    base = Path(model_dir).resolve()
    rows = []
    if not base.is_dir():
        return {"complete": False, "directory": str(base), "error": "checkpoint_directory_missing", "files": []}
    for item in lock["files"]:
        relative = safe_relative(item["path"])
        path = base.joinpath(*relative.parts)
        row = {"path": item["path"], "size_ok": False, "identity_ok": False}
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base):
            row["error"] = "missing_or_unsafe_file"
        elif path.stat().st_size != item["size"]:
            row["error"] = "file_size_differs_from_publisher"
        else:
            row["size_ok"] = True
            if "sha256" in item:
                if hash_weights or not path.name.endswith(".safetensors"):
                    row["sha256"] = digest(path)
                    row["identity_ok"] = row["sha256"] == item["sha256"]
                else:
                    row["error"] = "large_weight_hash_not_requested"
            else:
                value = hashlib.sha1()
                value.update(f"blob {item['size']}\0".encode("ascii"))
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        value.update(block)
                row["git_blob_sha1"] = value.hexdigest()
                row["identity_ok"] = row["git_blob_sha1"] == item["git_blob_sha1"]
        rows.append(row)
    return {"complete": bool(rows) and all(x["identity_ok"] for x in rows),
            "model_id": lock["model_id"], "revision": lock["revision"],
            "directory": str(base), "weights_hashed": hash_weights, "files": rows}


def verify_source(source_root, lock):
    base = Path(source_root).resolve()
    rows = []
    for item in lock["files"]:
        relative = safe_relative(item["path"])
        path = base.joinpath(*relative.parts)
        okay = (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
                and path.stat().st_size == item["size"] and digest(path) == item["sha256"])
        rows.append({"path": item["path"], "verified": okay})
    expected = {item["path"] for item in lock["files"]}
    extra_executable_sources = [str(path.relative_to(base)).replace("\\", "/") for path in base.rglob("*")
                                if path.is_file() and path.suffix in {".py", ".so", ".pyd"}
                                and str(path.relative_to(base)).replace("\\", "/") not in expected]
    return {"complete": bool(rows) and all(x["verified"] for x in rows) and not extra_executable_sources,
            "revision": lock["revision"], "files": rows, "extra_executable_sources": extra_executable_sources}


def extract_source(task, destination):
    lock = read_json(task / "upstream-lock.json")
    archive = task / lock["archive_file"]
    if digest(archive) != lock["archive_sha256"]:
        raise ValueError("Bundled publisher source archive hash differs")
    dest = Path(destination)
    if not dest.is_absolute() or dest.exists() or dest.is_symlink():
        raise ValueError("Source extraction requires a NEW absolute directory")
    expected = {item["path"]: item for item in lock["files"]}
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("Source archive member set differs")
        for name in names:
            safe_relative(name)
            item = source.getinfo(name)
            if item.file_size != expected[name]["size"] or item.file_size > 20_000_000:
                raise ValueError("Source archive member size differs")
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != expected[name]["sha256"]:
                raise ValueError("Source archive member digest differs")
        dest.mkdir(parents=True)
        for name in names:
            path = dest.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(source.read(name))
    result = verify_source(dest, lock)
    if not result["complete"]:
        raise ValueError("Extracted source failed read-back")
    return result


def audio_contract(path):
    source = Path(path)
    with wave.open(str(source), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16000 or audio.getcomptype() != "NONE":
            raise ValueError("Candidate fixture contract is mono 16 kHz PCM16 WAV")
        if not 0 < duration <= 180:
            raise ValueError("ForcedAligner pilot audio must be within (0, 180] seconds, matching the pinned wrapper utility limit")
    return {"path": str(source.resolve()), "duration_seconds": duration, "sha256": digest(source)}


def fixture_path(value, request_file):
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError("Fixture must be an existing local file")
    path = Path(value)
    if not path.is_absolute():
        path = Path(request_file).resolve().parent / path
    if not path.is_file():
        raise ValueError("Fixture file missing: " + str(path))
    return path.resolve()


def normalized_text(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value) if c.isalnum() or c == "'").casefold()


def validate_alignment(items, text, duration):
    if not isinstance(items, list) or not items:
        raise ValueError("Empty alignment")
    previous_start = previous_end = 0.0
    for item in items:
        if not isinstance(item.get("text"), str) or not normalized_text(item["text"]):
            raise ValueError("Alignment item text missing")
        start, end = item.get("start_time"), item.get("end_time")
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in (start, end)):
            raise ValueError("Alignment timestamp is not finite numeric")
        if not (0 <= start <= end <= duration + 0.1) or start < previous_start or end < previous_end:
            raise ValueError("Alignment timestamps outside audio or out of order")
        previous_start, previous_end = start, end
    if normalized_text("".join(item["text"] for item in items)) != normalized_text(text):
        raise ValueError("Alignment lost or changed source text units")
    return {"item_count": len(items), "text_units_preserved": True, "timestamps_finite_monotonic_in_audio": True,
            "accuracy_against_independent_timestamps_evaluated": False}


def validate_embeddings(queries, documents):
    arrays = (queries, documents)
    dimension = None
    for rows in arrays:
        if not isinstance(rows, list) or not rows:
            raise ValueError("Embedding batch empty")
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ValueError("Embedding vector empty")
            if dimension is None:
                dimension = len(row)
            if len(row) != dimension or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
                raise ValueError("Embedding dimensions/types/finiteness invalid")
            if not math.isclose(math.sqrt(sum(v * v for v in row)), 1.0, rel_tol=0.005, abs_tol=0.005):
                raise ValueError("Publisher L2 normalization missing")
    scores = [[sum(a * b for a, b in zip(query, doc)) for doc in documents] for query in queries]
    return {"dimension": dimension, "query_count": len(queries), "document_count": len(documents),
            "cosine_scores": scores, "ranked_document_indices": [sorted(range(len(row)), key=lambda i: (-row[i], i)) for row in scores],
            "retrieval_accuracy_against_qrels_evaluated": False}


def validate_mimo_response(response, served_name):
    if response.get("error") or response.get("model") != served_name:
        raise ValueError("MiMo response error or model mismatch")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("MiMo generation missing, truncated, or incomplete")
    message = choices[0].get("message", {})
    if not isinstance(message.get("content"), str) or not message["content"].strip():
        raise ValueError("MiMo visible text empty")
    usage = response.get("usage", {})
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(name)) is not int or usage[name] <= 0:
            raise ValueError("MiMo token usage missing")
    if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        raise ValueError("MiMo token accounting differs")
    return {"nonempty_visible_text": True, "finish_reason": "stop", "usage": usage,
            "raw_thinking_preserved": True, "reasoning_parser": None, "answer_accuracy_evaluated": False}


def request_json(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("HTTP response exceeded candidate limit")
    return json.loads(raw)


def terminate_owned(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def worker(args, task, output):
    # Only the bounded subprocess enters this branch; plan/preflight never imports Torch.
    versions = {}
    profile = read_json(task / "runtime.json")
    for name in set(profile["python_distributions"] + profile["native_packages_must_remain_unchanged"]):
        try:
            distribution = importlib.metadata.distribution(name)
            versions[name] = {"version": distribution.version, "root": str(distribution.locate_file(""))}
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    json_new(output / "environment.json", {"python": sys.version, "executable": sys.executable, "distributions": versions})
    if args.task != SLUGS[1]:
        source = Path(args.upstream_root).resolve()
        check = verify_source(source, read_json(task / "upstream-lock.json"))
        if not check["complete"]:
            raise ValueError("Publisher source differs from the pinned snapshot")
        sys.path.insert(0, str(source if args.task == SLUGS[0] else source / "src"))
    if args.task == SLUGS[0]:
        import inspect
        import torch
        from qwen_asr import Qwen3ForcedAligner
        expected_api = Path(args.upstream_root) / "qwen_asr/inference/qwen3_forced_aligner.py"
        if Path(inspect.getfile(Qwen3ForcedAligner)).resolve() != expected_api.resolve():
            raise ValueError("Imported ForcedAligner API is not the selected pinned source")
        requests = read_json(args.requests)
        if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
            raise ValueError("Require 1..16 audio/text requests")
        fixtures = []
        for item in requests:
            if not isinstance(item.get("text"), str) or not normalized_text(item["text"]):
                raise ValueError("Nonempty known transcript required")
            if item.get("language") not in {"Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish", "Cantonese"}:
                raise ValueError("Explicit publisher-supported language required")
            fixtures.append(audio_contract(fixture_path(item["audio"], args.requests)))
        model = Qwen3ForcedAligner.from_pretrained(str(Path(args.model_dir).resolve()), dtype=torch.bfloat16,
                                                  device_map="cuda:0", attn_implementation="sdpa")
        results = []
        for item, fixture in zip(requests, fixtures):
            started = time.monotonic()
            batch = model.align(audio=fixture["path"], text=item["text"], language=item["language"])
            if len(batch) != 1:
                raise ValueError("ForcedAligner batch shape differs")
            values = [{"text": span.text, "start_time": span.start_time, "end_time": span.end_time} for span in batch[0]]
            results.append({"input": item, "fixture": fixture, "items": values,
                            "validation": validate_alignment(values, item["text"], fixture["duration_seconds"]),
                            "wall_seconds": time.monotonic() - started})
        return {"mode": "publisher_transformers_sdpa_alignment", "results": results}
    if args.task == SLUGS[2]:
        from embedding_backends.common import ensure_tokenizers_thread_safety, ensure_transformers_special_tokens_compat
        from embedding_backends.trident_qwen3vl import TridentQwen3VLBackend
        ensure_transformers_special_tokens_compat()
        ensure_tokenizers_thread_safety()
        payload = read_json(args.requests)
        for role in ("queries", "documents"):
            if not isinstance(payload.get(role), list) or not 1 <= len(payload[role]) <= 16:
                raise ValueError("Require 1..16 queries and 1..16 documents")
            for item in payload[role]:
                images = item.get("images", [])
                if not isinstance(images, list) or len(images) > 2:
                    raise ValueError("Trident candidate accepts at most two existing local images per input")
                item["images"] = [str(fixture_path(p, args.requests)) for p in images]
                if not images and not (isinstance(item.get("text"), str) and item["text"].strip()):
                    raise ValueError("Text, local image, or fused input required")
        backend = TridentQwen3VLBackend({"model_type": "trident_qwen3vl", "checkpoint": str(Path(args.model_dir).resolve()),
                                         "dtype": "bfloat16"}, device="cuda:0")
        backend.load()
        embeddings = {}
        for name, role in (("queries", "query"), ("documents", "doc")):
            backend.args_dict["role"] = role
            inputs = [{**item, "instruction": backend.default_instruction(role)} for item in payload[name]]
            values, _ = backend.process(inputs, role=role)
            embeddings[name] = values.tolist()
        return {"mode": "publisher_embedding_backend", "inputs": payload, "embeddings": embeddings,
                "validation": validate_embeddings(embeddings["queries"], embeddings["documents"])}
    served_name = "mimo-v2.6-distill-qwen-9b-candidate"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(Path(args.model_dir).resolve()),
               "--served-model-name", served_name, "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "8192",
               "--max-num-seqs", "4", "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.80", "--enforce-eager"]
    json_new(output / "server-command.json", {"argv": command, "reasoning_parser": None,
        "tokenizer_replaced": False, "checkpoint_generation_config_preserved": True})
    started = time.monotonic()
    child = None
    try:
        with (output / "server.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + min(args.max_seconds - 30, 600)
            while True:
                if child.poll() is not None:
                    raise RuntimeError("MiMo vLLM exited before readiness")
                try:
                    models = request_json(f"http://127.0.0.1:{args.port}/v1/models", timeout=3)
                    if any(x.get("id") == served_name for x in models.get("data", [])):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("MiMo load/readiness deadline")
                time.sleep(1)
            payload = {"model": served_name, "messages": [{"role": "user", "content": "What is 15% of 240?"}],
                       "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": True}}
            json_new(output / "request.json", payload)
            response = request_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", payload,
                                    timeout=max(1, args.max_seconds - (time.monotonic() - started) - 15))
            json_new(output / "response.json", response)
            validation = validate_mimo_response(response, served_name)
        return {"mode": "vllm_native_checkpoint_chat_template", "wall_seconds": time.monotonic() - started,
                "validation": validation, "formal_sampling_config_available": False}
    finally:
        if child is not None:
            terminate_owned(child)
            json_new(output / "server-cleanup.json", {"owned_pid": child.pid, "exited": child.poll() is not None,
                                                      "exit_code": child.returncode, "unrelated_processes_signalled": False})


def plan(task, args):
    profile = read_json(task / "runtime.json")
    lock = read_json(task / "model-lock.json")
    result = {"mode": args.action, "task": args.task, "model_id": lock["model_id"], "revision": lock["revision"],
              "weight_bytes": lock["weight_bytes"], "runtime_profile": profile,
              "downloads_performed": False, "gpu_imported": False, "service_started": False, **flags()}
    if args.action == "preflight":
        result["checkpoint"] = verify_checkpoint(args.model_dir, lock, args.hash_weights)
        result["distributions"] = {}
        for name in profile["python_distributions"]:
            try:
                result["distributions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["distributions"][name] = None
        if (task / "upstream-lock.json").is_file():
            result["publisher_source"] = verify_source(args.upstream_root, read_json(task / "upstream-lock.json")) if args.upstream_root else {"complete": False, "error": "source_root_not_provided"}
        result["file_and_distribution_inventory_complete"] = (result["checkpoint"]["complete"]
            and all(value is not None for value in result["distributions"].values())
            and result.get("publisher_source", {"complete": True})["complete"])
        result["dependency_import_compatibility_tested"] = False
    return result


def main(fixed_task=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_task is None:
        parser.add_argument("task", choices=SLUGS)
    parser.add_argument("action", choices=("plan", "preflight", "prepare-source", "run"))
    parser.add_argument("--model-dir", default="/root/private_data/guoxin-candidates/models/REPLACE_WITH_TASK_CHECKPOINT")
    parser.add_argument("--upstream-root")
    parser.add_argument("--requests", help="Local JSON fixtures; audio/image paths must exist on execution host")
    parser.add_argument("--output", help="New absolute run directory for run; no overwrite")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--port", type=int, default=30109)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if fixed_task is not None:
        args.task = fixed_task
    task = ROOT / "tasks" / args.task
    if args.action in ("plan", "preflight"):
        result = plan(task, args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if args.action == "plan" or result["file_and_distribution_inventory_complete"] else 2
    if args.action == "prepare-source":
        if not args.upstream_root:
            parser.error("prepare-source requires --upstream-root NEW_ABSOLUTE_DIRECTORY")
        print(json.dumps(extract_source(task, args.upstream_root), ensure_ascii=False))
        return 0
    if sys.platform != "linux" or not 60 <= args.max_seconds <= 1800 or not 1024 <= args.port <= 65535:
        parser.error("GPU run requires Linux, max-seconds 60..1800, port 1024..65535")
    if not args.output or not Path(args.output).is_absolute():
        parser.error("GPU run requires --output NEW_ABSOLUTE_DIRECTORY")
    output = Path(args.output)
    if args._worker:
        result = worker(args, task, output)
        json_new(output / "worker-result.json", {"development_contract_passed": True, **result, **flags()})
        return 0
    if args.task != SLUGS[1] and (not args.upstream_root or not args.requests):
        parser.error("This task requires --upstream-root PINNED_SOURCE --requests LOCAL_FIXTURES_JSON")
    if output.exists() or output.is_symlink():
        parser.error("Run output directory already exists")
    check = verify_checkpoint(args.model_dir, read_json(task / "model-lock.json"), True)
    if not check["complete"]:
        print(json.dumps({"error": "checkpoint_identity_preflight_failed", "checkpoint": check, **flags()}, indent=2))
        return 2
    output.mkdir(parents=True)
    json_new(output / "checkpoint-verification.json", check)
    run_id = str(uuid.uuid4())
    command = [sys.executable, "-B", str(Path(__file__).resolve()), args.task, "run", "--_worker", "--model-dir", args.model_dir,
               "--output", str(output), "--max-seconds", str(args.max_seconds), "--port", str(args.port)]
    if args.upstream_root:
        command += ["--upstream-root", args.upstream_root]
    if args.requests:
        command += ["--requests", args.requests]
    json_new(output / "run-start.json", {"run_id": run_id, "command": command, "max_seconds": args.max_seconds, **flags()})
    child = None
    status = "failed"
    started = time.monotonic()
    try:
        with (output / "worker.log").open("xb") as log:
            environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            try:
                code = child.wait(timeout=args.max_seconds)
                status = "completed" if code == 0 and (output / "worker-result.json").is_file() else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        # The process group was created exclusively for this worker and its model children.
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    time.sleep(2)
            child.wait(timeout=10)
        json_new(output / "run-final.json", {"run_id": run_id, "status": status,
            "wall_seconds": time.monotonic() - started, "worker_exit_code": None if child is None else child.returncode,
            "owned_process_group_signalled_only": True, **flags()})
    print(json.dumps({"run_id": run_id, "status": status, "output": str(output), **flags()}))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
