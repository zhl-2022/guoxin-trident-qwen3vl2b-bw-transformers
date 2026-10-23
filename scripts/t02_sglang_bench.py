#!/usr/bin/env python3
"""Bounded, archived SGLang 0.5.12 client against an existing local vLLM server.

This executes the upstream client, not the server. Template load coverage is not
official accuracy/acceptance. Native output_lens can fall back to requested
lengths when the stream has no usage, so exact output length is not certified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

VERSION = "0.5.12"
SOURCE = "https://github.com/sgl-project/sglang/tree/v0.5.12/python/sglang"
CLIENT_BACKEND = "sglang-oai-chat"
SERVER_FRAMEWORK = "vllm"
BACKEND_BOUNDARY = ("SGLang0.5.12 image dataset permits sglang-oai-chat, not vllm-chat; both aliases map to the same upstream OpenAI chat sender. "
                    "The sender alias does not identify the server framework or create a SGLang-serving contest piece.")
LOADS = [(n, c) for n in (4096, 16384) for c in (1, 8, 64)]
SOURCE_FILES = ("__init__.py", "bench_serving.py", "benchmark/datasets/image.py",
                "benchmark/datasets/common.py", "benchmark/utils.py")
LENGTH_BOUNDARY = "upstream output_lens may use stream usage or requested-length fallback; exact 1024-token output is not independently certified"
VISIBILITY_VARIABLES = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def device_discovery_policy(allow, environment):
    """Record visibility without importing Torch, enumerating or widening devices."""
    caller = {name: {"is_set": name in environment, "value": environment.get(name),
                     "nonempty": bool(environment.get(name))} for name in VISIBILITY_VARIABLES}
    effective = {name: (dict(caller[name]) if allow else {"is_set": True, "value": "", "nonempty": False})
                 for name in VISIBILITY_VARIABLES}
    return {"allowed": bool(allow), "policy": "preserve_caller_exactly" if allow else "hide_all_client_devices",
            "compatibility_reason": "Opt-in DAS client import compatibility: native Triton autotuner can query the visible device label before benchmark CLI startup",
            "caller_visibility": caller, "effective_visibility": effective,
            "caller_visibility_preserved_exactly": effective == caller,
            "device_scope_expanded_by_driver": False, "driver_device_enumeration_performed": False,
            "runtime_device_availability_verified": False, "client_model_inference_requested": False}


def client_environment(allow, environment):
    env = {k: v for k, v in environment.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
           and k not in {"OPENAI_API_KEY", "API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
    env.update(NO_PROXY="127.0.0.0/8,::1,localhost", no_proxy="127.0.0.0/8,::1,localhost", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not allow:
        env.update({name: "" for name in VISIBILITY_VARIABLES})
    return env


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--client-python", type=Path, default=Path(sys.executable))
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--served-model-name", required=True)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--output-dir", type=Path, default=Path("evidence/session004/t02-sglang"))
    p.add_argument("--timeout-seconds", type=float, default=1800)
    p.add_argument("--total-timeout-seconds", type=float, default=6400)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--load", action="append", choices=[f"{n}:{c}" for n, c in LOADS])
    p.add_argument("--round", action="append", type=int, choices=(1, 2), dest="rounds")
    p.add_argument("--runtime-dir", type=Path, help="existing owned supervisor runtime; archive only, never stop or mutate it")
    p.add_argument("--expected-run-id", help="exact32 lowercase hex ID; required with --runtime-dir")
    p.add_argument("--client-command-json", help="CPU/mock override argv; explicitly marked unverified, never upstream client evidence")
    p.add_argument("--allow-client-device-discovery", action="store_true",
                   help="DAS import compatibility only: preserve caller CUDA/HIP/ROCR visibility exactly, including unset/empty; default hides client devices")
    return p


def build_plan(a):
    if not ipaddress.ip_address(a.host).is_loopback or not 1 <= a.port <= 65535:
        raise ValueError("host must be numeric loopback and port1..65535")
    if a.image_size <= 0 or not 0 <= a.warmup_requests <= 8:
        raise ValueError("positive image size and0..8 warmup requests required")
    if any(not math.isfinite(x) or x <= 0 for x in (a.timeout_seconds, a.total_timeout_seconds)):
        raise ValueError("positive finite deadlines required")
    if bool(a.runtime_dir) != bool(a.expected_run_id):
        raise ValueError("--runtime-dir and --expected-run-id must be supplied together")
    if a.expected_run_id and (len(a.expected_run_id) != 32 or any(c not in "0123456789abcdef" for c in a.expected_run_id)):
        raise ValueError("expected run ID must contain32 lowercase hex characters")
    # A venv interpreter is usually a symlink to the base executable. Resolving
    # that symlink loses pyvenv.cfg discovery and silently bypasses the venv.
    client_python = str(a.client_python.absolute())
    prefix = json.loads(a.client_command_json) if a.client_command_json else [client_python, "-B", "-m", "sglang.bench_serving"]
    if not isinstance(prefix, list) or not prefix or not all(isinstance(x, str) and x for x in prefix):
        raise ValueError("client prefix must be a nonempty JSON argv")
    output, model = a.output_dir.resolve(), a.model_path.resolve()
    selected = set(a.load or [f"{n}:{c}" for n, c in LOADS])
    rows = []
    for n, c in LOADS:
        if f"{n}:{c}" not in selected:
            continue
        for round_ in sorted(set(a.rounds or (1, 2))):
            name = f"round{round_}_{n}_1024_c{c}"
            result = output / "native" / (name + ".jsonl")
            argv = [*prefix, "--backend", CLIENT_BACKEND, "--host", a.host, "--port", str(a.port),
                    "--model", str(model), "--served-model-name", a.served_model_name, "--tokenizer", str(model),
                    "--dataset-name", "image", "--image-count", "1", "--image-resolution", f"{a.image_size}x{a.image_size}",
                    "--image-content", "random", "--image-format", "jpeg", "--random-input-len", str(n),
                    "--random-output-len", "1024", "--random-range-ratio", "1", "--seed", "1",
                    "--max-concurrency", str(c), "--num-prompts", str(max(8, 2*c)), "--request-rate", "inf",
                    "--warmup-requests", str(a.warmup_requests), "--ready-check-timeout-sec", "0",
                    "--extra-request-body", '{"temperature":0,"ignore_eos":true,"stream_options":{"include_usage":true}}',
                    "--output-details", "--output-file", str(result)]
            rows.append({"name": name, "round": round_, "scored_round": round_ == 2,
                         "client_backend": CLIENT_BACKEND, "server_framework": SERVER_FRAMEWORK,
                         "text_target_tokens": n, "output_target_tokens": 1024, "concurrency": c,
                         "expected_requests": max(8, 2*c), "warmup_requests": a.warmup_requests,
                         "command": argv, "native_result": str(result), "log": str(output / (name + ".log")), "status": "planned"})
    return {"schema_version": 1, "mode": "run" if a.run else "plan_only", "official_acceptance": False,
            "official_accuracy": False, "scope": "template-shaped SGLang image performance client; image policy and official acceptance unconfirmed",
            "client_type": "explicit_override_unverified" if a.client_command_json else "upstream_sglang",
            "sglang_version_required": VERSION, "source_reference": SOURCE, "output_length_boundary": LENGTH_BOUNDARY,
            "client_backend": CLIENT_BACKEND, "server_framework": SERVER_FRAMEWORK,
            "server_framework_verified_from_runtime": False, "backend_alias_boundary": BACKEND_BOUNDARY,
            "api_endpoint": f"http://{a.host}:{a.port}/v1/chat/completions",
            "upstream_sender": "async_request_openai_chat_completions",
            "upstream_sender_boundary": "Unmodified upstream sender combines reasoning_content/content and records stream success/usage or requested-length fallback; wrapper does not independently certify terminal framing or exact token count",
            "exact_output_tokens_independently_verified": False,
            "input_length_boundary": "generated text target; image/chat tokens and detokenization are separate",
            "model_path": str(model), "served_model_name": a.served_model_name, "host": a.host, "port": a.port,
            "client_python": client_python, "output_dir": str(output), "rows": rows,
            "allow_client_device_discovery": a.allow_client_device_discovery,
            "client_device_discovery": device_discovery_policy(a.allow_client_device_discovery, os.environ),
            "runtime_dir": str(a.runtime_dir.resolve()) if a.runtime_dir else None, "expected_run_id": a.expected_run_id,
            "timeout_seconds": a.timeout_seconds, "total_timeout_seconds": a.total_timeout_seconds,
            "measured_requests_planned": sum(r["expected_requests"] for r in rows),
            "extra_warmup_requests_planned": sum(r["warmup_requests"] for r in rows),
            "full_matrix": len(rows) == 12, "filesystem_mutated": False, "client_started": False}


def bounded(argv, log, seconds, env):
    """Always reap the client and its Linux process group, including descendants."""
    begin = time.monotonic()
    child = None
    with log.open("xb") as stream:
        try:
            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=env, start_new_session=sys.platform == "linux")
            try:
                code = child.wait(timeout=seconds)
                state = "exited"
            except subprocess.TimeoutExpired:
                code, state = 124, "timeout"
            except KeyboardInterrupt:
                code, state = 130, "interrupted"
        except OSError as exc:
            stream.write(str(exc).encode("utf-8"))
            code, state = 127, "client_error"
        finally:
            if child is not None:
                if sys.platform == "linux":
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif child.poll() is None:
                    child.kill()
                child.wait()
    return {"exit_code": code, "process_status": state, "client_wall_seconds": round(time.monotonic()-begin, 6),
            "log_sha256": sha(log), "child_reaped": child is None or child.returncode is not None,
            "linux_process_group_cleanup_attempted": child is not None and sys.platform == "linux"}


def runtime_snapshot(plan, output, phase):
    if not plan["runtime_dir"]:
        return {"provided": False, "server_identity_verified": False}
    root = Path(plan["runtime_dir"])
    state = json.loads((root / "state.json").read_text())
    run_id = plan["expected_run_id"]
    if state.get("run_id") != run_id:
        raise ValueError("runtime ownership changed; refusing to benchmark/archive another run")
    if phase == "before" and (state.get("status") != "ready" or state.get("url") != f"http://{plan['host']}:{plan['port']}"):
        raise ValueError("runtime is not ready at the requested loopback endpoint")
    config = json.loads((root / ("config-"+run_id+".json")).read_text())
    if "framework" in config and config["framework"] != SERVER_FRAMEWORK:
        raise ValueError("runtime framework differs from the declared vLLM server; sender alias cannot relabel it")
    if (config.get("model_path") != plan["model_path"] or config.get("served_name") != plan["served_model_name"]
            or config.get("port") != plan["port"] or config.get("host") != plan["host"]):
        raise ValueError("runtime model/served-name/endpoint differs from the requested workload")
    folder = output / "runtime" / phase
    folder.mkdir(parents=True)
    result = {"provided": True, "run_id": run_id, "server_identity_verified": True,
              "server_framework": SERVER_FRAMEWORK,
              "server_framework_verified": config.get("framework") == SERVER_FRAMEWORK, "files": []}
    for name in ("state.json", "config-"+run_id+".json", "server.log", "supervisor.log"):
        path = root / name
        if path.is_file():
            raw = path.read_bytes()
            (folder / name).write_bytes(raw)
            result["files"].append({"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return result


def client_identity(plan, output, env):
    if plan["client_type"] != "upstream_sglang":
        return {"type": plan["client_type"], "upstream_verified": False,
                "client_device_discovery": plan.get("client_device_discovery")}
    code = ("import importlib.metadata as m,json,sys;d=m.distribution('sglang');"
            "print(json.dumps({'version':d.version,'root':str(d.locate_file('sglang')),'python':sys.version," 
            "'native':{n:{'version':m.distribution(n).version,'root':str(m.distribution(n).locate_file(''))} "
            "for n in ['torch','transformers','vllm','triton']}}))")
    log = output / "client-identity.log"
    result = bounded([plan["client_python"], "-B", "-c", code], log, 30, env)
    if result["exit_code"]:
        raise ValueError("client identity failed; see client-identity.log")
    identity = json.loads(log.read_text())
    if identity["version"] != VERSION:
        raise ValueError("client version differs from pinned0.5.12")
    folder = output / "sources/upstream-sglang"
    entries = []
    for rel in SOURCE_FILES:
        path = Path(identity["root"]) / rel
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        dest.write_bytes(raw)
        entries.append({"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {**identity, "sources": entries, "upstream_verified": True, "identity_process": result,
            "client_device_discovery": plan.get("client_device_discovery")}


def inspect_result(path, expected):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("each native result must contain exactly one original JSONL record")
    data = json.loads(lines[0], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(data, dict):
        raise ValueError("native result must be an object")
    issues = []
    if type(data.get("completed")) is not int or data["completed"] != expected:
        issues.append("completed differs from expected")
    for name in ("generated_texts", "errors", "output_lens"):
        values = data.get(name)
        if not isinstance(values, list) or len(values) != expected:
            issues.append(name + " must contain one entry per request")
        elif name == "generated_texts" and any(not isinstance(x, str) or not x.strip() for x in values):
            issues.append("empty generated text")
        elif name == "errors" and any(x != "" for x in values):
            issues.append("nonempty request errors")
        elif name == "output_lens" and any(type(x) is not int or x != 1024 for x in values):
            issues.append("client-reported output lengths differ from target")
    metrics = {k: v for k, v in data.items() if k not in {"generated_texts", "errors", "input_lens", "output_lens", "ttfts", "itls"}}
    return {"issues": issues, "client_report_complete": not issues, "metrics": metrics,
            "exact_output_tokens_independently_verified": False, "output_length_boundary": LENGTH_BOUNDARY,
            "native_result_sha256": sha(path)}


def summarize(output, report):
    for n in (1, 2):
        (output / f"round{n}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report["rows"] if r["round"] == n), encoding="utf-8")
    lines = ["# SGLang client performance record", "", "Official acceptance: false. " + LENGTH_BOUNDARY,
             "Configured server framework: vllm. Upstream client sender alias: sglang-oai-chat; OpenAI /v1/chat/completions. " + BACKEND_BOUNDARY, "",
             "| Round | Text target | Concurrency | Completed | Output token/s | Mean TTFT ms | Client wall s | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        m = r.get("result", {}).get("metrics", {})
        lines.append("| " + " | ".join(str(x) for x in (r["round"], r["text_target_tokens"], r["concurrency"], m.get("completed", ""),
                     m.get("output_throughput", ""), m.get("mean_ttft_ms", ""), r.get("client_wall_seconds", ""), r["status"])) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(output / "run.json", report)


def execute(plan):
    output = Path(plan["output_dir"])
    if output.exists() or output.is_symlink():
        raise ValueError("output exists; refusing to overwrite any prior evidence")
    if not Path(plan["model_path"]).is_dir():
        raise ValueError("local model directory required; no model download")
    if plan["client_type"] == "upstream_sglang" and Path(plan["served_model_name"]).resolve() != Path(plan["model_path"]):
        raise ValueError("upstream0.5.12 image processor loads served_model_name; use the same local model path as server served name")
    output.mkdir(parents=True)
    (output / "native").mkdir()
    (output / "sources").mkdir()
    shutil.copyfile(__file__, output / "sources/t02_sglang_bench.py")
    plan.update(started_utc=stamp(), filesystem_mutated=True, driver_sha256=sha(Path(__file__)), status="preparing")
    allow_discovery = plan.get("allow_client_device_discovery", False)
    plan["client_device_discovery_at_plan"] = plan.get("client_device_discovery")
    plan["client_device_discovery"] = device_discovery_policy(allow_discovery, os.environ)
    env = client_environment(allow_discovery, os.environ)
    begin = time.monotonic()
    deadline = begin + plan["total_timeout_seconds"]
    summarize(output, plan)
    interrupted = False
    try:
        plan["runtime_before"] = runtime_snapshot(plan, output, "before")
        plan["server_framework_verified_from_runtime"] = plan["runtime_before"].get("server_framework_verified", False)
        plan["client_identity"] = client_identity(plan, output, env)
        for row in plan["rows"]:
            remaining = deadline - time.monotonic()
            if interrupted or remaining <= 0:
                row["status"] = "not_attempted_interrupted" if interrupted else "not_attempted_total_timeout"
                continue
            row.update(status="running", started_utc=stamp(), allocated_timeout_seconds=min(remaining, plan["timeout_seconds"]))
            summarize(output, plan)
            plan["client_started"] = True
            row.update(bounded(row["command"], Path(row["log"]), row["allocated_timeout_seconds"], env))
            row["ended_utc"] = stamp()
            row["status"] = row["process_status"]
            interrupted = row["process_status"] == "interrupted"
            path = Path(row["native_result"])
            if path.is_file():
                row["native_result_sha256"] = sha(path)
                try:
                    row["result"] = inspect_result(path, row["expected_requests"])
                    if row["process_status"] == "exited":
                        row["status"] = "complete_client_report" if row["exit_code"] == 0 and row["result"]["client_report_complete"] else "incomplete_client_report"
                except (ValueError, UnicodeError) as exc:
                    row.update(status="invalid_native_result", result_error=str(exc))
            elif row["process_status"] == "exited":
                row["status"] = "missing_native_result"
            summarize(output, plan)
            print(json.dumps({"name": row["name"], "status": row["status"], "wall_seconds": row["client_wall_seconds"]}), flush=True)
        plan["status"] = "complete_client_reports" if all(r["status"] == "complete_client_report" for r in plan["rows"]) else "incomplete_or_failed"
    except (ValueError, OSError) as exc:
        plan.update(status="preflight_or_archive_failed", error=str(exc))
    finally:
        try:
            plan["runtime_after"] = runtime_snapshot(plan, output, "after")
        except (ValueError, OSError) as exc:
            plan.update(runtime_archive_error=str(exc), status="runtime_identity_or_archive_failed")
        plan.update(ended_utc=stamp(), total_wall_seconds=round(time.monotonic()-begin, 6))
        summarize(output, plan)
        files = []
        for path in sorted(output.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
        dump(output / "archive-manifest.json", {"created_utc": stamp(), "files": files})
    return 0 if plan["status"] == "complete_client_reports" else 1


def main(argv=None):
    try:
        a = parser().parse_args(argv)
        plan = build_plan(a)
        if not a.run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return execute(plan)
    except (ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
