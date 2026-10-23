#!/usr/bin/env python3
"""Explicit development load driver for vLLM 0.18.1 random-mm; never an official score."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCES = {name: "https://raw.githubusercontent.com/vllm-project/vllm/v0.18.1/vllm/benchmarks/" + name + ".py"
           for name in ("serve", "datasets")}
LOADS = [(length, concurrency) for length in (4096, 16384) for concurrency in (1, 8, 64)]
METRICS = ("completed", "failed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
           "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2el_ms", "p99_e2el_ms",
           "total_input_tokens", "total_output_tokens")


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", action="store_true", help="explicitly execute the client; default only prints a plan")
    result.add_argument("--host", default="127.0.0.1", help="numeric loopback address only")
    result.add_argument("--port", type=int, default=30000)
    result.add_argument("--model-path", required=True, type=Path)
    result.add_argument("--served-model-name", required=True)
    result.add_argument("--tokenizer-path", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("development-bench/t02-run"), help="must not exist when --run is used")
    result.add_argument("--image-size", type=int, default=1024, help="explicit development image size; official load unconfirmed")
    result.add_argument("--timeout-seconds", type=float, default=1800)
    result.add_argument("--total-timeout-seconds", type=float, default=7200)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--vllm-command-json", default='["vllm"]', help="explicit CLI prefix argv, useful for an absolute native CLI or CPU mock")
    result.add_argument("--load", action="append", choices=[f"{length}:{c}" for length, c in LOADS], help="optional calibration subset, repeatable")
    result.add_argument("--round", action="append", dest="rounds", type=int, choices=(1, 2), help="optional calibration subset, repeatable")
    return result


def build_plan(args):
    try:
        valid_host = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        valid_host = False
    if not valid_host:
        raise ValueError("--host must be a numeric loopback address")
    if not 1 <= args.port <= 65535 or args.image_size < 1:
        raise ValueError("port must be 1..65535 and image size must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout_seconds, args.total_timeout_seconds)):
        raise ValueError("timeouts must be positive finite seconds")
    prefix = json.loads(args.vllm_command_json)
    if not isinstance(prefix, list) or not prefix or not all(isinstance(item, str) and item for item in prefix):
        raise ValueError("--vllm-command-json must be a nonempty argv array")
    model = args.model_path.resolve()
    tokenizer = (args.tokenizer_path or model).resolve()
    output = args.output_dir.resolve()
    rounds = sorted(set(args.rounds or (1, 2)))
    selected = set(args.load or [f"{length}:{c}" for length, c in LOADS])
    rows = []
    for length, concurrency in LOADS:
        if f"{length}:{concurrency}" not in selected:
            continue
        for round_ in rounds:
            name = f"round{round_}_{length}_1024_c{concurrency}"
            argv = [*prefix, "bench", "serve", "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
                    "--host", args.host, "--port", str(args.port), "--model", str(model),
                    "--served-model-name", args.served_model_name, "--tokenizer", str(tokenizer),
                    "--dataset-name", "random-mm", "--random-input-len", str(length),
                    "--random-output-len", "1024", "--random-range-ratio", "0", "--random-prefix-len", "0",
                    "--random-mm-base-items-per-request", "1", "--random-mm-num-mm-items-range-ratio", "0",
                    "--random-mm-limit-mm-per-prompt", '{"image":1,"video":0}',
                    "--random-mm-bucket-config", str({(args.image_size, args.image_size, 1): 1.0}),
                    "--max-concurrency", str(concurrency), "--num-prompts", str(max(8, concurrency * 2)),
                    "--request-rate", "inf", "--seed", str(args.seed), "--temperature", "0", "--ignore-eos",
                    "--num-warmups", "0", "--ready-check-timeout-sec", "0", "--disable-tqdm",
                    "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "99",
                    "--save-result", "--save-detailed", "--result-dir", str(output / "native"),
                    "--result-filename", name + ".json"]
            rows.append({"name": name, "round": round_, "scored_round": round_ == 2,
                         "target_text_input_tokens": length, "target_output_tokens": 1024,
                         "concurrency": concurrency, "expected_requests": max(8, concurrency * 2),
                         "command": argv, "native_result": str(output / "native" / (name + ".json")),
                         "log": str(output / (name + ".log")), "status": "planned"})
    return {"schema_version": 1, "mode": "run" if args.run else "plan_only", "official_acceptance": False,
            "benchmark_scope": "development synthetic random-mm; image specification and total input token target unconfirmed",
            "source_reference": SOURCES, "model_path": str(model), "tokenizer_path": str(tokenizer),
            "served_model_name": args.served_model_name, "host": args.host, "port": args.port,
            "output_dir": str(output), "image_size": [args.image_size, args.image_size],
            "images_per_request": 1, "full_matrix": len(rows) == 12,
            "requests_planned": sum(row["expected_requests"] for row in rows),
            "timeout_seconds": args.timeout_seconds, "total_timeout_seconds": args.total_timeout_seconds,
            "extra_readiness_requests": 0, "warmup_requests_inside_each_command": 0,
            "filesystem_mutated": False, "client_started": False, "rows": rows}


def output_integrity(data, expected_requests, target_output_tokens):
    """Check save-detailed evidence; HTTP/client success alone is insufficient."""
    required = ("output_lens", "generated_texts", "errors", "total_output_tokens")
    missing = [key for key in required if key not in data]
    issues = []
    arrays = {}
    for key in required[:3]:
        value = data.get(key)
        arrays[key] = value if isinstance(value, list) else None
        if key in data and (not isinstance(value, list) or len(value) != expected_requests):
            issues.append({"field": key, "reason": "requires one list entry per request",
                           "actual_entries": len(value) if isinstance(value, list) else None})
    requests = []
    for index in range(expected_requests):
        item = {"index": index}
        for field in required[:3]:
            values = arrays[field]
            value = values[index] if values is not None and index < len(values) else None
            if field == "output_lens":
                item["output_tokens"] = value
                if type(value) is not int or value != target_output_tokens:
                    issues.append({"request": index, "field": field, "reason": "output length differs from target", "actual": value})
            elif field == "generated_texts":
                item["generated_text_characters"] = len(value) if isinstance(value, str) else None
                if not isinstance(value, str) or not value.strip():
                    issues.append({"request": index, "field": field, "reason": "missing or empty generated text"})
            else:
                item["error"] = value[:1000] if isinstance(value, str) else value
                if not isinstance(value, str) or value != "":
                    issues.append({"request": index, "field": field, "reason": "request error is not an empty string"})
        requests.append(item)
    lengths = arrays["output_lens"]
    actual_total = sum(lengths) if lengths is not None and all(type(value) is int for value in lengths) else None
    reported = data.get("total_output_tokens")
    expected_total = expected_requests * target_output_tokens
    if "total_output_tokens" in data and (type(reported) is not int or reported != actual_total or reported != expected_total):
        issues.append({"field": "total_output_tokens", "reason": "reported, per-request and target totals must match",
                       "reported": reported, "sum_output_lens": actual_total, "expected": expected_total})
    return {"verified": not missing and not issues, "status": "unverified" if missing else "failed" if issues else "verified",
            "missing_fields": missing, "issues": issues, "requests": requests,
            "reported_total_output_tokens": reported, "sum_output_lens": actual_total,
            "expected_total_output_tokens": expected_total}


def summarize(output, report):
    for round_ in (1, 2):
        rows = [row for row in report["rows"] if row["round"] == round_]
        (output / f"round{round_}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    lines = ["# T02 development benchmark", "", "Development synthetic load; official acceptance is false.", "",
             "Requested input lengths describe generated text. Image/chat tokens and detokenization mismatch require server-side calibration.", "",
             "| Round | Text target | Output target | Concurrency | Expected | Completed | Failed | Native duration s | Request/s | Output token/s | Mean TTFT ms | p99 TTFT ms | Mean TPOT ms | p99 TPOT ms | Mean E2EL ms | p99 E2EL ms | Client wall s | Status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["rows"]:
        metrics = row.get("metrics", {})
        lines.append("| " + " | ".join(str(value) for value in (row["round"], row["target_text_input_tokens"],
                     row["target_output_tokens"], row["concurrency"], row["expected_requests"],
                     metrics.get("completed", ""), metrics.get("failed", ""), metrics.get("duration", ""),
                     metrics.get("request_throughput", ""), metrics.get("output_throughput", ""),
                     metrics.get("mean_ttft_ms", ""), metrics.get("p99_ttft_ms", ""),
                     metrics.get("mean_tpot_ms", ""), metrics.get("p99_tpot_ms", ""),
                     metrics.get("mean_e2el_ms", ""), metrics.get("p99_e2el_ms", ""),
                     row.get("client_wall_seconds", ""), row["status"])) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "run.json", report)


def execute(plan):
    output = Path(plan["output_dir"])
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists; refusing to overwrite any previous run")
    if not Path(plan["model_path"]).is_dir() or not Path(plan["tokenizer_path"]).is_dir():
        raise ValueError("--run requires local model/tokenizer directories; no remote model downloads are requested")
    output.mkdir(parents=True, exist_ok=False)
    (output / "native").mkdir()
    plan.update(started_utc=stamp(), filesystem_mutated=True, driver_sha256=sha(Path(__file__)))
    try:
        dist = importlib.metadata.distribution("vllm")
        plan["native_vllm_version"] = dist.version
        plan["native_source_sha256"] = {name: sha(Path(dist.locate_file("vllm/benchmarks/" + name + ".py"))) for name in SOURCES}
    except (importlib.metadata.PackageNotFoundError, OSError):
        plan["native_vllm_version"] = None
        plan["native_source_sha256"] = {}
    # The native aiohttp client trusts environment proxies; keep local benchmark traffic local.
    env = {key: value for key, value in os.environ.items() if key.lower() not in
           {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    env.update(NO_PROXY="127.0.0.0/8,::1,localhost", no_proxy="127.0.0.0/8,::1,localhost")
    deadline = time.monotonic() + plan["total_timeout_seconds"]
    interrupted = False
    summarize(output, plan)
    for row in plan["rows"]:
        remaining = deadline - time.monotonic()
        if interrupted or remaining <= 0:
            row["status"] = "not_attempted_interrupted" if interrupted else "not_attempted_total_timeout"
            summarize(output, plan)
            continue
        row.update(started_utc=stamp(), status="running")
        summarize(output, plan)
        begin = time.monotonic()
        process = None
        plan["client_started"] = True
        try:
            with Path(row["log"]).open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(row["command"], stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, env=env, start_new_session=sys.platform == "linux")
                try:
                    row["exit_code"] = process.wait(timeout=min(plan["timeout_seconds"], remaining))
                except subprocess.TimeoutExpired:
                    row.update(exit_code=124, status="timeout")
                except KeyboardInterrupt:
                    row.update(exit_code=130, status="interrupted")
                    interrupted = True
                finally:
                    if sys.platform == "linux":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif process.poll() is None:
                        process.kill()
                    process.wait()
        except OSError as exc:
            row.update(exit_code=127, status="client_error", error=str(exc))
        row.update(ended_utc=stamp(), client_wall_seconds=round(time.monotonic() - begin, 6))
        if Path(row["log"]).is_file():
            row["log_sha256"] = sha(Path(row["log"]))
        result = Path(row["native_result"])
        if result.is_file():
            row["native_result_sha256"] = sha(result)
            try:
                data = json.loads(result.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(data, dict):
                    raise ValueError("native result is not a JSON object")
                row["metrics"] = {key: data[key] for key in METRICS if key in data}
                integrity = output_integrity(data, row["expected_requests"], row["target_output_tokens"])
                row["output_integrity"] = integrity
                if row["status"] == "running":
                    if (row["exit_code"] != 0 or type(data.get("completed")) is not int
                            or data["completed"] != row["expected_requests"] or type(data.get("failed")) is not int or data["failed"] != 0):
                        row["status"] = "incomplete_or_failed"
                    elif integrity["status"] == "unverified":
                        row["status"] = "unverified_output_integrity"
                    else:
                        row["status"] = "complete" if integrity["verified"] else "incomplete_output_integrity"
            except (ValueError, UnicodeError) as exc:
                row.update(status="invalid_native_json", error=str(exc))
        elif row["status"] == "running":
            row["status"] = "missing_native_result" if row["exit_code"] == 0 else "client_failed"
        summarize(output, plan)
        print(json.dumps({"name": row["name"], "status": row["status"], "exit_code": row["exit_code"],
                          "client_wall_seconds": row["client_wall_seconds"]}), flush=True)
    plan.update(ended_utc=stamp(), status="complete" if all(row["status"] == "complete" for row in plan["rows"]) else "incomplete_or_failed")
    summarize(output, plan)
    return 0 if plan["status"] == "complete" else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if not args.run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return execute(plan)
    except (ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
