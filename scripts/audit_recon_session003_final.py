#!/usr/bin/env python3
"""Offline audit of the downloaded T02 development matrix; never official acceptance.

Default output is JSON on stdout. An explicit --output creates a new file only.
Remote absolute paths in run.json are identifiers, never local read targets.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

EXPECTED = [(length, concurrency, round_) for length in (4096, 16384)
            for concurrency in (1, 8, 64) for round_ in (1, 2)]
MODEL = "recon2reason-reasoning-4b"
METRICS = ("completed", "failed", "duration", "request_throughput", "output_throughput",
           "total_token_throughput", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
           "mean_e2el_ms", "p99_e2el_ms", "total_input_tokens", "total_output_tokens")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: " + key)
        value[key] = item
    return value


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))


def read(path, inventory):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing regular evidence file: " + str(path))
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("evidence file exceeds 64 MiB: " + str(path))
    raw = path.read_bytes()
    inventory.append({"path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw)})
    return raw


def validate_outputs(data, count):
    issues, requests = [], []
    for field in ("output_lens", "generated_texts", "errors"):
        values = data.get(field)
        if not isinstance(values, list) or len(values) != count:
            issues.append(f"{field} must contain exactly {count} entries")
    for index in range(count):
        def item(field):
            values = data.get(field)
            return values[index] if isinstance(values, list) and index < len(values) else None
        length, text, error = item("output_lens"), item("generated_texts"), item("errors")
        problems = []
        if type(length) is not int or length != 1024:
            problems.append("output_tokens_not_1024")
        if not isinstance(text, str) or not text.strip():
            problems.append("empty_or_missing_generated_text")
        if not isinstance(error, str) or error:
            problems.append("nonempty_or_missing_error")
        requests.append({"index": index, "output_tokens": length,
                         "generated_text_characters": len(text) if isinstance(text, str) else None,
                         "error": error[:1000] if isinstance(error, str) else error, "issues": problems})
    lengths = data.get("output_lens")
    total = sum(lengths) if isinstance(lengths, list) and all(type(v) is int for v in lengths) else None
    if type(data.get("total_output_tokens")) is not int or data["total_output_tokens"] != count * 1024 or total != count * 1024:
        issues.append("reported/per-request/expected output token totals differ")
    return {"verified": not issues and all(not r["issues"] for r in requests), "issues": issues,
            "requests": requests, "reported_output_tokens": data.get("total_output_tokens"),
            "sum_output_tokens": total, "expected_output_tokens": count * 1024}


def option(command, flag):
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command) or command.count(flag) != 1:
        raise ValueError("command requires exactly one " + flag)
    index = command.index(flag)
    if index + 1 == len(command):
        raise ValueError("command option lacks value: " + flag)
    return command[index + 1]


def remote_path(value):
    return PurePosixPath(value.replace("\\", "/"))


def command_issues(command, run, length, concurrency, count, name):
    expected = {"--backend": "openai-chat", "--endpoint": "/v1/chat/completions",
                "--host": run.get("host"), "--port": str(run.get("port")),
                "--model": run.get("model_path"), "--tokenizer": run.get("tokenizer_path"),
                "--served-model-name": MODEL, "--dataset-name": "random-mm",
                "--random-input-len": str(length), "--random-output-len": "1024",
                "--random-range-ratio": "0", "--random-prefix-len": "0",
                "--random-mm-base-items-per-request": "1", "--random-mm-num-mm-items-range-ratio": "0",
                "--max-concurrency": str(concurrency), "--num-prompts": str(count),
                "--request-rate": "inf", "--temperature": "0", "--num-warmups": "0",
                "--ready-check-timeout-sec": "0", "--result-filename": name + ".json", "--seed": "1"}
    issues = []
    try:
        if not any(command[i:i + 2] == ["bench", "serve"] for i in range(len(command) - 1)):
            issues.append("command is not bench serve")
        for key, value in expected.items():
            if option(command, key) != value:
                issues.append("command mismatch: " + key)
        if remote_path(option(command, "--result-dir")) != remote_path(run["output_dir"]) / "native":
            issues.append("command result directory differs")
        if parse(option(command, "--random-mm-limit-mm-per-prompt")) != {"image": 1, "video": 0}:
            issues.append("command multimodal limits differ")
        if ast.literal_eval(option(command, "--random-mm-bucket-config")) != {(1024, 1024, 1): 1.0}:
            issues.append("command image bucket differs")
        for flag in ("--ignore-eos", "--save-result", "--save-detailed"):
            if command.count(flag) != 1:
                issues.append("command requires flag: " + flag)
    except (TypeError, ValueError, SyntaxError) as exc:
        issues.append(str(exc))
    return issues


def runtime_inventory(directory):
    result = {"provided": directory is not None, "files": [], "profiles": [], "states": [],
              "live_process_or_port_checked": False, "platform_shutdown_verified": False}
    if directory is None:
        return result
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("runtime-dir must be an existing local directory")
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        raw = read(path, result["files"])
        if path.suffix == ".log":
            for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                memory = re.search(r"Available KV cache memory:\s*([0-9.]+) GiB", line)
                tokens = re.search(r"GPU KV cache size:\s*([0-9,]+) tokens", line)
                if memory or tokens:
                    result["profiles"].append({"path": str(path.resolve()), "line": number, "raw_line": line,
                                                "kv_memory_gib": float(memory[1]) if memory else None,
                                                "kv_tokens": int(tokens[1].replace(",", "")) if tokens else None})
        if path.suffix == ".json" and "state" in path.stem:
            value = parse(raw)
            if isinstance(value, dict):
                result["states"].append({"path": str(path.resolve()), "run_id": value.get("run_id"),
                                         "status": value.get("status"), "reason": value.get("reason"),
                                         "created_utc": value.get("created_utc"), "ready_utc": value.get("ready_utc"),
                                         "ended_utc": value.get("ended_utc"), "command": value.get("command")})
    result["recorded_stopped_state_present"] = any(row["status"] == "stopped" for row in result["states"])
    return result


def audit_run(directory, driver_source=None, require_full=True):
    directory = directory.resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("matrix-dir must be an existing local directory")
    files, issues, rows = [], [], []
    run = parse(read(directory / "run.json", files))
    if not isinstance(run, dict) or run.get("schema_version") != 1 or not isinstance(run.get("rows"), list):
        raise ValueError("unsupported development run.json schema")
    all_names = [f"round{r}_{length}_1024_c{c}" for length, c, r in EXPECTED]
    actual_names = [row.get("name") if isinstance(row, dict) else None for row in run["rows"]]
    selected = EXPECTED if require_full else [cell for cell, name in zip(EXPECTED, all_names) if name in actual_names]
    if not selected:
        issues.append("retry has no recognized matrix cells")
    planned_requests = sum(max(8, 2 * concurrency) for _, concurrency, _ in selected)
    for key, value in (("full_matrix", len(selected) == 12), ("requests_planned", planned_requests), ("mode", "run"),
                       ("status", "complete"), ("official_acceptance", False), ("served_model_name", MODEL),
                       ("image_size", [1024, 1024]), ("images_per_request", 1),
                       ("extra_readiness_requests", 0), ("warmup_requests_inside_each_command", 0)):
        if run.get(key) != value:
            issues.append("run field mismatch: " + key)
    try:
        if not ipaddress.ip_address(run.get("host")).is_loopback:
            issues.append("run host is not numeric loopback")
    except (TypeError, ValueError):
        issues.append("run host is not numeric loopback")
    expected_names = [f"round{r}_{length}_1024_c{c}" for length, c, r in selected]
    if actual_names != expected_names:
        issues.append("matrix rows/order differ from expected cells")
    duplicates = [name for name, count in Counter(actual_names).items() if count > 1]
    if duplicates:
        issues.append("duplicate cell names: " + repr(duplicates))
    lookup = {row.get("name"): row for row in run["rows"] if isinstance(row, dict)}
    source = driver_source or Path(__file__).with_name("t02_development_bench.py")
    source_raw = read(source, files)
    driver_match = sha(source_raw) == run.get("driver_sha256")
    if not driver_match:
        issues.append("driver source SHA differs from the recorded execution source")
    total_wall = 0.0
    for length, concurrency, round_ in selected:
        name, count = f"round{round_}_{length}_1024_c{concurrency}", max(8, 2 * concurrency)
        source_row = lookup.get(name)
        row = {"name": name, "round": round_, "target_text_input_tokens": length, "concurrency": concurrency,
               "expected_requests": count, "issues": [], "verified": False,
               "completed_requests_reported": None, "failed_requests_reported": None,
               "requests_with_full_output": None, "sum_output_tokens": None}
        rows.append(row)
        if source_row is None:
            row["issues"].append("missing matrix cell")
            continue
        row["recorded_status"] = source_row.get("status")
        for key, value in (("round", round_), ("target_text_input_tokens", length), ("target_output_tokens", 1024),
                           ("concurrency", concurrency), ("expected_requests", count), ("scored_round", round_ == 2),
                           ("status", "complete"), ("exit_code", 0)):
            if source_row.get(key) != value:
                row["issues"].append("row field mismatch: " + key)
        row["issues"].extend(command_issues(source_row.get("command"), run, length, concurrency, count, name))
        row["budget_upper_bound_before_seconds"] = run.get("total_timeout_seconds", 0) - total_wall
        wall = source_row.get("client_wall_seconds")
        if isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(wall) and wall >= 0:
            total_wall += wall
        else:
            row["issues"].append("missing/invalid client wall duration")
        row["client_wall_seconds"] = wall
        row["started_utc"], row["ended_utc"] = source_row.get("started_utc"), source_row.get("ended_utc")
        try:
            start = datetime.fromisoformat(row["started_utc"])
            end = datetime.fromisoformat(row["ended_utc"])
            if start.tzinfo is None or end.tzinfo is None or end < start:
                raise ValueError("row timestamps must be timezone aware and ordered")
            row["utc_elapsed_seconds"] = (end - start).total_seconds()
        except (TypeError, ValueError) as exc:
            row["issues"].append("invalid row timing: " + str(exc))
        log_raw = None
        try:
            log_raw = read(directory / (name + ".log"), files)
        except (OSError, ValueError) as exc:
            row["issues"].append(str(exc))
        try:
            native_raw = read(directory / "native" / (name + ".json"), files)
            for field, raw in (("native_result", native_raw), ("log", log_raw)):
                if raw is None or sha(raw) != source_row.get(field + "_sha256"):
                    row["issues"].append(field + " SHA mismatch")
                expected_remote = remote_path(run["output_dir"]) / ("native" if field == "native_result" else "") / (name + (".json" if field == "native_result" else ".log"))
                if remote_path(source_row.get(field, "")) != expected_remote:
                    row["issues"].append(field + " remote identity differs")
            data = parse(native_raw)
            if not isinstance(data, dict):
                raise ValueError("native result must be an object")
            row["metrics"] = {key: data[key] for key in METRICS if key in data}
            if row["metrics"] != source_row.get("metrics"):
                row["issues"].append("recorded metrics differ from native JSON")
            for field in METRICS:
                value = data.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    row["issues"].append("missing/invalid native metric: " + field)
            duration = data.get("duration")
            if type(duration) not in (int, float) or duration <= 0:
                row["issues"].append("native duration must be positive")
            elif all(type(data.get(key)) in (int, float) for key in ("completed", "total_output_tokens", "total_input_tokens")):
                for field, numerator in (("request_throughput", data["completed"]),
                                         ("output_throughput", data["total_output_tokens"]),
                                         ("total_token_throughput", data["total_output_tokens"] + data["total_input_tokens"])):
                    value = data.get(field)
                    if type(value) not in (int, float) or not math.isclose(value, numerator / duration, rel_tol=1e-6, abs_tol=1e-9):
                        row["issues"].append("native throughput/token/duration mismatch: " + field)
            if type(data.get("completed")) is not int or data["completed"] != count or type(data.get("failed")) is not int or data["failed"] != 0:
                row["issues"].append("native completed/failed counts differ")
            row["output_integrity"] = validate_outputs(data, count)
            row.update(completed_requests_reported=data.get("completed"), failed_requests_reported=data.get("failed"),
                       requests_with_full_output=sum(not item["issues"] for item in row["output_integrity"]["requests"]),
                       sum_output_tokens=row["output_integrity"]["sum_output_tokens"])
            if not row["output_integrity"]["verified"]:
                row["issues"].append("independent output integrity failed")
            recorded_integrity = source_row.get("output_integrity")
            if not isinstance(recorded_integrity, dict) or recorded_integrity.get("verified") is not True:
                row["issues"].append("driver did not verify output integrity")
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
            row["issues"].append(str(exc))
        row["verified"] = not row["issues"]
    for round_ in (1, 2):
        try:
            raw = read(directory / f"round{round_}.jsonl", files)
            values = [parse(line) for line in raw.decode("utf-8").splitlines()]
            if values != [row for row in run["rows"] if row.get("round") == round_]:
                issues.append(f"round{round_}.jsonl differs from run.json")
        except (OSError, ValueError, UnicodeError) as exc:
            issues.append(str(exc))
    try:
        read(directory / "summary.md", files)
    except (OSError, ValueError) as exc:
        issues.append(str(exc))
    stable = all(Path(item["path"]).is_file() and sha(Path(item["path"]).read_bytes()) == item["sha256"] for item in files)
    if not stable:
        issues.append("source evidence changed during the audit")
    complete = not issues and bool(rows) and all(row["verified"] for row in rows)
    return {"schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "offline_downloaded_development_matrix_audit", "matrix_dir": str(directory),
            "run_complete_verified": complete, "matrix_complete_verified": complete and len(rows) == 12,
            "official_accuracy": False, "official_performance": False,
            "formal_ready": False, "submitted": False, "expected_cells": len(selected), "expected_requests": planned_requests,
            "server_state_scope": "Persistent server with fixed seed=1 permits prefix-cache reuse across rounds/loads; no independent cold start or cache-disabled baseline is verified.",
            "expected_output_tokens": planned_requests * 1024, "verified_cells": sum(row["verified"] for row in rows),
            "verified_requests": sum(row["expected_requests"] for row in rows if row["verified"]),
            "verified_output_tokens": sum(row["expected_requests"] * 1024 for row in rows if row["verified"]),
            "observed_native_completed_requests": sum(row["completed_requests_reported"] for row in rows if type(row["completed_requests_reported"]) is int),
            "observed_sum_output_tokens": sum(row["sum_output_tokens"] for row in rows if type(row["sum_output_tokens"]) is int),
            "cells_missing_native_request_counts": [row["name"] for row in rows if row["completed_requests_reported"] is None],
            "driver_sha256_match": driver_match, "reported_driver_sha256": run.get("driver_sha256"),
            "native_vllm_version": run.get("native_vllm_version"),
            "reported_native_source_sha256": run.get("native_source_sha256"),
            "identity": {key: run.get(key) for key in ("model_path", "tokenizer_path", "served_model_name", "host", "port", "image_size", "images_per_request", "driver_sha256", "native_vllm_version", "native_source_sha256")},
            "started_utc": run.get("started_utc"), "ended_utc": run.get("ended_utc"),
            "deadline": {"total_timeout_seconds": run.get("total_timeout_seconds"),
                         "per_cell_timeout_seconds": run.get("timeout_seconds"), "client_wall_sum_seconds": total_wall,
                         "budget_note": "Upper bounds ignore metadata/summary overhead; deadline starts after initial metadata collection.",
                         "first_incomplete_cell": next((row["name"] for row in rows if not row["verified"]), None)},
            "issues": issues, "rows": rows, "source_files": files, "source_files_stable_during_audit": stable}


def audit(directory, runtime=None, driver_source=None, retries=()):
    report = audit_run(directory, driver_source)
    report["runtime"] = runtime_inventory(runtime)
    report["retries"] = [audit_run(path, driver_source, require_full=False) for path in retries]
    report["combined_coverage"] = []
    for length, concurrency, round_ in EXPECTED:
        name = f"round{round_}_{length}_1024_c{concurrency}"
        candidates = [report, *report["retries"]]
        eligible = [run for run in candidates if run["driver_sha256_match"] and run["identity"] == report["identity"]
                    and not [issue for issue in run["issues"] if issue != "run field mismatch: status"]]
        selected = next(((run, row) for run in eligible for row in run["rows"] if row["name"] == name and row["verified"]), None)
        report["combined_coverage"].append({"name": name, "verified": selected is not None,
                                             "source_run_dir": selected[0]["matrix_dir"] if selected else None,
                                             "source_is_retry": selected is not None and selected[0] is not report,
                                             "requests": max(8, concurrency * 2)})
    report["combined_development_coverage_complete"] = all(row["verified"] for row in report["combined_coverage"])
    report["separate_runs_note"] = "Retry evidence never changes the original run status. Combined coverage is development request coverage only; same server configuration and official performance equivalence are not automatically verified."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--driver-source", type=Path)
    parser.add_argument("--retry-dir", type=Path, action="append", default=[], help="separate retry run folder; original failures remain visible")
    parser.add_argument("--output", type=Path, help="optional new JSON file; existing output is never overwritten")
    args = parser.parse_args(argv)
    try:
        report = audit(args.matrix_dir, args.runtime_dir, args.driver_source, args.retry_dir)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as target:
                target.write(text)
        else:
            print(text, end="")
        return 0 if report["matrix_complete_verified"] or report["combined_development_coverage_complete"] else 1
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        print(json.dumps({"audit_error": str(exc), "matrix_complete_verified": False,
                          "official_accuracy": False, "official_performance": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
