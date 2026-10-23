#!/usr/bin/env python3
"""Bounded, explicit two-cell Recon development retry. Default: read-only plan."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

OLD = "04b0c91fd0596360f142021800f1487af5a83264cb6de5971b954b45dc80b422"
NEW = "66e9a97c63469698bfc09fbc1158a28fbd7a48f52877abb21156785a33ab1221"
DRIVER = "bc09711bb58fb5a7ae4676a840956a4da1caf49f6e6034b8015c5dd9a5bae14e"
OLD_RUN_ID = "d70f7e5f590745e28488c55820169551"
MISSING = ["round1_16384_1024_c64", "round2_16384_1024_c64"]
HARD_END = datetime(2026, 10, 19, 19, 48, tzinfo=timezone.utc)  # 2026-10-20 03:48 Beijing.


def now():
    return datetime.now(timezone.utc)


def seconds_left():
    return (HARD_END - now()).total_seconds()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, expected):
    return bool(expected) and token(pid) == expected


def port_open(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def exact_state(task, run_id):
    state = read(task / "runtime/state.json")
    require(state.get("run_id") == run_id, "state changed to another run; refusing to control it")
    expected = task / "runtime" / ("config-" + run_id + ".json")
    require(Path(state["config_path"]) == expected, "state config path does not belong to this run")
    config = read(expected)
    require(config.get("run_id") == run_id and Path(config["task"]).resolve() == task, "config identity mismatch")
    return state, config


def cleanup_snapshot(task, run_id, port):
    state, _ = exact_state(task, run_id)
    result = {"captured_utc": now().isoformat(), "run_id": run_id, "state_status": state.get("status"),
              "supervisor_same_identity_alive": alive(state.get("supervisor_pid", 0), state.get("supervisor_token")),
              "server_same_identity_alive": alive(state.get("server_pid", 0), state.get("server_token")),
              "port_open": port_open(port), "platform_shutdown_verified": False}
    result["owned_service_cleanup_verified"] = not any(result[k] for k in
        ("supervisor_same_identity_alive", "server_same_identity_alive", "port_open"))
    return result


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def copy_new(source, target):
    require(source.is_file() and not source.is_symlink(), "missing/nonregular source: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(source.read_bytes())
    return {"source": str(source), "copy": str(target), "bytes": target.stat().st_size, "sha256": sha(target)}


def validate(root):
    task = (root / "tasks/recon2reason-bw-vllm").resolve()
    report_path = root / "evidence/session003/recon-matrix/run.json"
    report = read(report_path)
    require(report.get("status") == "incomplete_or_failed" and report.get("ended_utc"), "original matrix has not ended incomplete")
    rows = report.get("rows", [])
    expected_names = [f"round{r}_{length}_1024_c{c}" for length in (4096, 16384) for c in (1, 8, 64) for r in (1, 2)]
    require([r.get("name") for r in rows] == expected_names, "original matrix cells/order differ")
    require([r["name"] for r in rows if r.get("status") != "complete"] == MISSING, "expected exactly the final two c64 cells missing")
    require(rows[-2].get("status") == "timeout" and rows[-1].get("status") == "not_attempted_total_timeout", "unexpected original terminal states")
    require(report.get("full_matrix") is True and report.get("requests_planned") == 608, "original plan identity mismatch")
    require(report.get("driver_sha256") == DRIVER, "original driver source differs")
    require(sha(root / "scripts/t02_development_bench.py") == DRIVER, "current development driver differs")
    require(sha(root / "scripts/recon2reason_runtime.py") == NEW, "canonical runtime must already be the reviewed 66 source")
    archive = root / "evidence/session003/recon-original-runtime"
    require(sha(archive / "sources/scripts/recon2reason_runtime.py") == OLD, "preserved original 04 source is missing/different")
    state = read(task / "runtime/state.json")
    old_id = state.get("run_id", "")
    require(re.fullmatch("[0-9a-f]{32}", old_id), "old run ID is invalid")
    require(old_id == OLD_RUN_ID, "runtime no longer identifies the root-confirmed original run; refusing to stop another run")
    state, config = exact_state(task, old_id)
    require(config.get("served_name") == report.get("served_model_name") == "recon2reason-reasoning-4b", "served model differs")
    require(config.get("model_path") == report.get("model_path") and config.get("port") == report.get("port") == 30000, "model/port differs")
    require(config.get("framework") == "vllm" and config.get("precision") == "bf16" and config.get("tp") == 1, "original configuration differs")
    extra = config.get("extra")
    require(isinstance(extra, list) and all(isinstance(v, str) for v in extra), "config extra is not argv")
    require("--max-num-seqs" in extra and extra[extra.index("--max-num-seqs") + 1] == "16", "original profile is not maxseq16")
    output = root / "evidence/session003/recon-matrix-retry"
    control = root / "evidence/session003/recon-retry-control"
    require(not output.exists() and not output.is_symlink(), "retry output already exists")
    require(not control.exists() and not control.is_symlink(), "retry control output already exists")
    for target in ("state-before-stop.json", "state-after-stop.json", "server.log", "supervisor.log", "archive-manifest.json", "config-" + old_id + ".json"):
        require(not (archive / target).exists(), "old archive target already exists: " + target)
    return task, report_path, report, archive, state, config, output, control


def descendants(parent):
    """Only descendants of our own client process; retain Linux start tokens."""
    candidates = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            candidates[int(entry.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            pass
    found = {parent}
    while True:
        added = {pid for pid, (ppid, _) in candidates.items() if ppid in found} - found
        if not added:
            break
        found.update(added)
    return {pid: candidates[pid][1] for pid in found - {parent} if pid in candidates}


def bounded(argv, stream, timeout, env, record):
    require(timeout > 0, "no subprocess budget remains")
    record("command", argv=argv, timeout_seconds=timeout)
    process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               cwd=env["RECON_RETRY_ROOT"], env=env, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            owned_children = descendants(process.pid)
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            for pid, birth in owned_children.items():
                if alive(pid, birth):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            record("client_timeout_cleanup", descendants_recorded=owned_children)


@contextmanager
def runtime_lock(runtime):
    import fcntl
    with (runtime / "control.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(root, log_path):
    require(sys.platform == "linux", "--run requires the SCNet Linux container")
    require(seconds_left() >= 1800, "too little time remains for two bounded retries")
    task, original, report, archive, old_state, config, output, control = validate(root)
    original_sha = sha(original)
    old_id, owned_id = old_state["run_id"], uuid.uuid4().hex
    env = os.environ.copy()
    env.update(RECON_RETRY_ROOT=str(root), RECON2REASON_MODEL_PATH=config["model_path"],
               RECON2REASON_SERVED_NAME=config["served_name"], RECON2REASON_PORT="30000", RECON2REASON_TP="1",
               RECON2REASON_MAX_LEN=str(config["max_len"]), RECON2REASON_MEM_FRAC=str(config["mem_frac"]),
               RECON2REASON_EXTRA_ARGS_JSON=json.dumps(config["extra"]), RECON2REASON_PYTHON_BIN=sys.executable,
               RECON2REASON_MAX_RUNTIME_SECONDS=str(min(3300, int(seconds_left() - 30))),
               RECON2REASON_LOAD_TIMEOUT=str(min(900, int(seconds_left() - 1000))),
               RECON2REASON_STOP_GRACE_SECONDS="30")
    control.mkdir(parents=True, exist_ok=False)
    events = []
    started_launch = False
    driver_code = 2
    cleanup_ok = False
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        def record(case, **values):
            event = {"utc": now().isoformat(), "case": case, **values}
            events.append(event)
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(event, ensure_ascii=False), flush=True)

        record("scope", kind="two_cell_independent_development_retry", official_acceptance=False,
               old_run_id=old_id, owned_run_id=owned_id, hard_end_utc=HARD_END.isoformat(), original_report_sha256=original_sha)
        preserved = [copy_new(task / "runtime/state.json", archive / "state-before-stop.json"),
                     copy_new(Path(old_state["config_path"]), archive / ("config-" + old_id + ".json"))]
        try:
            rc = bounded(["bash", str(task / "stop.sh"), "--run-id", old_id], log, min(50, seconds_left() - 90), env, record)
            require(rc == 0, "old matching stop failed")
            with runtime_lock(task / "runtime"):
                clean = cleanup_snapshot(task, old_id, 30000)
                require(clean["owned_service_cleanup_verified"], "old tokens/port are still active; refusing new service")
                write_new(archive / "cleanup-check.json", clean)
                preserved.append(copy_new(task / "runtime/state.json", archive / "state-after-stop.json"))
                for name in ("server.log", "supervisor.log"):
                    source, target = task / "runtime" / name, archive / name
                    require(source.is_file() and not source.is_symlink() and not target.exists(), "unsafe log archive target")
                    source.rename(target)
                    preserved.append({"source": str(source), "copy": str(target), "bytes": target.stat().st_size, "sha256": sha(target), "operation": "rename_after_owned_stop"})
                write_new(archive / "archive-manifest.json", {"files": preserved, "old_source_sha256": OLD, "old_run_id": old_id})
            for rel in ("scripts/recon2reason_runtime.py", "scripts/t02_development_bench.py", "tasks/recon2reason-bw-vllm/env.sh", "tasks/recon2reason-bw-vllm/run.sh", "tasks/recon2reason-bw-vllm/stop.sh"):
                copy_new(root / rel, control / "sources" / rel)
            started_launch = True
            rc = bounded(["bash", str(task / "run.sh"), "--run-id", owned_id], log,
                         min(int(env["RECON2REASON_LOAD_TIMEOUT"]) + 60, seconds_left() - 120), env, record)
            require(rc == 0, "new owned service launch failed")
            state, _ = exact_state(task, owned_id)
            require(state.get("status") == "ready" and alive(state.get("supervisor_pid"), state.get("supervisor_token"))
                    and alive(state.get("server_pid"), state.get("server_token")), "new service not ready with exact owned tokens")
            copy_new(task / "runtime/state.json", control / "ready-state.json")
            budget = min(3000, int(seconds_left() - 90))
            require(budget >= 1500, "insufficient remaining benchmark budget after startup")
            command = report["rows"][0]["command"]
            prefix = command[:command.index("bench")]
            argv = [sys.executable, "-B", str(root / "scripts/t02_development_bench.py"), "--run",
                    "--model-path", report["model_path"], "--tokenizer-path", report["tokenizer_path"],
                    "--served-model-name", report["served_model_name"], "--image-size", "1024", "--host", "127.0.0.1", "--port", "30000",
                    "--vllm-command-json", json.dumps(prefix), "--output-dir", str(output), "--load", "16384:64",
                    "--timeout-seconds", str(min(1800, budget)), "--total-timeout-seconds", str(budget)]
            rc = bounded(argv, log, min(budget + 10, seconds_left() - 70), env, record)
            record("retry_driver_finished", exit_code=rc, retry_run_json_present=(output / "run.json").is_file())
            driver_code = rc
        finally:
            if started_launch:
                try:
                    state, _ = exact_state(task, owned_id)
                    if state.get("status") == "ready":
                        try:
                            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                            with opener.open("http://127.0.0.1:30000/metrics", timeout=2) as response:
                                raw = response.read(4 * 1024 * 1024 + 1)
                            require(len(raw) <= 4 * 1024 * 1024, "metrics response too large")
                            with (control / "metrics-before-stop.txt").open("xb") as target:
                                target.write(raw)
                        except (OSError, ValueError) as exc:
                            record("metrics_capture_error", error=str(exc))
                    rc = bounded(["bash", str(task / "stop.sh"), "--run-id", owned_id], log,
                                 max(1, min(50, seconds_left() - 5)), env, record)
                    clean = cleanup_snapshot(task, owned_id, 30000)
                    write_new(control / "cleanup-check.json", clean)
                    record("owned_cleanup", stop_exit_code=rc, **clean)
                    cleanup_ok = rc == 0 and clean["owned_service_cleanup_verified"]
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    record("owned_cleanup_error", error=str(exc), cleanup_verified=False)
            record("original_report_preserved", unchanged=sha(original) == original_sha)
            write_new(control / "control-events.json", {"events": events, "helper_sha256": sha(Path(__file__)), "hard_end_utc": HARD_END.isoformat(), "official_acceptance": False})
    return 0 if driver_code == 0 and cleanup_ok and sha(original) == original_sha else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/root/private_data/guoxin-hackathon-20261020-session003"))
    parser.add_argument("--log", type=Path, default=Path("/root/hg-recon-matrix-retry-003.txt"))
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        validated = validate(root)
        print(json.dumps({"mode": "run" if args.run else "plan_only", "missing_cells": MISSING,
                          "original_run_id": validated[4]["run_id"], "retry_output": str(validated[6]),
                          "hard_end_utc": HARD_END.isoformat(), "remaining_seconds": seconds_left(),
                          "per_cell_timeout_ceiling": 1800, "total_timeout_ceiling": 3000,
                          "official_acceptance": False, "filesystem_mutated": False}, ensure_ascii=False), flush=True)
        if not args.run:
            return 0
        require(not args.log.exists(), "helper log already exists; do not redirect outer launch to this path")
        return execute(root, args.log)
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc), "official_acceptance": False}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
