#!/usr/bin/env python3
"""Stdlib-only Linux supervisor and fail-closed official evaluation bridge."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

SCRIPT = Path(__file__).resolve()
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump(path: Path, data: dict) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def process_token(pid: int) -> str | None:
    """Linux start ticks protect against signaling a recycled PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def matches(pid: int, token: str | None) -> bool:
    return bool(token) and process_token(pid) == token


def healthy(url: str) -> bool:
    try:
        with OPENER.open(url + "/health", timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def configuration(task: Path) -> dict:
    integer = lambda name, default: int(os.environ.get(name, str(default)))
    framework = os.environ.get("FRAMEWORK", "sglang")
    if framework not in {"sglang", "vllm"} or os.environ.get("PRECISION", "bf16") != "bf16":
        raise ValueError("this task supports SGLang or vLLM with BF16 precision")
    default_extra = '["--enforce-eager"]' if framework == "vllm" else "[]"
    extra = json.loads(os.environ.get("EXTRA_ARGS_JSON", default_extra))
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("EXTRA_ARGS_JSON must be a JSON array of strings")
    protected = {"--served-model-name", "--host", "--port", "--dtype"}
    if framework == "sglang":
        protected.update({"--model-path", "--tp", "--tp-size", "--context-length", "--mem-fraction-static"})
    else:
        protected.update({"--model", "--tensor-parallel-size", "-tp", "--max-model-len", "--gpu-memory-utilization"})

    def overrides_core(item: str) -> bool:
        option = item.split("=", 1)[0].split(".", 1)[0].replace("_", "-")
        if option in protected:
            return True
        # vLLM's FlexibleArgumentParser accepts underscores and argparse prefixes.
        # Reject abbreviations as well as exact names so extra args cannot replace core settings.
        return option.startswith("--") and any(flag.startswith(option) for flag in protected if flag.startswith("--"))

    if any(overrides_core(item) for item in extra):
        raise ValueError("EXTRA_ARGS_JSON cannot override model, network, precision, TP, or memory settings")
    values = {
        "task": str(task), "framework": framework, "precision": "bf16",
        "model_id": os.environ.get("MODEL_ID", "BAAI/Recon2Reason-Reasoning-4B"),
        "model_path": os.environ.get("MODEL_PATH", "/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B"),
        "served_name": os.environ.get("SERVED_NAME", "recon2reason-reasoning-4b"),
        "host": "127.0.0.1", "port": integer("PORT", 30000), "tp": integer("TP", 1),
        "max_len": integer("MAX_LEN", 32768), "mem_frac": float(os.environ.get("MEM_FRAC", "0.85")),
        "load_timeout": integer("LOAD_TIMEOUT", 1800), "max_runtime": integer("MAX_RUNTIME_SECONDS", 7200),
        "stop_grace": integer("STOP_GRACE_SECONDS", 30), "extra": extra,
    }
    if not 1 <= values["port"] <= 65535 or values["tp"] < 1 or values["max_len"] < 17408:
        raise ValueError("invalid PORT/TP/MAX_LEN; MAX_LEN must contain 16K input plus 1K output")
    if not 0 < values["mem_frac"] < 1 or min(values["load_timeout"], values["max_runtime"], values["stop_grace"]) <= 0:
        raise ValueError("memory fraction must be between 0 and 1 and all timeouts must be positive")
    common_args = ["--served-model-name", values["served_name"], "--host", values["host"],
                   "--port", str(values["port"])]
    if framework == "sglang":
        values["command"] = [sys.executable, "-m", "sglang.launch_server", "--model-path", values["model_path"],
            *common_args, "--tp-size", str(values["tp"]), "--dtype", "bfloat16", "--context-length", str(values["max_len"]),
            "--mem-fraction-static", str(values["mem_frac"]), *extra]
    else:
        values["command"] = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", values["model_path"],
            *common_args, "--tensor-parallel-size", str(values["tp"]), "--dtype", "bfloat16", "--max-model-len", str(values["max_len"]),
            "--gpu-memory-utilization", str(values["mem_frac"]), *extra]
    values["url"] = f"http://127.0.0.1:{values['port']}"
    return values


def checked_run_id(value: str | None) -> str | None:
    if value is not None and (len(value) != 32 or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("--run-id must contain exactly32 lowercase hexadecimal characters")
    return value


@contextmanager
def control_lock(runtime: Path):
    """Serialize launcher/stop state decisions separately from the service lifetime lock."""
    import fcntl

    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def launch(task: Path, dry_run: bool, run_id: str | None = None) -> int:
    run_id = checked_run_id(run_id)
    config = configuration(task)
    if dry_run:
        if run_id is not None:
            config["run_id"] = run_id
        print(json.dumps({"dry_run": True, **config}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != "linux":
        return fail("actual serving requires the Linux SCNet container; use --dry-run locally")
    if not Path(config["model_path"]).is_dir():
        return fail(f"model directory does not exist: {config['model_path']}")
    runtime = task / "runtime"
    state_path = runtime / "state.json"
    with control_lock(runtime):
        if healthy(config["url"]):
            return fail("the chosen port already has a healthy service; refusing to reuse or stop it")
        if state_path.exists():
            previous = read(state_path)
            if matches(previous.get("supervisor_pid", 0), previous.get("supervisor_token")):
                return fail("this task already has a live supervisor; use stop.sh first")
            if matches(previous.get("server_pid", 0), previous.get("server_token")):
                return fail("the recorded server is still alive; inspect state before launching")
        config["run_id"] = run_id or uuid.uuid4().hex
        config_path = runtime / f"config-{config['run_id']}.json"
        # Caller identities are immutable; never replace an earlier run's configuration.
        with config_path.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with (runtime / "supervisor.log").open("a", encoding="utf-8") as log:
            worker = subprocess.Popen([sys.executable, str(SCRIPT), "supervise", "--config", str(config_path)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ready = False
    previous_term_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(_number, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        deadline = time.monotonic() + config["load_timeout"] + 15
        while time.monotonic() < deadline:
            if state_path.exists():
                state = read(state_path)
                if state.get("run_id") == config["run_id"]:
                    if state.get("status") == "ready":
                        ready = True
                        print(json.dumps(state, ensure_ascii=False, indent=2))
                        print("Service ready. The supervisor continues enforcing its runtime limit.")
                        return 0
                    if state.get("status") in {"failed", "stopped", "expired"}:
                        return fail(f"startup failed: {state.get('reason')}; see runtime/server.log", 1)
            if worker.poll() is not None:
                return fail("supervisor exited before readiness; see runtime/supervisor.log", 1)
            time.sleep(0.25)
        return fail("startup deadline exceeded; supervisor was asked to clean up", 1)
    except KeyboardInterrupt:
        return fail("startup interrupted; cleaning up the matching supervisor", 130)
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        if not ready and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=config["stop_grace"] + 10)
            except subprocess.TimeoutExpired:
                print("ERROR: supervisor cleanup did not finish; inspect state.json before retrying", file=sys.stderr)


def terminate_server(child: subprocess.Popen, grace: int) -> None:
    # A worker's server starts a new session; only its own process group is signaled.
    group = child.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        child.poll()
        time.sleep(0.1)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def supervise(config_path: Path) -> int:
    import fcntl

    config = read(config_path)
    runtime = Path(config["task"]) / "runtime"
    lock = (runtime / "service.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return fail("another supervisor holds this task lock", 1)
    state_path = runtime / "state.json"
    stopped = False

    def handle_stop(_number, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    state = {"run_id": config["run_id"], "status": "starting", "created_utc": stamp(),
        "supervisor_pid": os.getpid(), "supervisor_token": process_token(os.getpid()),
        "config_path": str(config_path), "url": config["url"], "max_runtime_seconds": config["max_runtime"],
        "framework": config["framework"], "precision": config["precision"], "command": config["command"]}
    child = None
    began = time.monotonic()
    exit_code = 1
    dump(state_path, state)
    print(json.dumps({"event": "starting", **state}), flush=True)
    try:
        with (runtime / "server.log").open("a", encoding="utf-8") as log:
            log.write(f"\n=== run {config['run_id']} at {stamp()} ===\n{json.dumps(config['command'])}\n")
            log.flush()
            child = subprocess.Popen(config["command"], cwd=config["task"], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(server_pid=child.pid, server_token=process_token(child.pid))
            dump(state_path, state)
            while True:
                elapsed = time.monotonic() - began
                if stopped:
                    state.update(status="stopped", reason="stop requested")
                    exit_code = 0
                    break
                if child.poll() is not None:
                    state.update(status="failed", reason=f"server exited with code {child.returncode}")
                    break
                if elapsed >= config["max_runtime"]:
                    state.update(status="expired", reason="maximum runtime reached")
                    exit_code = 0
                    break
                if state["status"] != "ready":
                    if elapsed >= config["load_timeout"]:
                        state.update(status="failed", reason="health-check timeout")
                        break
                    if healthy(config["url"]):
                        state.update(status="ready", ready_utc=stamp())
                        dump(state_path, state)
                        print(json.dumps({"event": "ready", "run_id": config["run_id"]}), flush=True)
                time.sleep(0.25)
    except Exception as exc:
        state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if child is not None:
            terminate_server(child, config["stop_grace"])
        state.update(ended_utc=stamp(), elapsed_seconds=round(time.monotonic() - began, 3))
        dump(state_path, state)
        print(json.dumps({"event": "finished", **state}), flush=True)
        lock.close()
    return exit_code


def stop(task: Path, run_id: str | None = None) -> int:
    run_id = checked_run_id(run_id)
    path = task / "runtime" / "state.json"
    if not path.exists():
        print("No state file; no process was signaled.")
        return 0
    with control_lock(path.parent):
        state = read(path)
        if run_id is not None and state.get("run_id") != run_id:
            return fail("runtime run_id differs from --run-id; refusing to stop another run")
        pid = state.get("supervisor_pid", 0)
        if not matches(pid, state.get("supervisor_token")):
            if matches(state.get("server_pid", 0), state.get("server_token")):
                return fail("supervisor identity is gone but its recorded server is alive; inspect state before cleanup")
            print("The recorded supervisor is no longer running; no process was signaled.")
            return 0
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(SCRIPT).encode() not in command or b"supervise" not in command or str(state["config_path"]).encode() not in command:
            return fail("supervisor command does not match this script and run config; refusing to signal")
        config = read(Path(state["config_path"]))
        if config["run_id"] != state["run_id"] or Path(config["task"]).resolve() != task:
            return fail("run config does not belong to this task")
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + config["stop_grace"] + 10
    while matches(pid, state["supervisor_token"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    if matches(pid, state["supervisor_token"]):
        return fail("supervisor did not finish cleanup; inspect runtime logs", 1)
    print("Stopped the matching task supervisor and its server process group.")
    return 0


def environment(task: Path) -> int:
    packages = {}
    for package in ["torch", "triton", "transformers", "tokenizers", "sglang", "sglang-das", "vllm", "vllm-plugin-das", "vllm-hcu"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {"captured_utc": stamp(), "python": sys.version, "image_requested": os.environ.get("IMAGE", ""),
        "image_candidate_name": os.environ.get("IMAGE_CANDIDATE_NAME", ""),
        "image_effective": None, "image_note": "Configured candidate only; this script cannot attest to the running container digest.",
        "card": os.environ.get("CARD", ""), "framework": os.environ.get("FRAMEWORK", ""),
        "precision": os.environ.get("PRECISION", ""), "packages": packages,
        "configuration": configuration(task), "gpu_available": False}
    for command in (["hy-smi"], ["cat", "/opt/dtk/.info/version"]):
        try:
            captured = subprocess.run(command, capture_output=True, text=True, timeout=30)
            result[" ".join(command)] = {"code": captured.returncode, "stdout": captured.stdout, "stderr": captured.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[" ".join(command)] = {"error": str(exc)}
    try:
        import torch
        result["hip"] = torch.version.hip
        result["gpu_available"] = torch.cuda.is_available()
        result["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError) as exc:
        result["torch_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gpu_available"] else 2


def t02_load_plan() -> list[dict]:
    """Template load prescription only; no client, model, network or measured results."""
    return [{"round": round_, "scored": round_ == 2, "input_tokens": length,
             "output_tokens": 1024, "concurrency": concurrency,
             "requests": max(8, concurrency * 2), "images_per_request": 1,
             "image_resolution": os.environ.get("IMAGE_RES") or None}
            for length in (4096, 16384) for concurrency in (1, 8, 64) for round_ in (1, 2)]


def official(task: Path, kind: str, dry_run: bool = False) -> int:
    """Preserve and execute externally supplied materials, without inventing their schema.

    Exit zero means the execution pipeline completed. Every result remains
    executed_uninterpreted until the actual task-specific scoring contract is integrated.
    """
    prefix = f"OFFICIAL_{kind.upper()}"
    required = [prefix + "_SCRIPT"] + (["OFFICIAL_BASELINE", "OFFICIAL_DATA_DIR"] if kind == "verify" else [])
    missing = [name for name in required
               if not os.environ.get(name)]
    script = Path(os.environ[prefix + "_SCRIPT"]).resolve() if os.environ.get(prefix + "_SCRIPT") else None
    baseline = Path(os.environ["OFFICIAL_BASELINE"]).resolve() if os.environ.get("OFFICIAL_BASELINE") else None
    data = Path(os.environ["OFFICIAL_DATA_DIR"]).resolve() if os.environ.get("OFFICIAL_DATA_DIR") else None
    problems = ["missing " + name for name in missing]
    for label, path in (("script", script), ("baseline", baseline)):
        if path is not None and (not path.is_file() or path.stat().st_size == 0):
            problems.append(label + " must be a nonempty regular file")
    if data is not None and (not data.is_dir() or not any(data.iterdir())):
        problems.append("official data directory must be nonempty")
    extra = json.loads(os.environ.get(prefix + "_ARGS_JSON", "[]"))
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError(prefix + "_ARGS_JSON must be an array of strings")
    selfcheck_args = json.loads(os.environ.get("OFFICIAL_SELFCHECK_ARGS_JSON", '["--selfcheck"]'))
    if not isinstance(selfcheck_args, list) or not selfcheck_args or not all(isinstance(item, str) for item in selfcheck_args):
        raise ValueError("OFFICIAL_SELFCHECK_ARGS_JSON must be a nonempty array of strings")
    default_outputs = (["verify_result.json"] if kind == "verify" else
                       ["bench/round1.jsonl", "bench/round2.jsonl", "bench/summary.md"] +
                       [f"bench/round{round_}_{length}_1024_c{concurrency}.log"
                        for round_ in (1, 2) for length in (4096, 16384) for concurrency in (1, 8, 64)])
    names = json.loads(os.environ.get(prefix + "_OUTPUTS_JSON", json.dumps(default_outputs)))
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(prefix + "_OUTPUTS_JSON must be a nonempty array of relative filenames")
    output_files = []
    for name in names:
        relative = Path(name)
        path = (task / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(task) or path == task:
            raise ValueError("official output must stay inside the task: " + name)
        if path.is_dir() or path in {script, baseline} or (data is not None and path.is_relative_to(data)):
            raise ValueError("official output cannot replace an input or directory: " + name)
        output_files.append(path)
    if len(set(output_files)) != len(output_files):
        raise ValueError("duplicate official output paths")
    timeout = float(os.environ.get("OFFICIAL_TIMEOUT_SECONDS", "7200" if kind == "bench" else "3600"))
    selfcheck_timeout = float(os.environ.get("OFFICIAL_SELFCHECK_TIMEOUT_SECONDS", "120"))
    if not 0 < timeout < float("inf") or not 0 < selfcheck_timeout < float("inf"):
        raise ValueError("official execution timeouts must be positive finite seconds")
    base_command = ([sys.executable, str(script)] if script and script.suffix == ".py"
                    else ["bash", str(script)] if script else None)
    command = [*base_command, *extra] if base_command else None
    plan = {"kind": kind, "dry_run": dry_run, "official_acceptance": False,
            "identity_expected": {key: os.environ.get(key) for key in ("MODEL_ID", "CARD", "FRAMEWORK", "PRECISION")},
            "script": str(script) if script else None, "baseline": str(baseline) if baseline else None,
            "data": str(data) if data else None, "command": command,
            "selfcheck_command": [*base_command, *selfcheck_args] if kind == "verify" and base_command else None,
            "timeout_seconds": timeout, "selfcheck_timeout_seconds": selfcheck_timeout,
            "expected_outputs": [str(path.relative_to(task)) for path in output_files],
            "problems": problems, "material_schema": "not_assumed",
            "result_interpretation": "execution_only; task-specific scoring contract still required",
            "service_health_checked": False, "subprocess_executed": False, "filesystem_mutated": False}
    if kind == "bench":
        plan["t02_template_load_plan"] = t02_load_plan()
        plan["template_requests_total"] = sum(item["requests"] for item in plan["t02_template_load_plan"])
        plan["image_resolution_confirmed"] = False
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 2 if problems else 0
    if problems:
        return fail("official materials missing or invalid: " + "; ".join(problems))
    assert script is not None
    if not healthy(configuration(task)["url"]):
        return fail("serving is not healthy; no official evaluation was started")
    runtime = task / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    run_dir = runtime / f"official-{kind}-{uuid.uuid4().hex}"
    run_dir.mkdir()
    record_path = run_dir / "record.json"
    materials = run_dir / "materials"
    materials.mkdir()
    saved_sources = {}
    for label, path in (("script", script), ("baseline", baseline)):
        if path is None:
            continue
        raw = path.read_bytes()
        (materials / (label + path.suffix)).write_bytes(raw)
        saved_sources[label] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    record = {**plan, "dry_run": False, "started_utc": stamp(), "status": "preparing",
              "materials": saved_sources, "steps": [], "previous_outputs": [], "outputs": [],
              "service_health_checked": True, "filesystem_mutated": True}
    dump(record_path, record)

    def finish(status: str, reason: str, code: int) -> int:
        record.update(status=status, reason=reason, ended_utc=stamp())
        dump(record_path, record)
        print(json.dumps({"status": status, "official_acceptance": False, "reason": reason,
                          "record": str(record_path)}, ensure_ascii=False))
        return code

    def unchanged() -> bool:
        return all(Path(item["path"]).is_file() and hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
                   for item in saved_sources.values())

    env = os.environ.copy()
    if baseline is not None:
        env.update(OFFICIAL_BASELINE=str(baseline), BASELINE_PATH=str(baseline))
    if data is not None:
        env.update(OFFICIAL_DATA_DIR=str(data), DATA_DIR=str(data))
    if kind == "bench":
        env["BENCH_MODE"] = "image"

    def step(label: str, argv: list[str], seconds: float) -> int:
        log = run_dir / (label + ".log")
        entry = {"step": label, "command": argv, "started_utc": stamp(), "log": str(log), "timeout_seconds": seconds}
        record["steps"].append(entry)
        record["subprocess_executed"] = True
        dump(record_path, record)
        with log.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(argv, cwd=task, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=sys.platform == "linux")
            except OSError as exc:
                entry.update(exit_code=127, ended_utc=stamp(), error=str(exc))
                return 127
            try:
                code = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                code = 124
                entry["timed_out"] = True
            except KeyboardInterrupt:
                code = 130
                entry["interrupted"] = True
            finally:
                # Reap the direct child even when wait() was interrupted. Also
                # remove descendants left behind by an exited Linux evaluator.
                if sys.platform == "linux":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
        entry.update(exit_code=code, ended_utc=stamp(), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        dump(record_path, record)
        return code

    def snapshot_outputs(skip_unchanged_previous: bool = False) -> bool:
        previous = {item["path"]: item["sha256"] for item in record["previous_outputs"]}
        for path in output_files:
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(task):
                    return False
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if skip_unchanged_previous and previous.get(str(path)) == digest:
                    continue
                saved = run_dir / "outputs" / path.relative_to(task)
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(raw)
                record["outputs"].append({"path": str(path), "snapshot": str(saved),
                                          "bytes": len(raw), "sha256": digest})
        return True

    # Preserve pre-existing results before selfcheck could write any output.
    archive = run_dir / "previous-outputs"
    for path in output_files:
        if path.exists():
            saved = archive / path.relative_to(task)
            saved.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copy2(path, saved)
            record["previous_outputs"].append({"path": str(path), "archive": str(saved), "sha256": digest})
    dump(record_path, record)
    if kind == "verify":
        code = step("selfcheck", plan["selfcheck_command"], selfcheck_timeout)
        if record["steps"][-1].get("interrupted"):
            safe = snapshot_outputs(skip_unchanged_previous=True)
            record["partial_outputs_safe"] = safe
            return finish("interrupted", "selfcheck interrupted; partial new outputs and previous outputs preserved separately", 130)
        if not unchanged():
            return finish("materials_changed", "script or baseline changed during selfcheck; evaluation not started", 2)
        if code:
            return finish("selfcheck_failed", "selfcheck failed or timed out; evaluation not started", code if code > 0 else 1)
    # Only after successful selfcheck remove stale working copies; archives stay intact.
    for path in output_files:
        if path.exists():
            if path.is_symlink() or not path.resolve().is_relative_to(task):
                return finish("unsafe_output", "selfcheck changed an output path to leave the task directory", 2)
            path.unlink()
    dump(record_path, record)
    code = step("evaluation" if kind == "verify" else "benchmark", command, timeout)
    safe = snapshot_outputs()
    if record["steps"][-1].get("interrupted"):
        record["partial_outputs_safe"] = safe
        return finish("interrupted", "external execution interrupted; partial new outputs and previous outputs preserved separately", 130)
    if not safe:
        return finish("unsafe_output", "script produced an output outside the task directory", 2)
    if not unchanged():
        return finish("materials_changed", "script or baseline changed during execution", 2)
    if code:
        return finish("execution_failed", "external script failed or timed out; partial outputs preserved", code if code > 0 else 1)
    if len(record["outputs"]) != len(output_files) or any(item["bytes"] == 0 for item in record["outputs"]):
        return finish("outputs_incomplete", "missing, empty or stale output; previous outputs are preserved separately", 2)
    if kind == "verify":
        # The public template has these identity fields, but the actual T02/T04 schema
        # has not been supplied. Check explicit contradictions without guessing metrics.
        try:
            result = read(output_files[0])
        except (ValueError, UnicodeError):
            result = None
        if isinstance(result, dict):
            mismatch = {key: {"expected": expected, "reported": result[key]}
                        for key, expected in plan["identity_expected"].items()
                        if key in result and result[key] != expected}
            if mismatch:
                record["identity_mismatch"] = mismatch
                return finish("identity_mismatch", "result explicitly identifies a different task", 1)
            if result.get("passed") is False:
                return finish("reported_failure", "external result explicitly reports passed=false", 1)
    return finish("executed_uninterpreted", "fresh outputs preserved; no task-specific score/acceptance interpretation performed", 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "supervise", "stop", "env", "verify", "bench"])
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", help="optional exact caller identity for launch/stop;32 lowercase hex characters")
    args = parser.parse_args()
    try:
        if args.run_id is not None and args.action not in {"launch", "stop"}:
            return fail("--run-id is only supported for launch/stop")
        if args.action == "supervise":
            if not args.config:
                return fail("--config is required for the internal supervisor")
            return supervise(args.config.resolve())
        if not args.task_dir:
            return fail("--task-dir is required")
        task = args.task_dir.resolve()
        if args.action == "launch":
            return launch(task, args.dry_run, args.run_id)
        if args.action == "stop":
            return stop(task, args.run_id)
        if args.action == "env":
            return environment(task)
        return official(task, args.action, args.dry_run)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
