#!/usr/bin/env python3
"""Bounded RoboBrain/ASR development checks; default plan only, never official PASS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_state(task):
    try:
        value = json.loads((task / "runtime/state.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def process_matches(state):
    try:
        raw = Path(f"/proc/{int(state['supervisor_pid'])}/stat").read_text().rsplit(")", 1)[1].split()
        return raw[0] != "Z" and raw[19] == state.get("supervisor_token")
    except (OSError, ValueError, KeyError, IndexError):
        return False


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def build_plan(args):
    if not 120 <= args.timeout_seconds <= 3600:
        raise ValueError("total timeout must be120..3600 seconds, including cleanup reserve")
    if not 60 <= args.load_timeout <= args.model_runtime_seconds <= 1800:
        raise ValueError("require60 <= load-timeout <= model-runtime-seconds <=1800")
    project = args.project.resolve()
    output = (args.output or project / "evidence/session003/robo-asr").resolve()
    if output == project or not output.is_relative_to(project):
        raise ValueError("output must be a new directory inside this project")
    venv = (args.asr_venv or project / "tasks/qwen3-asr-bw-vllm/runtime/audio-venv").resolve()
    ports = {"robo": int(os.environ.get("ROBOBRAIN_PORT", "30001")),
             "asr": int(os.environ.get("QWEN_ASR_PORT", "30002"))}
    if len(set(ports.values())) != 2 or any(not 1 <= port <= 65535 for port in ports.values()):
        raise ValueError("require two distinct valid loopback ports")
    return {"schema_version": 1, "session": "session003", "mode": "run" if args.run else "plan_only",
            "project": str(project), "output": str(output), "base_python": args.base_python,
            "asr_venv": str(venv), "ports": ports, "total_timeout_seconds": args.timeout_seconds,
            "model_runtime_seconds": args.model_runtime_seconds, "load_timeout": args.load_timeout,
            "cleanup_reserve_seconds": 120,
            "official_accuracy": False, "official_performance": False, "official_acceptance": False,
            "scope": "development:24 numerical cases,7 synthetic image requests,3 publisher-derived WAVs,SSE output",
            "realtime_audio_input_verified": False, "network": "explicit --run only; clients use127.0.0.1",
            "filesystem_mutated": False, "phases": [],
            "steps_planned": {"robo": ["env_check", "24-case GPU numerical", "launch/readiness", "7 heldout requests", "stop/archive"],
                              "asr": ["check existing isolated venv", "env_check", "launch/readiness", "zh/en/39s WAV", "zh SSE", "stop/archive"]}}


class Runner:
    def __init__(self, plan):
        self.report = plan
        self.project = Path(plan["project"])
        self.output = Path(plan["output"])
        self.started = time.monotonic()
        self.deadline = self.started + plan["total_timeout_seconds"]
        self.work_deadline = self.deadline - self.report["cleanup_reserve_seconds"]
        self.interrupted = False

    def save(self):
        target = self.output / "report.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(target)

    def snapshot(self, task, destination, run_id=None):
        destination.mkdir(parents=True, exist_ok=True)
        paths = [task / "env.log", task / "runtime/state.json", task / "runtime/server.log", task / "runtime/supervisor.log"]
        if run_id:
            paths.append(task / "runtime" / ("config-" + run_id + ".json"))
        copied = []
        for path in paths:
            if path.is_file() and not path.is_symlink():
                target = destination / path.relative_to(task)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append({"source": str(path), "snapshot": str(target), "sha256": sha(target),
                               "bytes": target.stat().st_size})
        return copied

    @staticmethod
    def reap(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def step(self, phase, name, argv, env, timeout, cleanup=False):
        deadline = self.deadline if cleanup else self.work_deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session deadline reached before " + name)
        item = {"name": name, "command": argv, "started_utc": stamp(), "status": "running"}
        phase["steps"].append(item)
        log_path = self.output / phase["name"] / (name + ".log")
        self.save()
        process = None
        code = 127
        try:
            with log_path.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(argv, cwd=self.project, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    code = process.wait(timeout=min(timeout, remaining))
                except subprocess.TimeoutExpired:
                    code = 124
                except KeyboardInterrupt:
                    self.interrupted = True
                    code = 130
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
        except OSError as exc:
            item["error"] = str(exc)
        item.update(exit_code=code, ended_utc=stamp(), status="complete" if code == 0 else "failed",
                    log=str(log_path))
        if log_path.is_file():
            item["log_sha256"] = sha(log_path)
        self.save()
        if code:
            raise RuntimeError(f"{name} failed with exit{code}; see{log_path}")

    def launch(self, phase, task, env):
        previous = read_state(task).get("run_id")
        phase["previous_run_id"] = previous
        log_path = self.output / phase["name"] / "launch.log"
        item = {"name": "launch", "started_utc": stamp(), "status": "running", "log": str(log_path)}
        phase["steps"].append(item)
        phase["launch_attempted"] = True
        self.save()
        code = 127
        try:
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(["bash", str(task / "run.sh")], cwd=self.project, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
                phase["launcher_pid"] = process.pid
                ready_deadline = min(self.work_deadline, time.monotonic() + self.report["load_timeout"] + 20)
                try:
                    while time.monotonic() < ready_deadline:
                        state = read_state(task)
                        if state.get("run_id") and state["run_id"] != previous:
                            phase["owned_run_id"] = state["run_id"]
                            if state.get("status") == "ready":
                                code = process.wait(timeout=15)
                                if code:
                                    raise RuntimeError("launcher failed despite ready state")
                                phase["ready_state"] = state
                                phase["ready_snapshot"] = self.snapshot(task, self.output / phase["name"] / "ready", state["run_id"])
                                item.update(status="ready", exit_code=code)
                                return
                            if state.get("status") in {"failed", "stopped", "expired"}:
                                raise RuntimeError("service startup ended: " + str(state.get("reason")))
                        if process.poll() is not None:
                            code = process.returncode
                            raise RuntimeError("launcher exited before a fresh ready state: " + str(code))
                        time.sleep(0.25)
                    raise TimeoutError("service readiness deadline reached")
                except KeyboardInterrupt:
                    self.interrupted = True
                    code = 130
                    raise
                finally:
                    self.reap(process)
        finally:
            # A launch can fail just before the next poll sees its fresh state.
            # The launcher's own cleanup has completed above; retain that run's
            # identity for an additional exact-run stop/check in phase.finally.
            state = read_state(task)
            if state.get("run_id") and state["run_id"] != previous:
                phase.setdefault("owned_run_id", state["run_id"])
            item.update(ended_utc=stamp(), exit_code=code)
            if item["status"] != "ready":
                item["status"] = "failed"
            if log_path.is_file():
                item["log_sha256"] = sha(log_path)
            self.save()

    def cleanup(self, phase, task, env):
        owned = phase.get("owned_run_id")
        state = read_state(task)
        phase["cleanup_started_utc"] = stamp()
        if owned and state.get("run_id") == owned:
            try:
                self.step(phase, "stop", ["bash", str(task / "stop.sh")], env, 45, cleanup=True)
            except (RuntimeError, TimeoutError, OSError) as exc:
                phase["stop_error"] = str(exc)
        elif owned:
            phase["stop_error"] = "runtime ownership changed; refusing to stop another run"
        until = min(self.deadline, time.monotonic() + 10)
        while time.monotonic() < until:
            state = read_state(task)
            if not port_open(phase["port"]) and not (state.get("run_id") == owned and process_matches(state)):
                break
            time.sleep(0.25)
        state = read_state(task)
        phase["cleanup_verified"] = (not port_open(phase["port"]) and
                                     not (state.get("run_id") == owned and process_matches(state)) and
                                     not (owned and state.get("run_id") != owned))
        phase["final_state"] = state
        phase["final_snapshots"] = self.snapshot(task, self.output / phase["name"] / "stopped", owned)
        phase["cleanup_ended_utc"] = stamp()
        self.save()

    def phase(self, name):
        task = self.project / "tasks" / ("robobrain-bw-vllm" if name == "robo" else "qwen3-asr-bw-vllm")
        phase = {"name": name, "port": self.report["ports"][name], "status": "running", "steps": [],
                 "started_utc": stamp(), "official_acceptance": False}
        self.report["phases"].append(phase)
        directory = self.output / name
        directory.mkdir()
        phase["previous_snapshots"] = self.snapshot(task, directory / "before", read_state(task).get("run_id"))
        env = os.environ.copy()
        env.pop("VLLM_PLUGINS", None)
        env["PYTHONPATH"] = str(self.project / "scripts") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["SCNET_BASE_PYTHON"] = self.report["base_python"]
        prefix = "ROBOBRAIN" if name == "robo" else "QWEN_ASR"
        env.update({prefix + "_MAX_RUNTIME_SECONDS": str(self.report["model_runtime_seconds"]),
                    prefix + "_LOAD_TIMEOUT": str(self.report["load_timeout"]),
                    prefix + "_PORT": str(phase["port"])})
        env[prefix + "_PYTHON_BIN"] = self.report["base_python"] if name == "robo" else str(Path(self.report["asr_venv"]) / "bin/python")
        base_python = self.report["base_python"]
        try:
            if port_open(phase["port"]) or process_matches(read_state(task)):
                raise RuntimeError("existing service/supervisor detected; refusing to reuse or stop it")
            if name == "asr":
                samples = self.project / "fixtures/asr/development-20261020"
                manifest = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
                expected = {Path(row["path"]).name: row["sha256"] for row in manifest["samples"]}
                phase["audio_inputs"] = []
                for filename in ("zh_16000_pcm16.wav", "en_16000_pcm16.wav", "zh_en_repeated_39s.wav"):
                    actual = sha(samples / filename)
                    if actual != expected.get(filename):
                        raise ValueError("ASR development fixture hash mismatch: " + filename)
                    phase["audio_inputs"].append({"path": str(samples / filename), "sha256": actual})
                self.step(phase, "venv-check", ["bash", str(task / "prepare.sh"), "--check-existing", self.report["asr_venv"]], env, 180)
            self.step(phase, "env-check", ["bash", str(task / "env_check.sh")], env, 120)
            if name == "robo":
                env.update(ROBOBRAIN_PROFILE_WIDTH="1024", ROBOBRAIN_PROFILE_HEIGHT="1024", ROBOBRAIN_MAX_NUM_SEQS="4")
                self.step(phase, "numerical", [base_python, "-B", "scripts/validate_robobrain_vision_math_compat.py",
                                               "--max-length", "2049", "--output", str(directory / "numerical.json")], env, 300)
                numerical = json.loads((directory / "numerical.json").read_text(encoding="utf-8"))
                if (not numerical.get("gpu_numerical_verified") or len(numerical.get("cases", [])) != 24
                        or not all(case.get("passed") for case in numerical["cases"])):
                    raise RuntimeError("expected24 completed GPU numerical cases")
            self.launch(phase, task, env)
            if name == "robo":
                self.step(phase, "heldout", [base_python, "-B", "scripts/robobrain_heldout_fixtures.py",
                                             "--output-dir", str(directory / "heldout"), "--base-url", f"http://127.0.0.1:{phase['port']}",
                                             "--model", "robobrain2-5-8b-nv", "--timeout", "120", "--run"], env, 900)
            else:
                samples = self.project / "fixtures/asr/development-20261020"
                for label, filename, long in (("zh", "zh_16000_pcm16.wav", False),
                                              ("en", "en_16000_pcm16.wav", False),
                                              ("long", "zh_en_repeated_39s.wav", True)):
                    command = ["bash", str(task / "smoke.sh"), "--audio", str(samples / filename),
                               "--timeout", "180", "--log", str(directory / (label + ".jsonl"))]
                    if long:
                        command.append("--long-audio")
                    self.step(phase, label, command, env, 200)
                self.step(phase, "stream", [base_python, "-B", "scripts/qwen3_asr_stream_smoke.py",
                                            "--audio", str(samples / "zh_16000_pcm16.wav"),
                                            "--base-url", f"http://127.0.0.1:{phase['port']}", "--model", "qwen3-asr-1.7b",
                                            "--timeout", "180", "--run", "--log", str(directory / "zh-stream.jsonl")], env, 200)
            phase["status"] = "development_checks_completed"
        except KeyboardInterrupt:
            self.interrupted = True
            phase.update(status="interrupted", error="keyboard interrupt")
        except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, TypeError) as exc:
            phase.update(status="failed", error=str(exc))
        finally:
            if phase.get("launch_attempted"):
                self.cleanup(phase, task, env)
            else:
                phase["cleanup_verified"] = not port_open(phase["port"]) and not process_matches(read_state(task))
                phase["final_snapshots"] = self.snapshot(task, directory / "stopped", read_state(task).get("run_id"))
            phase["ended_utc"] = stamp()
            if not phase.get("cleanup_verified"):
                phase["status"] = "failed_cleanup_unconfirmed"
            self.save()

    def run(self):
        if sys.platform != "linux":
            raise ValueError("--run requires the native Linux image; default plan works locally")
        if self.output.exists() or self.output.is_symlink():
            raise ValueError("output exists; refusing to overwrite previous evidence")
        self.output.mkdir(parents=True, exist_ok=False)
        self.report.update(started_utc=stamp(), filesystem_mutated=True, status="running")
        files = [Path(__file__)] + [self.project / "scripts" / name for name in
                  ("recon2reason_runtime.py", "robobrain_vision_math_compat.py", "validate_robobrain_vision_math_compat.py",
                   "robobrain_heldout_fixtures.py", "qwen3_asr_smoke.py", "qwen3_asr_stream_smoke.py",
                   "qwen_asr_tf55_compat.py", "prepare_qwen_asr_task.py")]
        for slug in ("robobrain-bw-vllm", "qwen3-asr-bw-vllm"):
            files += list((self.project / "tasks" / slug).glob("*.sh"))
            files += [self.project / "tasks" / slug / "task.json"]
        files += [self.project / "tasks/qwen3-asr-bw-vllm" / name for name in
                  ("requirements-extra.txt", "native-stack.json", "compat/qwen_asr_tf55_compat-0.0.1.dist-info/entry_points.txt")]
        manifest = self.project / "fixtures/asr/development-20261020/manifest.json"
        if manifest.is_file():
            files.append(manifest)
        self.report["sources"] = []
        for path in files:
            if path.is_file() and not path.is_symlink():
                try:
                    relative = path.relative_to(self.project)
                except ValueError:
                    relative = Path("executed-driver") / path.name
                target = self.output / "sources" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                self.report["sources"].append({"path": str(path), "snapshot": str(target), "sha256": sha(target)})
        self.save()
        try:
            for name in ("robo", "asr"):
                if self.interrupted or time.monotonic() >= self.work_deadline:
                    self.report["phases"].append({"name": name, "status": "not_attempted_deadline_or_interrupt"})
                    continue
                if self.report["phases"] and not self.report["phases"][-1].get("cleanup_verified", True):
                    self.report["phases"].append({"name": name, "status": "not_attempted_cleanup_unconfirmed"})
                    continue
                self.phase(name)
        except KeyboardInterrupt:
            self.interrupted = True
            self.report["interrupted_between_phases"] = True
        success = len(self.report["phases"]) == 2 and all(p["status"] == "development_checks_completed" for p in self.report["phases"])
        self.report.update(status="development_checks_completed" if success else "failed_or_incomplete",
                           ended_utc=stamp(), wall_seconds=time.monotonic() - self.started)
        self.save()
        print(json.dumps({"report": str(self.output / "report.json"), "status": self.report["status"],
                          "official_acceptance": False}), flush=True)
        return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asr-venv", type=Path)
    parser.add_argument("--base-python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--model-runtime-seconds", type=int, default=1800)
    parser.add_argument("--load-timeout", type=int, default=600)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if not args.run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        previous_term = signal.getsignal(signal.SIGTERM)

        def interrupt(_number, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        try:
            return Runner(plan).run()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "official_acceptance": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
