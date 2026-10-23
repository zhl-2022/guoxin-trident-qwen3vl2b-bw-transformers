#!/usr/bin/env python3
"""Repair only missing reviewed CPU wheels in a SHA-bound partial004c venv.

The original bootstrap record/logs stay unchanged. Every attempt gets a new
directory and records its own pip/help/native checks, including failed help.
No SGLang/Torch/native reinstall, dependency resolution, or model launch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ALLOWED = {"orjson", "pybase64", "setproctitle", "msgspec", "partial-json-parser", "interegular"}
SGLANG_WHEEL_SHA = "4b88edfd3870d7dc8dd880f97d104f637baf105538827f968483d6cac56258f8"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def native_snapshot(expected):
    output = {}
    for name, version in expected.items():
        d = md.distribution(name)
        if d.version != version:
            raise ValueError("native version changed: " + name)
        f = next((x for x in d.files or () if str(x).endswith(".dist-info/METADATA")), None)
        output[name] = {"version": d.version, "root": str(d.locate_file("")),
                        "metadata_sha256": digest(Path(d.locate_file(f)).read_bytes()) if f else None}
    return output


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_inputs(record_path, record_sha, lock_path, lock_sha):
    raw, lockraw = record_path.read_bytes(), lock_path.read_bytes()
    if digest(raw) != record_sha or digest(lockraw) != lock_sha:
        raise ValueError("bootstrap or CPU lock SHA differs from explicitly reviewed bytes")
    old, wheels = json.loads(raw), json.loads(lockraw)
    target = Path(old["venv"])
    if not target.is_absolute() or not (target / "pyvenv.cfg").is_file() or not (target / "bin/python").is_file():
        raise ValueError("bootstrap must identify a real existing absolute venv")
    if record_path.absolute() != target / "client-bootstrap-evidence/record.json":
        raise ValueError("record must be the original evidence record inside the exact target venv")
    if old.get("native_unchanged") is not True or old.get("native_before") != old.get("native_after"):
        raise ValueError("original bootstrap has not proved native preservation")
    if not any(x.get("label") == "pip" and x.get("exit_code") == 0 for x in old.get("steps", [])):
        raise ValueError("original SGLang installation did not complete")
    sg = [x for x in old.get("wheel_lock", []) if x.get("name") == "sglang"]
    if len(sg) != 1 or sg[0].get("version") != "0.5.12" or sg[0].get("sha256") != SGLANG_WHEEL_SHA:
        raise ValueError("original SGLang wheel identity differs")
    for step in old.get("steps", []):
        if step.get("label") in {"pip", "help"}:
            log = record_path.parent / (step["label"]+".log")
            if digest(log.read_bytes()) != step["log_sha256"]:
                raise ValueError("original bootstrap log changed")
    if not isinstance(wheels, list) or not wheels:
        raise ValueError("explicit nonempty CPU lock required")
    seen = set()
    for w in wheels:
        if not isinstance(w, dict) or set(w) != {"name", "version", "url", "sha256"}:
            raise ValueError("CPU lock entry requires exact name/version/url/sha256")
        name = w["name"]
        if name not in ALLOWED or name in seen or name in old["protected_native"]:
            raise ValueError("unreviewed, duplicate or protected wheel")
        seen.add(name)
        url, sha = w["url"], w["sha256"]
        if not isinstance(url, str) or not url.startswith("https://files.pythonhosted.org/packages/") or not url.endswith(".whl") or any(x.isspace() for x in url):
            raise ValueError("official explicit PyPI wheel required")
        filename = url.rsplit("/", 1)[1]
        if filename.split("-", 1)[0].replace("_", "-") != name or filename.split("-")[1] != w["version"]:
            raise ValueError("wheel filename/name/version mismatch")
        if not isinstance(sha, str) or len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha):
            raise ValueError("exact CPU SHA256 required")
    return old, wheels, target


def run(argv, log, timeout, env):
    child = None
    begin = time.monotonic()
    with log.open("xb") as stream:
        try:
            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                     stdin=subprocess.DEVNULL, start_new_session=sys.platform == "linux")
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                code = 124
            except KeyboardInterrupt:
                code = 130
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
    return {"argv": argv, "exit_code": code, "wall_seconds": round(time.monotonic()-begin, 6),
            "log": log.name, "log_sha256": digest(log.read_bytes())}


def client_snapshot(python, protected, names, folder, phase, env):
    # Metadata only; importing this helper does not import an accelerator library.
    code = ("import json,sys,importlib.metadata as m,importlib.util as u;"
            "s=u.spec_from_file_location('repair_snapshot_probe'," + repr(str(Path(__file__).resolve())) + ");"
            "probe_module=u.module_from_spec(s);s.loader.exec_module(probe_module);"
            "d={};\nfor n in " + repr(["sglang", *names]) + ":\n"
            " try:\n  distribution=m.distribution(n);d[n]={'version':distribution.version,'root':str(distribution.locate_file(''))}\n"
            " except m.PackageNotFoundError:d[n]=None\n"
            "print(json.dumps({'native':probe_module.native_snapshot(" + repr(protected) + "),'packages':d}))")
    log = folder / (phase+"-client-metadata.log")
    result = run([python, "-B", "-c", code], log, 30, env)
    if result["exit_code"]:
        raise ValueError("client metadata probe failed: " + log.name)
    return json.loads(log.read_text())


def apply(old, wheels, target, record_path, lock_path):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10) or sys.prefix != sys.base_prefix:
        raise ValueError("repair must use the Linux Python3.10 native base interpreter")
    expected = old["protected_native"]
    before = native_snapshot(expected)
    if before != old["native_before"]:
        raise ValueError("base native protected snapshot differs from original004c")
    folder = target / "client-repair-evidence" / ("attempt-"+uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "original-bootstrap-record.json").write_bytes(record_path.read_bytes())
    (folder / "reviewed-cpu-wheels.json").write_bytes(lock_path.read_bytes())
    (folder / "repair_sglang_client.py").write_bytes(Path(__file__).read_bytes())
    state = {"created_utc": stamp(), "status": "preparing", "bootstrap_sha256": digest(record_path.read_bytes()),
             "lock_sha256": digest(lock_path.read_bytes()), "target": str(target), "native_before": before,
             "steps": [], "compatibility_proven": False, "scope": "CPU dependency repair and help probe only; no model/service launched"}
    write(folder / "record.json", state)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_") and k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    python = str(target / "bin/python")  # Preserve venv interpreter symlink path.
    try:
        client = client_snapshot(python, expected, [w["name"] for w in wheels], folder, "before", env)
        state["client_before"] = client
        sg = client["packages"]["sglang"]
        if client["native"] != before or not sg or sg["version"] != "0.5.12" or not Path(sg["root"]).resolve().is_relative_to(target.resolve()):
            raise ValueError("client native/SGLang identity differs from isolated004c")
        selected = [w for w in wheels if client["packages"][w["name"]] is None]
        state["reused_packages"] = {w["name"]: client["packages"][w["name"]] for w in wheels if client["packages"][w["name"]] is not None}
        state["selected_missing_wheels"] = selected
        lock = folder / "missing-wheels.txt"
        lock.write_text("".join(f"{w['name']} @ {w['url']} --hash=sha256:{w['sha256']}\n" for w in selected))
        if selected:
            pip = run([python, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
                       "install", "--no-deps", "--require-hashes", "--only-binary=:all:", "-r", str(lock)], folder / "pip.log", 180, env)
            state["steps"].append({"label": "pip", **pip})
            if pip["exit_code"]:
                raise ValueError("reviewed missing CPU wheel installation failed")
        help_result = run([python, "-B", "-m", "sglang.bench_serving", "--help"], folder / "help.log", 60, env)
        state["steps"].append({"label": "help", **help_result})
        state["status"] = "help_complete" if help_result["exit_code"] == 0 else "help_failed"
        after = client_snapshot(python, expected, [w["name"] for w in wheels], folder, "after", env)
        state["client_after"] = after
        if after["native"] != before or after["packages"]["sglang"] != sg:
            raise ValueError("repair altered protected native or installed SGLang identity")
        for w in selected:
            p = after["packages"][w["name"]]
            if not p or p["version"] != w["version"] or not Path(p["root"]).resolve().is_relative_to(target.resolve()):
                raise ValueError("CPU wheel did not install in isolated venv as locked")
        state["compatibility_proven"] = help_result["exit_code"] == 0
        return 0 if state["compatibility_proven"] else 1
    except BaseException as exc:
        state.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        try:
            state["native_after"] = native_snapshot(expected)
            state["native_unchanged"] = before == state["native_after"]
        except Exception as exc:
            state.update(native_unchanged=False, native_after_error=str(exc))
        if not state["native_unchanged"]:
            state["compatibility_proven"] = False
            state["status"] = "native_changed"
        state["ended_utc"] = stamp()
        write(folder / "record.json", state)
        print(json.dumps({"record": str(folder / "record.json"), "status": state["status"], "native_unchanged": state["native_unchanged"]}))
        if not state["native_unchanged"]:
            raise ValueError("protected base native identity changed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-record", required=True, type=Path)
    p.add_argument("--bootstrap-sha256", required=True)
    p.add_argument("--cpu-lock", required=True, type=Path)
    p.add_argument("--cpu-lock-sha256", required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args(argv)
    try:
        old, wheels, target = validate_inputs(a.bootstrap_record, a.bootstrap_sha256, a.cpu_lock, a.cpu_lock_sha256)
        if not a.apply:
            print(json.dumps({"mode": "plan_only", "target": str(target), "wheels": wheels, "filesystem_mutated": False}))
            return 0
        return apply(old, wheels, target, a.bootstrap_record, a.cpu_lock)
    except (OSError, ValueError, md.PackageNotFoundError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
