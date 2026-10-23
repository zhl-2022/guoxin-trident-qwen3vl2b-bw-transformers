#!/usr/bin/env python3
"""Plan or explicitly prepare the reviewed CPU-only ASR venv; never start a service."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import venv
import wave

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/qwen3-asr-bw-vllm"
CONFIG_SHA256 = "56fd1f1ea939dff47533b052962d294375e85482ed310e48ccad402cf5699818"
LOCK_LINE = re.compile(r"([a-z0-9-]+) @ (https://files\.pythonhosted\.org/\S+\.whl) --hash=sha256:([0-9a-f]{64})")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(task):
    wheels = []
    for line in (task / "requirements-extra.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = LOCK_LINE.fullmatch(line)
        if not match:
            raise ValueError("Lock must contain only reviewed PyPI wheel URLs with SHA256")
        name, url, digest = match.groups()
        filename = url.rsplit("/", 1)[1]
        if canonical(filename.split("-", 1)[0]) != name:
            raise ValueError("Wheel name does not match requirement: " + name)
        wheels.append({"name": name, "version": filename.split("-")[1],
                       "url": url, "sha256": digest, "requirement": line})
    if len(wheels) != 7 or len({item["name"] for item in wheels}) != 7:
        raise ValueError("Expected the seven reviewed CPU audio wheels")
    return wheels


def installed(name):
    dist = md.distribution(name)
    return {"version": dist.version, "root": str(Path(dist.locate_file("")).resolve())}


def native_snapshot(task):
    expected = json.loads((task / "native-stack.json").read_text(encoding="utf-8"))
    actual = {}
    for name, version in expected.items():
        item = installed(name)
        if item["version"] != version:
            raise ValueError(f"Native {name} differs: expected {version}, got {item['version']}; re-review image")
        if sys.prefix != sys.base_prefix and Path(item["root"]).is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("Protected native distribution is shadowed inside venv: " + name)
        actual[name] = item
    return actual


def check_target(target):
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise ValueError("Require a new absolute venv directory; existing paths are refused")


def target_platform():
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10) or platform.machine() != "x86_64":
        raise ValueError("Apply/check requires the captured Linux x86_64 Python 3.10 image")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise ValueError("Reviewed soundfile wheel requires glibc >=2.28")


def plan(task, target):
    wheels = read_lock(task)
    return {"mode": "plan_only", "venv": str(target), "task": str(task),
            "target_platform": "Linux x86_64 / Python 3.10 / glibc>=2.28",
            "system_site_packages": True, "wheels": wheels,
            "protected_native": json.loads((task / "native-stack.json").read_text(encoding="utf-8")),
            "install_flags": ["--no-deps", "--require-hashes", "--only-binary=:all:"],
            "compatibility_plugin": "qwen_asr_tf55_compat",
            "downloads_performed": False, "model_loaded": False, "gpu_checked": False,
            "service_started": False, "filesystem_mutated": False,
            "next_steps": ["bash prepare.sh --apply", "bash run.sh", "bash stop.sh"]}


def validate_current(task, before=None):
    """CPU imports, dependency closure and codec test; no Torch/model/GPU imports."""
    target_platform()
    if sys.prefix == sys.base_prefix:
        raise ValueError("CPU validation must run in an isolated system-site-packages venv")
    snapshot = native_snapshot(task)
    if before is not None and snapshot != before:
        raise ValueError("Protected native distribution version or path changed")
    for item in read_lock(task):
        if installed(item["name"])["version"] != item["version"]:
            raise ValueError("Audio package version differs: " + item["name"])
    # Locate the captured config without importing vLLM or initializing an accelerator.
    config = Path(md.distribution("vllm").locate_file("vllm/transformers_utils/configs/qwen3_asr.py"))
    if hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Native ASR config source differs; re-review compatibility plugin")
    from packaging.requirements import Requirement
    visited = set()

    def closure(name):
        name = canonical(name)
        if name in visited:
            return
        visited.add(name)
        dist = md.distribution(name)
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            child = md.distribution(req.name)
            if req.specifier and not req.specifier.contains(child.version, prereleases=True):
                raise ValueError(f"Audio dependency mismatch: {raw}; got {child.version}")
            closure(req.name)

    closure("librosa")
    closure("soundfile")
    import librosa
    import numpy as np
    import soundfile  # noqa: F401 -- decoder import is part of the CPU check
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    waveform, sample_rate = librosa.load(io.BytesIO(buf.getvalue()), sr=16000)
    resampled = librosa.resample(waveform, orig_sr=16000, target_sr=8000)
    if sample_rate != 16000 or waveform.shape != (1600,) or waveform.dtype != np.float32 or resampled.shape != (800,) or not np.isfinite(resampled).all():
        raise ValueError("CPU WAV decode/resample check failed")
    return {"status": "passed", "python": sys.executable, "protected": snapshot,
            "audio_closure_packages": sorted(visited), "wav_samples": 1600,
            "resampled_samples": 800, "native_config_sha256": CONFIG_SHA256,
            "model_loaded": False, "gpu_checked": False, "service_started": False}


def clean_environment():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PIP_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def apply(task, target):
    target_platform()
    if sys.prefix != sys.base_prefix:
        raise ValueError("Run --apply with the native base interpreter, outside a venv")
    check_target(target)
    snapshot = native_snapshot(task)
    missing = []
    for item in read_lock(task):
        try:
            current = installed(item["name"])
        except md.PackageNotFoundError:
            missing.append(item)
            continue
        if current["version"] != item["version"]:
            raise ValueError("Refusing to shadow existing audio distribution: " + item["name"])
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(target)
    evidence = target / "asr-prepare-evidence"
    evidence.mkdir()
    (evidence / "native-before.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    selected = evidence / "missing-wheels.lock.txt"
    selected.write_text("".join(item["requirement"] + "\n" for item in missing), encoding="utf-8")
    python = target / "bin/python"
    env = clean_environment()
    env["NUMBA_CACHE_DIR"] = str(target / "numba-cache")
    if missing:
        command = [str(python), "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
                   "install", "--no-deps", "--require-hashes", "--only-binary=:all:", "--no-cache-dir", "-r", str(selected)]
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        (evidence / "pip-install.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise ValueError(f"Locked installation failed; inspect retained {evidence}/pip-install.log")
    result = subprocess.run([str(python), "-B", str(Path(__file__).resolve()), "--task-dir", str(task),
                             "--validate-current", "--before", str(evidence / "native-before.json")],
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    (evidence / "validation.stdout").write_text(result.stdout, encoding="utf-8")
    (evidence / "validation.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"CPU validation failed; inspect retained {evidence}")
    report = json.loads(result.stdout)
    (evidence / "ready.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=TASK)
    parser.add_argument("--venv", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--check-existing", type=Path, metavar="VENV")
    action.add_argument("--validate-current", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--dry-run", action="store_true", help="default; local plan only")
    parser.add_argument("--before", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        task = args.task_dir.resolve()
        target = args.venv or task / "runtime/audio-venv"
        if args.validate_current:
            before = json.loads(args.before.read_text(encoding="utf-8")) if args.before else None
            report = validate_current(task, before)
        elif args.check_existing:
            target_platform()
            existing = args.check_existing
            if not existing.is_absolute() or not (existing / "pyvenv.cfg").is_file():
                raise ValueError("Require an absolute existing venv containing pyvenv.cfg")
            result = subprocess.run([str(existing / "bin/python"), "-B", str(Path(__file__).resolve()),
                                     "--task-dir", str(task), "--validate-current"],
                                    env=clean_environment(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            if result.returncode:
                raise ValueError("Existing venv CPU validation failed: " + result.stderr[-2000:])
            report = json.loads(result.stdout)
        elif args.apply:
            report = apply(task, target)
        else:
            report = plan(task, target)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, md.PackageNotFoundError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc), "service_started": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
