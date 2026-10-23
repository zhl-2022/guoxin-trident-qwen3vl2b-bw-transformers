#!/usr/bin/env python3
"""Plan/reuse native packages and install only reviewed CPU wheels in a NEW venv.

Never runs pip against the native interpreter. All install resolution is explicit,
hash locked, --no-deps, binary-only, and limited to the candidate private directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid
import venv

from download_candidate_models_20261023 import BASE, ROOT, TASKS, download_file, file_digest, json_new, read_json

PROTECTED = {"torch", "torchvision", "torchaudio", "vllm", "vllm-hcu", "triton", "flash-attn", "lightop", "aiter"}


def canonical(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def installed_distribution(name):
    try:
        dist = md.distribution(name)
        return {"name": canonical(name), "version": dist.version, "root": str(Path(dist.locate_file("")).resolve()),
                "requires_dist": list(dist.requires or [])}
    except md.PackageNotFoundError:
        return None


def native_snapshot(expected):
    snapshot = {}
    for name, expected_version in expected.items():
        item = installed_distribution(name)
        if item is None or item["version"] != expected_version:
            raise ValueError("Native distribution differs from captured image; re-review before apply: " + name)
        if sys.prefix != sys.base_prefix and Path(item["root"]).is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("Protected native package is shadowed in candidate venv: " + name)
        snapshot[name] = {"version": item["version"], "root": item["root"]}
    return snapshot


def resolve_packages(requirements, pool, get_installed=installed_distribution):
    """Metadata-only closure; prefer satisfying native versions over CPU fallback."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.version import Version
    marker_env = {**default_environment(), "os_name": "posix", "sys_platform": "linux", "platform_machine": "x86_64", "platform_system": "Linux",
                  "python_version": "3.10", "python_full_version": "3.10.0", "extra": ""}
    available = {}
    for item in pool:
        name = canonical(item["name"])
        if name in PROTECTED or name.startswith(("nvidia-", "cuda-", "torch-")):
            raise ValueError("GPU/native package found in CPU wheel pool")
        available.setdefault(name, []).append(item)
    queue = list(requirements)
    constraints = {}
    selected = {}
    processed = set()
    while queue:
        requirement = Requirement(queue.pop(0))
        if requirement.marker and not requirement.marker.evaluate(marker_env):
            continue
        if requirement.url or requirement.extras:
            raise ValueError("Unreviewed URL/extras in dependency closure: " + str(requirement))
        name = canonical(requirement.name)
        constraints.setdefault(name, []).append(requirement.specifier)
        def fits(version):
            return all(spec.contains(version, prereleases=True) for spec in constraints[name])
        if name in selected and fits(selected[name]["version"]):
            continue
        native = get_installed(name)
        if native and fits(native["version"]):
            chosen = {**native, "mode": "reuse_native"}
        else:
            if name in PROTECTED or name.startswith(("nvidia-", "cuda-", "torch-")):
                raise ValueError("Native GPU dependency missing or incompatible; never install fallback: " + name)
            candidates = [x for x in available.get(name, []) if fits(x["version"])]
            if not candidates:
                raise ValueError("No reviewed CPU wheel satisfies dependency; stop for a reviewed lock extension: " + str(requirement))
            chosen = {**max(candidates, key=lambda x: Version(x["version"])), "mode": "install_cpu_wheel"}
        selected[name] = chosen
        key = (name, chosen["version"])
        if key in processed or name in PROTECTED:
            continue
        processed.add(key)
        queue.extend(chosen.get("requires_dist", []))
    # Later constraints must also be satisfied by the final selection.
    for name, item in selected.items():
        if not all(spec.contains(item["version"], prereleases=True) for spec in constraints[name]):
            raise ValueError("Dependency constraints conflict: " + name)
    return sorted(selected.values(), key=lambda x: canonical(x["name"]))


def build_plan(task, profile="native-first", get_installed=installed_distribution):
    config = read_json(ROOT / "tasks" / TASKS[task] / "env-plan.json")
    pool_path = ROOT / "scripts/candidate_cpu_wheels_20261023.json"
    if file_digest(pool_path) != config["cpu_wheel_pool_sha256"]:
        raise ValueError("Approved CPU wheel metadata pool hash differs")
    pool = read_json(pool_path)["wheels"]
    requirements = config["requirements_publisher_pinned"] if profile == "publisher-pinned" else config["requirements_native_first"]
    selected = resolve_packages(requirements, pool, get_installed)
    return {"schema_version": 1, "task": task, "profile": profile, "requirements": requirements,
            "selected": selected, "install_count": sum(x["mode"] == "install_cpu_wheel" for x in selected),
            "protected_native_expectations": config["protected_native_expectations"],
            "gpu_engine_started": False, "weights_loaded": False,
            "framework_eligibility": config["framework_eligibility"]}


def validate_current(plan, before):
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    snapshot = native_snapshot(plan["protected_native_expectations"])
    if snapshot != before:
        raise ValueError("Protected native versions or locations changed")
    actual = {}
    for item in plan["selected"]:
        current = installed_distribution(item["name"])
        if current is None or current["version"] != item["version"]:
            raise ValueError("Candidate dependency version differs: " + item["name"])
        if item["mode"] == "install_cpu_wheel" and not Path(current["root"]).is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("CPU dependency was not installed inside candidate venv")
        actual[canonical(item["name"])] = current["version"]
    environment = {**default_environment(), "extra": ""}
    # Check the selected inference closure, not unrelated native UI/training extras.
    for item in plan["selected"]:
        if canonical(item["name"]) in PROTECTED:
            continue
        for raw in item.get("requires_dist", []):
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate(environment):
                continue
            name = canonical(req.name)
            if name not in actual or not req.specifier.contains(actual[name], prereleases=True):
                raise ValueError("Selected inference dependency closure incomplete: " + raw)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("GPU initialized before CPU-only import validation")
    import transformers
    from transformers import AutoConfig, AutoModel, AutoProcessor  # noqa: F401
    if plan["task"] == "forcedaligner":
        import librosa  # noqa: F401
        import nagisa  # noqa: F401
        import soundfile  # noqa: F401
        import soynlp  # noqa: F401
    elif plan["task"] == "trident":
        from transformers.models.qwen3_vl import Qwen3VLModel  # noqa: F401
        import qwen_vl_utils  # noqa: F401
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only import validation initialized GPU unexpectedly")
    return {"cpu_imports_passed": True, "transformers": transformers.__version__, "selected_versions": actual,
            "protected_native_unchanged": True, "gpu_initialized": False, "weights_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply", "check"))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--profile", choices=("native-first", "publisher-pinned"), default="native-first")
    parser.add_argument("--label", default="001")
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--environment-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "check":
        if sys.prefix == sys.base_prefix or not args.environment_file:
            parser.error("check is called only by a prepared isolated candidate venv")
        record = read_json(args.environment_file)
        result = validate_current(record["plan"], record["native_before"])
        print(json.dumps(result, ensure_ascii=False))
        return 0
    plan = build_plan(args.task, args.profile)
    if args.action == "plan":
        print(json.dumps({"mode": "offline_plan", "downloads_performed": False, "filesystem_mutated": False, **plan}, ensure_ascii=False, indent=2))
        return 0
    if (sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10)
            or sys.prefix != sys.base_prefix or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", args.label)
            or not 60 <= args.max_seconds <= 7200):
        parser.error("apply requires base Linux x86_64 CPython3.10, safe new label, max-seconds60..7200")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise ValueError("Reviewed wheels require glibc>=2.28")
    if any(path.is_symlink() for path in (BASE, *BASE.parents)):
        raise ValueError("Private-data path contains symlink; review mapping first")
    before = native_snapshot(plan["protected_native_expectations"])
    target = BASE / "envs" / f"{args.task}-{args.label}"
    if target.exists() or target.is_symlink() or target.parent.is_symlink():
        raise ValueError("Candidate venv target exists or is unsafe; choose a new label")
    BASE.mkdir(parents=True, exist_ok=True)
    lock_path = BASE / ".candidate-env-preparation.lock"
    token = str(uuid.uuid4())
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "token": token}, handle)
    record = {"run_id": token, "status": "preparing", "plan": plan, "native_before": before, "target": str(target)}
    deadline = time.monotonic() + args.max_seconds
    try:
        cache = BASE / "cpu-wheels"
        if cache.is_symlink():
            raise ValueError("CPU wheel cache is a symlink")
        cache.mkdir(exist_ok=True)
        selected_wheels = [item for item in plan["selected"] if item["mode"] == "install_cpu_wheel"]
        for item in selected_wheels:
            if canonical(item["name"]) in PROTECTED or not item["url"].startswith("https://files.pythonhosted.org/"):
                raise ValueError("Unreviewed install source/package")
            download_file(item, cache / item["filename"], item["url"], deadline)
        target.parent.mkdir(exist_ok=True)
        # The captured SCNet /usr/bin/python3.10 lacks ensurepip. Reuse native
        # pip as an import through system-site-packages, but execute it with the
        # NEW venv interpreter so every installed wheel stays inside that venv.
        venv.EnvBuilder(with_pip=False, system_site_packages=True, clear=False).create(target)
        record_path = target / "candidate-environment.json"
        json_new(record_path, record)
        effective_lock = target / "requirements-effective.lock"
        lines = [f"{item['name']} @ {(cache / item['filename']).as_uri()} --hash=sha256:{item['sha256']}" for item in selected_wheels]
        effective_lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
        interpreter = target / "bin/python"
        if selected_wheels:
            command = [str(interpreter), "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", "--ignore-installed",
                       "--require-hashes", "--only-binary=:all:", "-r", str(effective_lock)]
            with (target / "pip-install.log").open("xb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=max(1, deadline - time.monotonic()), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        native_after = native_snapshot(plan["protected_native_expectations"])
        if native_after != before:
            raise ValueError("Base native metadata changed during isolated preparation")
        command = [str(interpreter), "-B", str(Path(__file__).resolve()), "check", "--task", args.task, "--environment-file", str(record_path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=max(1, deadline - time.monotonic()),
                                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        (target / "cpu-import-check.stdout.log").write_text(result.stdout, encoding="utf-8")
        (target / "cpu-import-check.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Candidate CPU imports failed; inspect preserved logs")
        record["status"] = "prepared_cpu_imports_verified_gpu_not_run"
        record["cpu_validation_exit_code"] = result.returncode
        record["native_after"] = native_after
        print(json.dumps({"status": record["status"], "python": str(interpreter), "gpu_initialized": False,
                          "framework_eligibility": plan["framework_eligibility"]}, ensure_ascii=False))
        return 0
    finally:
        if record["status"] == "preparing":
            record["status"] = "failed"
        json_new(BASE / f"candidate-env-{token}.json", record)
        if lock_path.is_file() and read_json(lock_path).get("token") == token:
            lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
