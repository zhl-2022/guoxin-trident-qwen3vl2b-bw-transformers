#!/usr/bin/env python3
"""Repair a recorded candidate venv using only approved cached CPU wheels.

Copyright (c) 2026 the task author. No repository-wide open-source license chosen.
No downloads, new venv, model loads, native GPU installs, or old-record rewrites.
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

BASE = Path("/root/private_data/guoxin-candidates")
APPROVED_POOL_SHA256 = "8daa740106938d69d0f1256d43a009afa190dd3f3b337cac996fb594c767af71"
PROTECTED = {"torch", "torchvision", "torchaudio", "vllm", "vllm-hcu", "triton", "flash-attn", "lightop", "aiter"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def distribution(name):
    try:
        d = md.distribution(name)
        text = d.read_text("METADATA")
        return {"name": canonical(name), "version": d.version,
                "root": str(Path(d.locate_file("")).resolve()), "requires_dist": list(d.requires or []),
                "metadata_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None}
    except md.PackageNotFoundError:
        return None


def protected_snapshot(names):
    return {canonical(name): distribution(name) for name in sorted(set(names) | PROTECTED)}


def require_safe_paths(target, record_path):
    # Native venv bin/python is normally a symlink to the base executable.
    # Reject directory/metadata/evidence symlinks; the child separately proves
    # sys.prefix is exactly this venv before reading candidate metadata/imports.
    directory = Path(target)
    for item in (BASE, BASE / "envs", directory, directory / "bin",
                 directory / "pyvenv.cfg", Path(record_path), directory / "partial-repair-evidence"):
        if item.is_symlink():
            raise ValueError("Candidate directory/metadata/evidence symlink refused: " + str(item))
    current = directory
    while True:
        if current.is_symlink():
            raise ValueError("Candidate ancestor symlink refused")
        if current.parent == current:
            break
        current = current.parent


def require_native(record, actual):
    expected = record["plan"]["protected_native_expectations"]
    before = record["native_before"]
    if set(expected) != set(before) or not expected or not set(expected).issubset(PROTECTED):
        raise ValueError("Original native_before/expectations set differs")
    for name, version in expected.items():
        item, original = actual.get(canonical(name)), before[name]
        if (item is None or item["version"] != version or item["version"] != original["version"]
                or Path(item["root"]).resolve() != Path(original["root"]).resolve()):
            raise ValueError("Protected native version/root differs: " + name)
        if original.get("metadata_sha256") and item.get("metadata_sha256") != original["metadata_sha256"]:
            raise ValueError("Protected original native METADATA SHA differs: " + name)


def load_identity(record_path, record_sha, pool_path, pool_sha):
    record_path, pool_path = Path(record_path), Path(pool_path)
    if not re.fullmatch(r"[0-9a-f]{64}", record_sha) or sha(record_path) != record_sha:
        raise ValueError("Original candidate-environment.json SHA differs")
    if pool_sha != APPROVED_POOL_SHA256 or sha(pool_path) != APPROVED_POOL_SHA256:
        raise ValueError("Approved wheel-pool SHA differs")
    if record_path.is_symlink() or pool_path.is_symlink():
        raise ValueError("Record/pool symlink refused")
    record = read(record_path)
    target = Path(record["target"])
    require_safe_paths(target, record_path)
    if (record_path.name != "candidate-environment.json" or target.is_symlink()
            or target.resolve() != record_path.resolve().parent
            or not target.resolve().is_relative_to((BASE / "envs").resolve())
            or not target.is_dir() or not (target / "pyvenv.cfg").is_file()):
        raise ValueError("Record does not identify an existing candidate venv under approved base")
    if "include-system-site-packages = true" not in (target / "pyvenv.cfg").read_text().lower():
        raise ValueError("Recorded venv does not reuse native site packages")
    if record["plan"].get("task") not in {"forcedaligner", "trident", "mimo9b"}:
        raise ValueError("Unknown original candidate task")
    pool = read(pool_path)["wheels"]
    indexed = {}
    for item in pool:
        name = canonical(item["name"])
        if name in PROTECTED or name.startswith(("nvidia-", "cuda-", "torch-")):
            raise ValueError("Protected wheel found in CPU approval pool")
        indexed[(name, item["version"])] = item
    selected = record["plan"]["selected"]
    if not selected or len({canonical(x["name"]) for x in selected}) != len(selected):
        raise ValueError("Invalid original selected closure")
    for item in selected:
        name = canonical(item["name"])
        if item.get("mode") not in {"reuse_native", "install_cpu_wheel"}:
            raise ValueError("Unknown original selected mode")
        if item["mode"] == "install_cpu_wheel":
            approved = indexed.get((name, item["version"]))
            fields = ("filename", "sha256", "size", "url", "requires_dist")
            if not approved or any(item.get(k) != approved.get(k) for k in fields):
                raise ValueError("Original selected CPU item differs from approved pool: " + name)
            if name in PROTECTED:
                raise ValueError("Original plan attempts GPU installation")
    return record, target, indexed


def choose_repairs(plan, actual, target, approved):
    pending, reused = [], []
    for item in plan["selected"]:
        name = canonical(item["name"])
        current = actual.get(name)
        if item["mode"] == "reuse_native":
            if (not current or current["version"] != item["version"]
                    or Path(current["root"]).resolve() != Path(item["root"]).resolve()):
                raise ValueError("Recorded native CPU reuse changed; reviewed plan extension required: " + name)
            reused.append({"name": name, "version": current["version"], "root": current["root"], "mode": "reuse_recorded_native"})
            continue
        correct = (current and current["version"] == item["version"]
                   and Path(current["root"]).resolve().is_relative_to(Path(target).resolve()))
        if correct:
            reused.append({"name": name, "version": current["version"], "root": current["root"], "mode": "reuse_satisfied_candidate"})
            continue
        if current and Path(current["root"]).resolve().is_relative_to(Path(target).resolve()):
            raise ValueError("Different version already shadows inside venv; do not guess uninstall ownership: " + name)
        if name in PROTECTED or name.startswith(("nvidia-", "cuda-", "torch-")):
            raise ValueError("Never repair protected native packages")
        pending.append(approved[(name, item["version"])])
    return pending, reused


def validate_metadata(record, current, target):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    versions = {}
    for item in record["plan"]["selected"]:
        name = canonical(item["name"])
        actual = current.get(name)
        if not actual or actual["version"] != item["version"]:
            raise ValueError("Candidate dependency version differs: " + name)
        root = Path(actual["root"]).resolve()
        if item["mode"] == "install_cpu_wheel" and not root.is_relative_to(Path(target).resolve()):
            raise ValueError("Expected CPU shadow is outside candidate venv: " + name)
        if item["mode"] == "reuse_native" and root != Path(item["root"]).resolve():
            raise ValueError("Native reuse root changed: " + name)
        versions[name] = actual["version"]
    env = {**default_environment(), "extra": ""}
    for item in record["plan"]["selected"]:
        if canonical(item["name"]) in PROTECTED:
            continue
        for raw in item.get("requires_dist", []):
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate(env):
                continue
            name = canonical(req.name)
            if req.url or req.extras or name not in versions or not req.specifier.contains(versions[name], prereleases=True):
                raise ValueError("Selected inference closure incomplete: " + raw)
    for raw in record["plan"]["requirements"]:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate(env):
            continue
        if canonical(req.name) not in versions or not req.specifier.contains(versions[canonical(req.name)], prereleases=True):
            raise ValueError("Original root requirement differs: " + raw)
    return versions


def check_imports(task, allow_discovery):
    import torch
    before = torch.cuda.is_initialized()
    if before and not allow_discovery:
        raise ValueError("GPU initialized before strict CPU-only check")
    import transformers
    from transformers import AutoConfig, AutoModel, AutoProcessor  # noqa: F401
    if task == "forcedaligner":
        import librosa, nagisa, soundfile, soynlp  # noqa: F401
    elif task == "trident":
        from transformers.models.qwen3_vl import Qwen3VLModel  # noqa: F401
        import qwen_vl_utils  # noqa: F401
    after = torch.cuda.is_initialized()
    if after and not allow_discovery:
        raise ValueError("Strict CPU-only imports initialized GPU")
    return {"dependency_imports_passed": True, "transformers": transformers.__version__,
            "device_discovery_allowed": allow_discovery,
            "gpu_initialized_after_torch_import": bool(before), "gpu_initialized_after_imports": bool(after),
            "no_gpu_initialization_observed": not before and not after,
            "cpu_only_hardware_access_proven": False,
            "gpu_model_inference_verified": False, "gpu_engine_started": False, "weights_loaded": False}


def child_environment():
    return {**{k:v for k,v in os.environ.items() if not k.startswith("PIP_") and k not in {"PYTHONHOME", "PYTHONPATH"}},
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def run_capture(command, stdout, stderr, timeout):
    with Path(stdout).open("xb") as out, Path(stderr).open("xb") as err:
        return subprocess.run(command, stdout=out, stderr=err, timeout=timeout, env=child_environment(), check=False).returncode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", required=True)
    p.add_argument("--record-sha256", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--pool-sha256", default=APPROVED_POOL_SHA256)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--allow-device-discovery", action="store_true")
    p.add_argument("--pip-timeout", type=int, default=600)
    p.add_argument("--check-timeout", type=int, default=240)
    p.add_argument("--_child", choices=("metadata", "imports"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    record, target, approved = load_identity(args.record, args.record_sha256, args.pool, args.pool_sha256)
    names = [canonical(x["name"]) for x in record["plan"]["selected"]]
    if args._child:
        if Path(sys.prefix).resolve() != target.resolve() or sys.prefix == sys.base_prefix:
            raise ValueError("Child is not inside the recorded existing candidate venv")
        current = {name: distribution(name) for name in names}
        protected = protected_snapshot(record["plan"]["protected_native_expectations"])
        require_native(record, protected)
        if args._child == "metadata":
            print(json.dumps({"selected": current, "protected": protected}, ensure_ascii=False))
        else:
            versions = validate_metadata(record, current, target)
            result = check_imports(record["plan"]["task"], args.allow_device_discovery)
            print(json.dumps({**result, "selected_versions": versions, "protected": protected}, ensure_ascii=False))
        return 0
    if (sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3,10)
            or sys.prefix != sys.base_prefix or not 300 <= args.pip_timeout <= 600 or not 60 <= args.check_timeout <= 600):
        p.error("Use base Linux x86_64 Python3.10; pip-timeout300..600, check-timeout60..600")
    before = protected_snapshot(record["plan"]["protected_native_expectations"])
    require_native(record, before)
    require_safe_paths(target, args.record)
    attempt = target / "partial-repair-evidence" / ("attempt-"+uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    original = Path(args.record).read_bytes()
    (attempt / "original-candidate-environment.json").write_bytes(original)
    (attempt / "repair-script.py").write_bytes(Path(__file__).read_bytes())
    (attempt / "approved-cpu-wheel-pool.json").write_bytes(Path(args.pool).read_bytes())
    write_new(attempt / "native-before.json", before)
    record_sha, started, state = args.record_sha256, time.monotonic(), "failed"
    result = {"status": state, "original_record_sha256": record_sha, "approved_pool_sha256": args.pool_sha256,
              "task": record["plan"]["task"], "target": str(target), "apply": args.apply,
              "device_discovery_allowed": args.allow_device_discovery, "downloads_performed": False,
              "repair_script_sha256": sha(Path(__file__).resolve()),
              "native_package_installation_performed": False, "official_accuracy_verified": False,
              "official_performance_verified": False, "submission_ready": False}
    lock = BASE / ".candidate-partial-repair.lock"
    token = uuid.uuid4().hex
    owned_lock = False
    try:
        if (BASE / ".candidate-env-preparation.lock").exists():
            raise ValueError("Original candidate preparation is still locked; no concurrent repair")
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor,"w") as f:
            json.dump({"token":token,"pid":os.getpid()}, f)
        owned_lock = True
        common = ["--record", str(Path(args.record).resolve()), "--record-sha256", record_sha,
                  "--pool", str(Path(args.pool).resolve()), "--pool-sha256", args.pool_sha256]
        interpreter = target / "bin/python"
        meta_command = [str(interpreter), "-B", str(Path(__file__).resolve()), *common, "--_child", "metadata"]
        write_new(attempt / "metadata-command.json", {"argv":meta_command})
        code = run_capture(meta_command, attempt/"metadata.stdout.json", attempt/"metadata.stderr.log", 120)
        if code:
            raise ValueError("Recorded venv metadata child failed; inspect preserved stderr")
        metadata = read(attempt/"metadata.stdout.json")
        if metadata["protected"] != before:
            raise ValueError("Venv shadows a protected native package or metadata differs")
        pending, reused = choose_repairs(record["plan"], metadata["selected"], target, approved)
        write_new(attempt / "repair-plan.json", {"install":pending,"reuse_without_install":reused})
        result["install_names"] = [canonical(x["name"]) for x in pending]
        result["reused_count"] = len(reused)
        cache = BASE / "cpu-wheels"
        lines = []
        for item in pending:
            filename = item["filename"]
            if Path(filename).name != filename or cache.is_symlink():
                raise ValueError("Unsafe wheel filename/cache")
            wheel = cache / filename
            if wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size != item["size"] or sha(wheel) != item["sha256"]:
                raise ValueError("Approved cached wheel missing or SHA/size differs; no download: " + filename)
            lines.append(f"{item['name']} @ {wheel.as_uri()} --hash=sha256:{item['sha256']}")
        effective = attempt / "repair-requirements.lock"
        effective.write_text("\n".join(lines)+"\n", encoding="utf-8")
        if args.apply and pending:
            command = [str(interpreter), "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", "--ignore-installed",
                       "--require-hashes", "--only-binary=:all:", "-r", str(effective)]
            write_new(attempt / "pip-command.json", {"argv":command,"timeout_seconds":args.pip_timeout})
            code = run_capture(command, attempt/"pip.stdout.log", attempt/"pip.stderr.log", args.pip_timeout)
            result["pip_exit_code"] = code
            if code:
                raise RuntimeError("Cached CPU repair installation failed")
        if args.apply:
            command = [str(interpreter), "-B", str(Path(__file__).resolve()), *common, "--_child", "imports"]
            if args.allow_device_discovery:
                command.append("--allow-device-discovery")
            write_new(attempt / "import-command.json", {"argv":command,"timeout_seconds":args.check_timeout})
            code = run_capture(command, attempt/"imports.stdout.json", attempt/"imports.stderr.log", args.check_timeout)
            result["import_check_exit_code"] = code
            if code:
                raise RuntimeError("Candidate dependency/import validation failed; original failure preserved")
            result["validation"] = read(attempt/"imports.stdout.json")
            state = "dependencies_imports_verified_device_discovery_observed" if result["validation"]["gpu_initialized_after_imports"] else "dependencies_imports_verified_no_gpu_init_observed"
        else:
            state = "repair_plan_only_no_install_no_import_validation"
    except Exception as exc:
        result["error"] = {"type":type(exc).__name__,"message":str(exc)}
        state = "failed"
    finally:
        after = protected_snapshot(record["plan"]["protected_native_expectations"])
        write_new(attempt / "native-after.json", after)
        result["protected_native_metadata_unchanged"] = after == before
        result["original_record_unchanged"] = sha(args.record) == record_sha
        if not result["protected_native_metadata_unchanged"] or not result["original_record_unchanged"]:
            state = "failed"
        result.update(status=state, elapsed_seconds=time.monotonic()-started, evidence_directory=str(attempt))
        write_new(attempt / "result.json", result)
        if owned_lock and lock.is_file() and read(lock).get("token")==token:
            lock.unlink()
    print(json.dumps(result, ensure_ascii=False))
    return 1 if state == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
