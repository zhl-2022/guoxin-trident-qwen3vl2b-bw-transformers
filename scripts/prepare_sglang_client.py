#!/usr/bin/env python3
"""Plan or build a NEW, isolated SGLang client venv; never resolve GPU dependencies.

Target: the recorded SCNet Python3.10/Linux x86_64/glibc>=2.34 image.
Missing CPU dependencies require an explicit reviewed wheel lock, not pip's
automatic dependency resolution. --help compatibility must be observed before
running any performance workload; a failed partial venv is preserved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
SGLANG = {"name": "sglang", "version": "0.5.12",
          "url": "https://files.pythonhosted.org/packages/84/cb/a7802dcbdc35ec90d15115d312d55bc2b7aee1aea04b48d4f43df1f67267/sglang-0.5.12-cp310-cp310-manylinux_2_34_x86_64.whl",
          "sha256": "4b88edfd3870d7dc8dd880f97d104f637baf105538827f968483d6cac56258f8"}
CPU_ALLOWLIST = {"aiohttp", "requests", "tqdm", "pybase64", "orjson", "psutil", "pyzmq", "datasets",
                 "huggingface-hub", "sentencepiece", "tiktoken", "packaging", "anthropic", "openai",
                 "partial-json-parser", "interegular", "outlines", "jsonschema", "jsonschema-specifications",
                 "referencing", "rpds-py", "blobfile", "modelscope", "ipython", "jiter", "distro",
                 "setproctitle", "msgspec", "diskcache", "regex", "lark", "cloudpickle", "pyarrow",
                 "xxhash", "multiprocess", "dill", "fsspec", "filelock", "pyyaml", "pydantic-settings",
                 "python-dotenv", "nest-asyncio", "rich", "httpx", "httpcore", "anyio", "sniffio",
                 "certifi", "idna", "charset-normalizer", "urllib3"}


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def snapshot(expected):
    result = {}
    for name, version in expected.items():
        dist = md.distribution(name)
        if dist.version != version:
            raise ValueError("protected native version differs: " + name)
        metadata = next((f for f in dist.files or () if str(f).endswith(".dist-info/METADATA")), None)
        result[name] = {"version": dist.version, "root": str(dist.locate_file("")),
                        "metadata_sha256": hashlib.sha256(Path(dist.locate_file(metadata)).read_bytes()).hexdigest() if metadata else None}
    return result


def locks(path, protected):
    extras = json.loads(path.read_text()) if path else []
    if not isinstance(extras, list):
        raise ValueError("extra lock must be an explicit JSON array")
    selected, seen = [SGLANG.copy()], {"sglang"}
    for item in extras:
        if not isinstance(item, dict) or set(item) != {"name", "version", "url", "sha256"}:
            raise ValueError("each CPU wheel needs exact name/version/url/sha256")
        name = item["name"].lower().replace("_", "-")
        if name not in CPU_ALLOWLIST or name in protected or name in seen:
            raise ValueError("unreviewed, duplicate or protected dependency: " + name)
        if not isinstance(item["version"], str) or not item["version"] or any(c.isspace() for c in item["version"]):
            raise ValueError("exact dependency version required")
        url, digest = item["url"], item["sha256"]
        if not isinstance(url, str) or not url.startswith("https://files.pythonhosted.org/packages/") or not url.endswith(".whl") or any(c.isspace() for c in url):
            raise ValueError("official PyPI wheel URL required")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("exact SHA256 required")
        selected.append({**item, "name": name})
        seen.add(name)
    return selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--venv", required=True, type=Path)
    p.add_argument("--native-lock", type=Path, default=ROOT / "tasks/qwen3-asr-bw-vllm/native-stack.json")
    p.add_argument("--extra-cpu-lock", type=Path)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args(argv)
    try:
        protected = json.loads(a.native_lock.read_text())
        if not isinstance(protected, dict) or not protected:
            raise ValueError("native lock must be a nonempty object")
        wheels = locks(a.extra_cpu_lock, protected)
        if not a.venv.is_absolute() or a.venv.exists() or a.venv.is_symlink():
            raise ValueError("venv must be a NEW absolute path; partial/old environments are never overwritten")
        record = {"created_utc": stamp(), "mode": "apply" if a.apply else "plan_only", "venv": str(a.venv),
                  "system_site_packages": True, "wheel_lock": wheels, "protected_native": protected,
                  "install_flags": ["--ignore-installed", "--no-deps", "--require-hashes", "--only-binary=:all:"],
                  "client_command": [str(a.venv / "bin/python"), "-B", "-m", "sglang.bench_serving"],
                  "compatibility_proven": False, "service_started": False, "filesystem_mutated": False,
                  "scope": "isolated client preparation; package declarations are not a claim of native-stack compatibility"}
        if not a.apply:
            print(json.dumps(record, indent=2))
            return 0
        if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
            raise ValueError("reviewed wheel requires Linux x86_64/Python3.10")
        libc, version = platform.libc_ver()
        if libc != "glibc" or tuple(int(x) for x in version.split(".")[:2]) < (2, 34):
            raise ValueError("reviewed wheel requires glibc>=2.34")
        if sys.prefix != sys.base_prefix:
            raise ValueError("run bootstrap using the base native Python, not another venv")
        record["native_before"] = snapshot(protected)
        # Record before EnvBuilder: Ubuntu images can omit ensurepip and a
        # creation failure must still leave useful, immutable failure evidence.
        a.venv.mkdir(parents=True, exist_ok=False)
        folder = a.venv / "client-bootstrap-evidence"
        folder.mkdir()
        python = str(a.venv / "bin/python")
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
        record.update(filesystem_mutated=True, steps=[], status="preparing",
                      pip_bootstrap="reuse native pip via isolated venv interpreter; no ensurepip or apt")
        target = folder / "record.json"
        target.write_text(json.dumps(record, indent=2) + "\n")
        try:
            # As in prepare_audio_env.sh, system-site-packages exposes the
            # image's pip, and invoking it through this interpreter targets
            # the venv's sys.prefix without bootstrapping/upgrading native pip.
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(a.venv)
            record["steps"].append({"label": "venv", "with_pip": False, "completed_utc": stamp()})
            lock = folder / "reviewed-wheels.txt"
            lock.write_text("".join(f"{w['name']} @ {w['url']} --hash=sha256:{w['sha256']}\n" for w in wheels))
            for label, command, timeout in (
                ("pip", [python, "-m", "pip", "install", "--ignore-installed", "--no-deps", "--require-hashes", "--only-binary=:all:", "-r", str(lock)], 300),
                ("help", [python, "-B", "-m", "sglang.bench_serving", "--help"], 60)):
                raw = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
                (folder / (label + ".log")).write_bytes(raw.stdout)
                record["steps"].append({"label": label, "command": command, "exit_code": raw.returncode,
                                        "log_sha256": hashlib.sha256(raw.stdout).hexdigest()})
                if raw.returncode:
                    raise ValueError(label + " failed; partial venv and original log preserved; review missing CPU dependency before using a new target")
            # Compare metadata using the actual client interpreter without importing GPU libraries.
            check = ("import sys,json;sys.path.insert(0," + repr(str(Path(__file__).resolve().parent)) + ");"
                     "from prepare_sglang_client import snapshot;print(json.dumps(snapshot(" + repr(protected) + ")))")
            native = subprocess.run([python, "-B", "-c", check], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30)
            (folder / "native-client-check.log").write_bytes(native.stdout + native.stderr)
            if native.returncode:
                raise ValueError("client native metadata check failed")
            record["native_client"] = json.loads(native.stdout)
            if record["native_client"] != record["native_before"]:
                raise ValueError("client shadows protected native packages")
            record["compatibility_proven"] = True
            record["status"] = "ready"
            return 0
        except BaseException as exc:
            record.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
            raise
        finally:
            try:
                record["native_after"] = snapshot(protected)
                record["native_unchanged"] = record["native_before"] == record["native_after"]
            except Exception as exc:
                record.update(native_unchanged=False, native_after_error=str(exc))
            record["ended_utc"] = stamp()
            if not record["native_unchanged"]:
                record["compatibility_proven"] = False
            target.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"record": str(target), "compatible": record["compatibility_proven"]}))
            if not record["native_unchanged"]:
                raise ValueError("base native packages changed")
    except (OSError, ValueError, subprocess.SubprocessError, md.PackageNotFoundError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
