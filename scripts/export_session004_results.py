#!/usr/bin/env python3
"""Export session004 evidence only; never traverse a venv or export model files.

The archive is a live, bounded capture, not an assertion that evaluation passed.
All selected reads and directory walks fail closed. The destination must be new.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid
import zipfile
import zlib

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_FILES = 10000
TASKS = (
    "recon2reason-bw-vllm", "robobrain-bw-vllm", "qwen3-asr-bw-vllm",
    "mimo-v2.6-distill-qwen9b-bw-vllm", "trident-qwen3vl2b-bw-transformers",
    "qwen3-forcedaligner-bw-transformers",
)
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".csv", ".yaml", ".yml", ".lock", ".stdout", ".stderr"}
SCRIPT_SUFFIXES = {".py", ".sh"}
BLOCKED_DIRS = {".git", "__pycache__", "site-packages", "dist-packages", "lib", "lib64", "bin", "include", "share", "models", "weights", "cpu-wheels", "cache", ".cache", "venv", ".venv", "audio-venv"}
BLOCKED_NAMES = {".env", ".netrc", ".pypirc", "id_rsa", "id_ed25519", "token", "token.txt", "tokens.json", "token.json"}
SENSITIVE_WORDS = ("cookie", "credential", "password", "secret", "access_token", "refresh_token", "auth_token")
TASK_CONFIGS = {"task.json", "runtime.json", "model-lock.json", "env-plan.json", "upstream-lock.json", "vllm-probe-config.json", "benchmark-plan.json", "requests.example.json", "native-stack.json", "requirements-extra.txt", "env.log", "smoke.log"}
ENV_EVIDENCE = {"candidate-environment.json", "requirements-effective.lock", "pip-install.log", "cpu-import-check.stdout.log", "cpu-import-check.stderr.log"}
RUNTIME_PATTERNS = ("state*.json", "config-*.json", "server*.log", "supervisor*.log", "cleanup*.json", "env.log", "smoke*.log", "smoke*.json", "start*.log")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def plain_stat(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Symlink/reparse point is not an export source: " + str(path))
    return info


def optional_stat(path):
    try:
        return plain_stat(path)
    except FileNotFoundError:
        return None


def safe_name(path):
    name = path.name.lower()
    return name not in BLOCKED_NAMES and not any(word in name for word in SENSITIVE_WORDS)


def selected_kind(path):
    if not safe_name(path):
        return None
    if path.suffix.lower() in SCRIPT_SUFFIXES:
        return "script_hash_only"
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"validation.stdout", "validation.stderr"}:
        return "payload"
    return None


def strict_walk(root):
    """Only walk an explicitly allowed evidence subtree; propagate every error."""
    root_info = optional_stat(root)
    if root_info is None:
        return
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("Expected evidence directory: " + str(root))
    def fail(error):
        raise error
    for current, directories, files in os.walk(root, topdown=True, onerror=fail, followlinks=False):
        directory = Path(current)
        plain_stat(directory)
        retained = []
        for name in sorted(directories):
            child = directory / name
            if name.lower() in BLOCKED_DIRS or not safe_name(child):
                continue
            child_info = plain_stat(child)
            if not stat.S_ISDIR(child_info.st_mode):
                raise ValueError("Expected plain directory: " + str(child))
            retained.append(name)
        directories[:] = retained
        for name in sorted(files):
            path = directory / name
            if selected_kind(path):
                info = plain_stat(path)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Expected regular evidence file: " + str(path))
                yield path


def read_stable(path, max_file_bytes):
    initial = plain_stat(path)
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("Expected regular file: " + str(path))
    if initial.st_size > max_file_bytes:
        raise ValueError("File exceeds export size limit: " + str(path))
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        data = stream.read(max_file_bytes + 1)
        after = os.fstat(stream.fileno())
    key = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if len(data) > max_file_bytes or key(initial) != key(before) or key(before) != key(after) or len(data) != after.st_size:
        raise ValueError("File changed during export read; retry after writers stop: " + str(path))
    return data, {"source": str(path.absolute()), "size": len(data), "sha256": sha256(data),
                  "crc32": f"{zlib.crc32(data) & 0xffffffff:08x}", "source_mtime_ns": after.st_mtime_ns, "captured_utc": utc_now()}


def top_files(directory):
    info = optional_stat(directory)
    if info is None:
        return []
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("Expected plain directory: " + str(directory))
    with os.scandir(directory) as entries:
        return sorted((Path(entry.path) for entry in entries), key=lambda path: path.name)


def gather_sources(project, candidates, sglang, extra_files, script_files):
    """Return archive-path => (source, kind), plus absent optional roots."""
    if not stat.S_ISDIR(plain_stat(project).st_mode):
        raise ValueError("Project must be an existing directory")
    selected = {}
    missing = []
    def add(path, archive_path, forced_kind=None):
        if optional_stat(path) is None:
            return
        kind = forced_kind or selected_kind(path)
        if kind is None:
            raise ValueError("File is outside the evidence allowlist: " + str(path))
        if archive_path in selected and selected[archive_path] != (path, kind):
            raise ValueError("Archive name collision: " + archive_path)
        selected[archive_path] = (path, kind)
    def tree(path, prefix, scripts_only=False):
        if optional_stat(path) is None:
            missing.append(str(path))
            return
        for source in strict_walk(path):
            if scripts_only and source.suffix.lower() not in SCRIPT_SUFFIXES:
                continue
            add(source, prefix + "/" + source.relative_to(path).as_posix())
    tree(project / "evidence/session004", "project/evidence/session004")
    tree(project / "results/session004", "project/results/session004")
    tree(project / "scripts", "project/scripts-hashes", scripts_only=True)
    for task in TASKS:
        task_root = project / "tasks" / task
        for name in sorted(TASK_CONFIGS):
            add(task_root / name, f"project/tasks/{task}/{name}")
        for source in top_files(task_root):
            if source.suffix in SCRIPT_SUFFIXES:
                add(source, f"project/tasks/{task}/{source.name}")
        tree(task_root / "results", f"project/tasks/{task}/results")
        for source in top_files(task_root / "runtime"):
            if any(fnmatch.fnmatchcase(source.name, pattern) for pattern in RUNTIME_PATTERNS):
                add(source, f"project/tasks/{task}/runtime/{source.name}")
        if task == "qwen3-asr-bw-vllm":
            tree(task_root / "runtime/audio-venv/asr-prepare-evidence", f"project/tasks/{task}/runtime/audio-venv/asr-prepare-evidence")
    # Input manifests only; fixture audio/images and model payloads stay remote.
    for source in strict_walk(project / "fixtures"):
        if source.name.endswith(("manifest.json", "manifest.jsonl")) or fnmatch.fnmatchcase(source.name, "requests*.json"):
            add(source, "project/fixtures/" + source.relative_to(project / "fixtures").as_posix())
    tree(candidates / "results", "candidates/results")
    for source in top_files(candidates):
        if fnmatch.fnmatchcase(source.name, "candidate-env-*.json"):
            add(source, "candidates/" + source.name)
    for env_path in top_files(candidates / "envs"):
        info = plain_stat(env_path)
        if not stat.S_ISDIR(info.st_mode):
            continue
        for name in sorted(ENV_EVIDENCE):
            add(env_path / name, f"candidates/envs/{env_path.name}/{name}")
        tree(env_path / "bootstrap-evidence", f"candidates/envs/{env_path.name}/bootstrap-evidence")
        tree(env_path / "partial-repair-evidence", f"candidates/envs/{env_path.name}/partial-repair-evidence")
    for name in ("client-bootstrap-evidence", "client-repair-evidence"):
        tree(sglang / name, "sglang-client/" + name)
    for index, source in enumerate(extra_files):
        if any(part.lower() in BLOCKED_DIRS for part in source.parts[:-1]):
            raise ValueError("--extra-file cannot read a model, cache, or environment directory: " + str(source))
        if optional_stat(source) is None:
            raise FileNotFoundError(source)
        add(source, f"extra/{index:03d}-{source.name}")
    for index, source in enumerate(script_files):
        if not safe_name(source) or source.suffix.lower() not in SCRIPT_SUFFIXES:
            raise ValueError("--script-file accepts plain .py/.sh helpers only")
        if optional_stat(source) is None:
            raise FileNotFoundError(source)
        add(source, f"external-script-hashes/{index:03d}-{source.name}", "script_hash_only")
    return selected, missing


def verify_archive(path, expected_manifest):
    """Re-read all bytes, CRC and hashes; manifest accounts for every entry."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected_names = sorted([item["archive_path"] for item in expected_manifest["files"]] + ["export-manifest.json"])
        if len(names) != len(set(names)) or sorted(names) != expected_names:
            raise ValueError("Archive has unexpected/missing/duplicate entries")
        if json.loads(archive.read("export-manifest.json")) != expected_manifest:
            raise ValueError("Archive manifest readback differs")
        for item in expected_manifest["files"]:
            data = archive.read(item["archive_path"])
            info = archive.getinfo(item["archive_path"])
            if len(data) != item["size"] or sha256(data) != item["sha256"] or f"{info.CRC:08x}" != item["crc32"] or zlib.crc32(data) & 0xffffffff != info.CRC:
                raise ValueError("Archive integrity check failed: " + item["archive_path"])
        damaged = archive.testzip()
        if damaged:
            raise ValueError("Archive CRC failed: " + damaged)


def export_results(project, candidates, sglang, output, extra_files=(), script_files=(), *, max_file_bytes=MAX_FILE_BYTES, max_total_bytes=MAX_TOTAL_BYTES):
    project, candidates, sglang, output = (Path(value).absolute() for value in (project, candidates, sglang, output))
    extra_files = [Path(value).absolute() for value in extra_files]
    script_files = [Path(value).absolute() for value in script_files]
    if optional_stat(output) is not None:
        raise FileExistsError("Destination already exists; use a new name: " + str(output))
    if not stat.S_ISDIR(plain_stat(output.parent).st_mode):
        raise ValueError("Output parent must already exist")
    started = utc_now()
    sources, missing = gather_sources(project, candidates, sglang, extra_files, script_files)
    if len(sources) > MAX_FILES:
        raise ValueError("Too many selected evidence files")
    manifest = {"schema": "hg-session004-evidence-export-v1", "started_utc": started,
                "project": str(project), "candidates": str(candidates), "sglang_venv": str(sglang),
                "limits": {"per_file_bytes": max_file_bytes, "total_uncompressed_bytes": max_total_bytes, "max_selected_files": MAX_FILES},
                "capture_semantics": "Live non-atomic capture. before/after state and script hashes are preserved separately. Changed files during an individual read fail closed.",
                "evaluation_status": "not_assessed_by_exporter", "missing_optional_roots": missing,
                "files": [], "script_hashes_before": [], "script_hashes_after": [], "state_snapshots": []}
    temp = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    total = 0
    def budget(size):
        nonlocal total
        total += size
        if total > max_total_bytes:
            raise ValueError("Total evidence exceeds export size limit")
    def snapshot(archive, phase):
        for task in TASKS:
            source = project / "tasks" / task / "runtime/state.json"
            if optional_stat(source) is None:
                manifest["state_snapshots"].append({"phase": phase, "task": task, "state": "absent", "captured_utc": utc_now()})
                continue
            data, item = read_stable(source, max_file_bytes)
            item["archive_path"] = f"export-snapshots/{phase}/{task}/state.json"
            budget(len(data))
            archive.writestr(item["archive_path"], data)
            manifest["files"].append(item)
            manifest["state_snapshots"].append({"phase": phase, "task": task, "state": "captured", "archive_path": item["archive_path"], "sha256": item["sha256"], "captured_utc": item["captured_utc"]})
    try:
        with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            snapshot(archive, "before")
            for name, (source, kind) in sorted(sources.items()):
                data, item = read_stable(source, max_file_bytes)
                item["archive_path"] = name
                if kind == "script_hash_only":
                    manifest["script_hashes_before"].append(item)
                else:
                    budget(len(data))
                    archive.writestr(name, data)
                    manifest["files"].append(item)
            snapshot(archive, "after")
            for name, (source, kind) in sorted(sources.items()):
                if kind == "script_hash_only":
                    _, item = read_stable(source, max_file_bytes)
                    item["archive_path"] = name
                    manifest["script_hashes_after"].append(item)
            manifest["script_hashes_unchanged"] = [{key: item[key] for key in ("source", "size", "sha256")} for item in manifest["script_hashes_before"]] == [{key: item[key] for key in ("source", "size", "sha256")} for item in manifest["script_hashes_after"]]
            manifest["finished_utc"] = utc_now()
            manifest["payload_uncompressed_bytes"] = total
            raw_manifest = json_bytes(manifest)
            if len(raw_manifest) > max_file_bytes:
                raise ValueError("Manifest exceeds per-file export size limit")
            budget(len(raw_manifest))
            archive.writestr("export-manifest.json", raw_manifest)
        if temp.stat().st_size > max_total_bytes:
            raise ValueError("Compressed archive exceeds export size limit")
        verify_archive(temp, manifest)
        with temp.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            archive_sha = digest.hexdigest()
        # Hard link is atomic and refuses an existing destination on Linux/NTFS.
        # rename/replace would silently overwrite an existing Linux destination.
        os.link(temp, output)
        return {"status": "exported_and_verified", "output": str(output), "size": output.stat().st_size,
                "sha256": archive_sha,
                "payload_files": len(manifest["files"]), "script_hashes": len(manifest["script_hashes_before"]),
                "script_hashes_unchanged": manifest["script_hashes_unchanged"], "finished_utc": manifest["finished_utc"], "evaluation_status": "not_assessed_by_exporter"}
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/private_data/guoxin-hackathon-20261023-session004"))
    parser.add_argument("--candidates", type=Path, default=Path("/root/private_data/guoxin-candidates"))
    parser.add_argument("--sglang-venv", type=Path, default=Path("/root/private_data/sglang-client-004c"))
    parser.add_argument("--output", type=Path, default=Path("/root/hg-session004-results-001.zip"))
    parser.add_argument("--extra-file", action="append", type=Path, default=[], help="Explicit extra evidence text/log/manifest; no binary files")
    parser.add_argument("--script-file", action="append", type=Path, default=[], help="Hash only an actual external .py/.sh helper")
    args = parser.parse_args(argv)
    result = export_results(args.project, args.candidates, args.sglang_venv, args.output, args.extra_file, args.script_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "export_failed", "error": str(error), "error_type": type(error).__name__, "utc": utc_now()}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
