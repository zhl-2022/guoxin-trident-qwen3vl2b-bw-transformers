#!/usr/bin/env python3
"""Freeze a bounded SCNet development upload, excluding live state and weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("recon2reason-bw-vllm", "robobrain-bw-vllm", "qwen3-asr-bw-vllm",
         "mimo-v2.6-distill-qwen9b-bw-vllm", "trident-qwen3vl2b-bw-transformers",
         "qwen3-forcedaligner-bw-transformers")
EXCLUDED_DIRS = {"runtime", ".git", "__pycache__", ".pytest_cache", "venv", ".venv", "audio-venv", "node_modules", "models", "weights", "build", "dist"}
EXCLUDED_TESTS = {"test_task_repositories.py", "test_update_task_repositories.py", "test_submission_entries.py"}
ALLOWED_SUFFIXES = {".py", ".sh", ".ps1", ".json", ".jsonl", ".md", ".txt", ".toml", ".properties", ".png", ".jpg", ".jpeg", ".wav"}
SMALL_TEST_EVIDENCE = ("evidence/vllm-state-001.json", "evidence/asr-audio-reviewed-plan-002.json", "evidence/robobrain-runtime-source-002.json")
MANIFEST_NAME = "session004-upload-manifest.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def collect():
    payloads = {}

    def add(path):
        relative = path.relative_to(ROOT)
        if set(relative.parts) & EXCLUDED_DIRS or path.suffix.lower() == ".log":
            return
        if path.is_symlink() or getattr(path.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse source refused: " + str(relative))
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in {"LICENSE", "METADATA", "entry_points.txt", ".gitignore"}:
            if path.name != "upstream-source.zip" or relative.parts[0] != "tasks":
                raise ValueError("Unexpected source type: " + str(relative))
        raw = path.read_bytes()
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError("Individual source exceeds5MiB: " + str(relative))
        if path.name == "upstream-source.zip":
            lock = json.loads((path.parent / "upstream-lock.json").read_text())
            if sha(raw) != lock["archive_sha256"]:
                raise ValueError("Candidate source archive differs from upstream lock")
            with zipfile.ZipFile(io.BytesIO(raw)) as source:
                if source.testzip() is not None:
                    raise ValueError("Candidate source archive CRC mismatch")
                expected = {item["path"]: item for item in lock["files"]}
                names = {item.filename for item in source.infolist() if not item.is_dir()}
                if names != set(expected):
                    raise ValueError("Candidate source archive file list differs from lock")
                for name, item in expected.items():
                    data = source.read(name)
                    if len(data) != item["size"] or sha(data) != item["sha256"]:
                        raise ValueError("Candidate source member differs from lock: " + name)
        payloads[relative.as_posix()] = raw

    for name in ("pyproject.toml", ".gitignore"):
        add(ROOT / name)
    for folder in ("src", "scripts", "tests", "fixtures", *("tasks/" + name for name in TASKS)):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and not (folder == "tests" and path.name in EXCLUDED_TESTS):
                add(path)
    for name in SMALL_TEST_EVIDENCE:
        add(ROOT / name)
    if sum(map(len, payloads.values())) > 25 * 1024 * 1024:
        raise ValueError("Upload payload exceeds25MiB")
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/session004-development-upload.zip")
    parser.add_argument("--verification", type=Path, default=ROOT / "evidence/session004-development-upload-verification.json")
    args = parser.parse_args()
    output = args.output.resolve()
    verification = args.verification.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if any(path.exists() for path in (output, verification, sidecar)):
        parser.error("Output already exists; preserve it and choose new paths")
    payloads = collect()
    rows = [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(payloads.items())]
    manifest = {"schema_version": 1, "session": "session004", "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "SCNet development execution upload; not a formal contest submission or official evaluation",
                "tasks": list(TASKS), "source_root": str(ROOT), "source_bytes_normalized": False,
                "weights_included": False, "runtime_or_logs_included": False,
                "historical_evidence_policy": "Only three small fixed CPU regression inputs; no historical raw runs/screenshots/logs",
                "excluded_test_modules": sorted(EXCLUDED_TESTS), "excluded_legacy_task": "recon2reason-bw-sglang",
                "files": rows, "payload_files": len(rows), "payload_bytes": sum(item["bytes"] for item in rows),
                "manifest_self_hash_excluded": True}
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    verification.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(payloads.items()):
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (stat.S_IFREG | (0o755 if name.endswith(".sh") else 0o644)) << 16
            archive.writestr(entry, raw)
        archive.writestr(MANIFEST_NAME, manifest_raw)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {*payloads, MANIFEST_NAME} or archive.testzip() is not None:
            raise ValueError("Upload archive name/CRC mismatch")
        for name in names:
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("Unsafe archive path")
            expected = manifest_raw if name == MANIFEST_NAME else payloads[name]
            if archive.read(name) != expected:
                raise ValueError("Upload member byte mismatch: " + name)
    with sidecar.open("xb") as stream:
        stream.write(manifest_raw)
    changed = sorted(name for name, raw in payloads.items() if not (ROOT / name).is_file() or sha((ROOT / name).read_bytes()) != sha(raw))
    result = {"schema_version": 1, "session": "session004", "archive": str(output), "bytes": output.stat().st_size,
              "sha256": sha(output.read_bytes()), "manifest": str(sidecar), "manifest_sha256": sha(manifest_raw),
              "payload_files": len(rows), "zip_entries": len(rows) + 1, "crc_verified": True,
              "all_member_bytes_verified": True, "all_member_sha256_verified": True,
              "source_changed_after_read": changed, "sources_stable_at_archive_verification": not changed,
              "source_files_modified": False, "old_stage_or_git_modified": False, "gpu_or_platform_used": False}
    with verification.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
