#!/usr/bin/env python3
"""Package six frozen session004 trees using exact per-repository manifests.

Default is a read-only plan. --pack creates NEW ZIPs and an external closure.
No Git, network, model imports, GPU, source rewrites or overwrite operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("recon2reason-bw-vllm", "robobrain-bw-vllm", "qwen3-asr-bw-vllm",
         "mimo-v2.6-distill-qwen9b-bw-vllm", "trident-qwen3vl2b-bw-transformers",
         "qwen3-forcedaligner-bw-transformers")
MANIFEST = "repository-manifest004.json"
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_REPOSITORY_BYTES = 100 * 1024 * 1024
MAX_FILES = 10000
BLOCK = 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def no_links(path):
    """Reject symlinks/reparse points at the target and every existing ancestor."""
    absolute = Path(path).absolute()
    for item in (absolute, *absolute.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse path refused: " + str(item))


def no_stage003(path):
    if any(part.casefold() == "repositories-stage003" for part in Path(path).parts):
        raise ValueError("Historical repositories-stage003 path refused")


def member_name(value):
    if not isinstance(value, str) or not value or any(char in value for char in "\\:\0<>|?*") or any(ord(char) < 32 for char in value):
        raise ValueError("Manifest requires a safe relative POSIX name")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Path traversal/alias refused: " + value)
    for part in path.parts:
        if part != part.rstrip(" .") or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part):
            raise ValueError("Windows path alias refused: " + value)
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest JSON key: " + key)
        result[key] = value
    return result


def regular_info(path, max_bytes=MAX_FILE_BYTES):
    no_links(path)
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        raise ValueError("Expected bounded regular source file: " + str(path))
    return info


def stable_identity(before, after):
    return (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)


def file_digest(path, max_bytes=MAX_FILE_BYTES):
    before = regular_info(path, max_bytes)
    count = crc = 0
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        opened = os.fstat(source.fileno())
        if not stable_identity(before, opened):
            raise ValueError("Source identity changed before read: " + str(path))
        for block in iter(lambda: source.read(BLOCK), b""):
            count += len(block)
            if count > max_bytes:
                raise ValueError("Source grew beyond size limit")
            digest.update(block)
            crc = zlib.crc32(block, crc)
        after = os.fstat(source.fileno())
    if not stable_identity(before, after) or not stable_identity(before, regular_info(path, max_bytes)) or count != before.st_size:
        raise ValueError("Source changed during read: " + str(path))
    return {"bytes": count, "sha256": digest.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}


def read_manifest(path):
    before = regular_info(path)
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(raw) != before.st_size or not stable_identity(before, after) or not stable_identity(before, regular_info(path)):
        raise ValueError("Manifest changed during read")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite manifest value")))
    return raw, value


def exact_tree(directory, expected_files):
    """Every scan failure, extra directory/file, link and missing file is fatal."""
    no_links(directory)
    allowed_dirs = {""}
    for name in expected_files:
        path = PurePosixPath(name)
        allowed_dirs.update(parent.as_posix() for parent in path.parents if parent.as_posix() != ".")
    seen = set()

    def failure(error):
        raise error

    for folder, dirs, files in os.walk(directory, followlinks=False, onerror=failure):
        for leaf in dirs:
            path = Path(folder) / leaf
            no_links(path)
            name = path.relative_to(directory).as_posix()
            member_name(name)
            if name not in allowed_dirs:
                raise ValueError("Unlisted source directory: " + name)
        for leaf in files:
            path = Path(folder) / leaf
            name = member_name(path.relative_to(directory).as_posix())
            regular_info(path)
            if name not in expected_files:
                raise ValueError("Unlisted source file: " + name)
            seen.add(name)
    if seen != set(expected_files):
        raise ValueError("Missing manifest source files: " + ", ".join(sorted(set(expected_files) - seen)))


def inspect_repository(directory, task):
    raw, manifest = read_manifest(directory / MANIFEST)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
            or manifest.get("session") != "session004" or manifest.get("primary_task") != task
            or manifest.get("manifest_self_hash_excluded") is not True):
        raise ValueError("Repository manifest schema/session/task differs: " + task)
    rows = manifest.get("files")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_FILES:
        raise ValueError("Invalid/empty repository allowlist: " + task)
    files = {}; folded = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Manifest file row is not an object")
        name = member_name(row.get("path"))
        if name == MANIFEST or name in files or name.casefold() in folded:
            raise ValueError("Duplicate/casefold/reserved manifest member: " + name)
        if type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= MAX_FILE_BYTES or not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Manifest file size/SHA is invalid: " + name)
        files[name] = {"path": name, "bytes": row["bytes"], "sha256": row["sha256"]}
        folded.add(name.casefold())
    payload_bytes = sum(row["bytes"] for row in files.values())
    if type(manifest.get("payload_bytes")) is not int or manifest["payload_bytes"] != payload_bytes or payload_bytes + len(raw) > MAX_REPOSITORY_BYTES:
        raise ValueError("Manifest total size differs/exceeds100MiB")
    manifest_row = {"path": MANIFEST, "bytes": len(raw), "sha256": sha(raw)}
    files[MANIFEST] = manifest_row
    exact_tree(directory, files)
    verified = []
    for name, row in sorted(files.items()):
        measured = file_digest(directory / name)
        if measured["bytes"] != row["bytes"] or measured["sha256"] != row["sha256"]:
            raise ValueError("Frozen source size/SHA differs: " + task + "/" + name)
        verified.append({**row, "crc32": measured["crc32"]})
    return {"task": task, "directory": str(directory), "manifest_sha256": sha(raw),
            "manifest_bytes": len(raw), "payload_files": len(rows), "zip_members": len(verified),
            "payload_bytes": payload_bytes, "total_archived_bytes": payload_bytes + len(raw), "files": verified}


def inspect_stage(stage_root):
    stage_root = Path(stage_root).absolute()
    no_stage003(stage_root); no_links(stage_root)
    info = stage_root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("Stage root is not a plain directory")
    # scandir errors propagate; no silent partial six-task inventory.
    with os.scandir(stage_root) as entries:
        items = list(entries)
    if {item.name for item in items} != set(TASKS) or len(items) != len(TASKS):
        raise ValueError("Frozen stage must contain exactly the six registered directories")
    for item in items:
        no_links(Path(item.path))
        if not item.is_dir(follow_symlinks=False):
            raise ValueError("Registered repository is not a directory")
    return [inspect_repository(stage_root / task, task) for task in TASKS]


def plan(stage_root, output_dir, closure=None):
    stage_root = Path(stage_root).absolute()
    output_dir = Path(output_dir).absolute()
    closure = Path(closure).absolute() if closure is not None else output_dir / "delivery-closure004.json"
    for path in (stage_root, output_dir, closure):
        no_stage003(path); no_links(path)
    source = stage_root.resolve(); output = output_dir.resolve(); closure_resolved = closure.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output) or closure_resolved.is_relative_to(source):
        raise ValueError("Output/closure cannot contain or be inside the frozen input")
    if output.is_relative_to(closure_resolved):
        raise ValueError("Closure file cannot be the output directory or one of its parents")
    if output_dir.exists() or closure.exists():
        raise ValueError("Output directory and external closure must be NEW; overwrite refused")
    if closure_resolved in {output / (task + ".zip") for task in TASKS}:
        raise ValueError("Closure cannot collide with a ZIP output")
    repositories = inspect_stage(stage_root)
    return {"stage_root": stage_root, "output_dir": output_dir, "closure": closure, "repositories": repositories}


def fingerprint(repositories):
    return sha(encoded(repositories))


def verify_zip(path, repository):
    expected = {row["path"]: row for row in repository["files"]}
    before = regular_info(path, MAX_REPOSITORY_BYTES + 10 * 1024 * 1024)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); names = [info.filename for info in infos]
        if len(names) != repository["zip_members"] or len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}) or set(names) != set(expected):
            raise ValueError("ZIP member count/set differs from exact manifest allowlist")
        for info in infos:
            member_name(info.filename)
            row = expected[info.filename]
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1 or info.file_size != row["bytes"]:
                raise ValueError("ZIP type/flags/size differs: " + info.filename)
            digest = hashlib.sha256(); count = crc = 0
            with archive.open(info) as source:
                for block in iter(lambda: source.read(BLOCK), b""):
                    count += len(block)
                    if count > row["bytes"]:
                        raise ValueError("ZIP member exceeds expected bytes")
                    digest.update(block); crc = zlib.crc32(block, crc)
            if (count != row["bytes"] or digest.hexdigest() != row["sha256"]
                    or f"{crc & 0xffffffff:08x}" != row["crc32"] or crc & 0xffffffff != info.CRC):
                raise ValueError("ZIP member size/SHA/CRC mismatch: " + info.filename)
    measured = file_digest(path, MAX_REPOSITORY_BYTES + 10 * 1024 * 1024)
    if not stable_identity(before, regular_info(path, MAX_REPOSITORY_BYTES + 10 * 1024 * 1024)):
        raise ValueError("ZIP changed between member verification and archive SHA read")
    return {"all_member_names_sizes_sha256_crc_verified": True, "complete_set_verified": True,
            "member_count": len(expected), "zip_bytes": measured["bytes"], "zip_sha256": measured["sha256"]}


def write_zip(path, repository):
    no_links(path)
    if Path(path).exists():
        raise ValueError("ZIP destination already exists")
    root = Path(repository["directory"])
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=False) as archive:
        for row in repository["files"]:
            source_path = root / row["path"]
            before = regular_info(source_path)
            info = zipfile.ZipInfo(row["path"])  # fixed transport timestamp, source bytes unchanged
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | (0o755 if row["path"].endswith(".sh") else 0o644)) << 16
            info.file_size = row["bytes"]
            digest = hashlib.sha256(); count = 0
            with source_path.open("rb") as source, archive.open(info, "w") as destination:
                if not stable_identity(before, os.fstat(source.fileno())):
                    raise ValueError("Source identity changed before ZIP read")
                for block in iter(lambda: source.read(BLOCK), b""):
                    count += len(block)
                    if count > row["bytes"]:
                        raise ValueError("Source grew while packaging")
                    digest.update(block); destination.write(block)
                after = os.fstat(source.fileno())
            if count != row["bytes"] or digest.hexdigest() != row["sha256"] or not stable_identity(before, after) or not stable_identity(before, regular_info(source_path)):
                raise ValueError("Frozen source changed while ZIP was written")
    return verify_zip(path, repository)


def package(prepared):
    # Revalidate ALL source sets and every source hash before output mutation.
    current = plan(prepared["stage_root"], prepared["output_dir"], prepared["closure"])
    if fingerprint(current["repositories"]) != fingerprint(prepared["repositories"]):
        raise ValueError("Frozen stage changed since plan")
    output = current["output_dir"]; closure = current["closure"]
    no_links(output); no_links(closure)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for repository in current["repositories"]:
        path = output / (repository["task"] + ".zip")
        verification = write_zip(path, repository)
        results.append({key: repository[key] for key in ("task", "directory", "manifest_sha256", "manifest_bytes", "payload_files", "zip_members", "payload_bytes", "total_archived_bytes")}
                       | {"zip": str(path), **verification})
    after = inspect_stage(current["stage_root"])
    if fingerprint(after) != fingerprint(current["repositories"]):
        raise ValueError("Frozen stage changed after ZIP readback; no successful closure generated")
    expected_outputs = {task + ".zip" for task in TASKS}
    with os.scandir(output) as entries:
        if {item.name for item in entries} != expected_outputs:
            raise ValueError("Unexpected output appeared before closure")
    report = {"schema_version": 1, "session": "session004", "status": "packaged_and_verified",
              "created_utc": datetime.now(timezone.utc).isoformat(), "stage_root": str(current["stage_root"]),
              "output_dir": str(output), "closure_path": str(closure), "repository_count": len(results),
              "repositories": results, "source_inventory_fingerprint": fingerprint(after),
              "exact_allowlists_and_all_source_hashes_verified_before_and_after": True,
              "all_zip_member_sets_sizes_sha256_crc_verified": True,
              "source_or_manifest_bytes_modified": False, "git_network_gpu_or_models_used": False,
              "official_evaluation_or_submission_assessed": False,
              "scope": "Local source repository transport closure only; manifest is the frozen allowlist, not a new content/accuracy audit",
              "zip_metadata_policy": "Fixed ZIP transport timestamps; regular files only; .sh mode755, other mode644; no source-byte normalization",
              "closure_self_hash_excluded": True}
    raw = encoded(report)
    closure.parent.mkdir(parents=True, exist_ok=True); no_links(closure)
    with closure.open("xb") as destination:
        destination.write(raw)
    if closure.read_bytes() != raw:
        raise ValueError("Closure readback differs")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=ROOT / "dist/repositories-stage004")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/final-delivery-zips-004")
    parser.add_argument("--closure", type=Path, help="External to ZIPs; default NEW output-dir/delivery-closure004.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--pack", action="store_true", help="Actually create NEW local ZIPs and closure")
    action.add_argument("--dry-run", action="store_true", help="Read-only manifest/source verification (default)")
    args = parser.parse_args(argv)
    try:
        prepared = plan(args.stage_root, args.output_dir, args.closure)
        if args.pack:
            result = package(prepared)
            result = {**result, "closure_sha256": sha(prepared["closure"].read_bytes())}
        else:
            result = {"mode": "dry_run_no_writes", "stage_root": str(prepared["stage_root"]),
                      "output_dir": str(prepared["output_dir"]), "closure": str(prepared["closure"]),
                      "repositories": [{key: row[key] for key in ("task", "manifest_sha256", "payload_files", "zip_members", "payload_bytes")} for row in prepared["repositories"]],
                      "source_inventory_fingerprint": fingerprint(prepared["repositories"]), "source_bytes_modified": False}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
