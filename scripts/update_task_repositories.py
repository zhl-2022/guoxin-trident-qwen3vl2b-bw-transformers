#!/usr/bin/env python3
"""Plan or explicitly update managed source/evidence in existing local repositories."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("task_repository_preparation", ROOT / "scripts/prepare_task_repositories.py")
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)
MAX_LOG_BYTES = 8 * 1024 * 1024
MANIFEST = "repository-manifest.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_file(root, name):
    name = preparation.relative_name(name)
    if preparation.forbidden_name(name):
        raise ValueError("Managed path cannot include runtime, credentials, weights or .git: " + name)
    return preparation.safe_source(root, name)


def read_manifest(destination, task):
    path = checked_file(destination, MANIFEST)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Require existing bounded repository-manifest.json: " + task)
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("primary_task") != task or not isinstance(data.get("files"), list):
        raise ValueError("Existing manifest identity/schema differs: " + task)
    entries = {}
    seen = set()
    for item in data["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Unexpected old manifest entry schema")
        name = preparation.relative_name(item["path"])
        checked_file(destination, name)
        if name == MANIFEST or name.casefold() in seen:
            raise ValueError("Duplicate/self-referential managed manifest path")
        if type(item["bytes"]) is not int or item["bytes"] < 0 or not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Invalid old manifest byte/hash metadata")
        seen.add(name.casefold())
        entries[name] = item
    return raw, entries


def validate_repository(destination, task):
    if not destination.is_dir():
        raise ValueError("Existing repository directory is missing: " + str(destination))
    for parent in (destination, *destination.parents):
        if parent.is_symlink() or (parent.exists() and getattr(parent.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Repository path cannot traverse a symlink/reparse point")
    git_path = destination / ".git"
    if not git_path.is_dir() or git_path.is_symlink() or getattr(git_path.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("Require an ordinary existing .git directory; linked worktrees are not accepted")
    state = json.loads(checked_file(destination, "ROOT_TASK.json").read_text(encoding="utf-8"))
    if state.get("primary_task") != task or state.get("submitted") is not False or state.get("formal_ready") is not False:
        raise ValueError("Existing target identity/formal state differs; review manually: " + task)
    return state


def prepare_update(source_root, repository_root, tasks, extra_evidence=(), normalize_text_lf=False):
    sources, prepared = preparation.plan(source_root, repository_root, tasks)
    for name in extra_evidence:
        relative = preparation.relative_name(name)
        if relative.split("/", 1)[0] not in {"evidence", "docs", "fixtures"}:
            raise ValueError("Extra evidence must stay in evidence/docs/fixtures")
        sources.add(relative, allow_asset=True)
    if extra_evidence:
        sources.collect_markdown_links()
        prepared = {task: preparation.task_payloads(sources, task)[0] for task in tasks}
    if sources.missing:
        raise ValueError("Missing required source/evidence; nothing updated: " + ", ".join(sorted(sources.missing)))
    updates = []
    for task, payloads in prepared.items():
        destination = repository_root / task
        state = validate_repository(destination, task)
        old_raw, old = read_manifest(destination, task)
        # Preserve explicit existing real metadata when the source still has no value.
        new_state = json.loads(payloads["ROOT_TASK.json"])
        for key in ("repository_url", "image_reference"):
            value = state.get(key)
            if new_state.get(key) is None and value:
                if not isinstance(value, str) or "\n" in value or "\r" in value:
                    raise ValueError("Existing repository metadata is malformed: " + key)
                new_state[key] = value
        payloads["ROOT_TASK.json"] = (json.dumps(new_state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if normalize_text_lf:
            # Normalize after retaining explicit local metadata, so its recorded hash is exact.
            payloads = preparation.normalize_payloads_text_lf(payloads)
        for name, raw in payloads.items():
            preparation.scan_payload(name, raw)
            if Path(name).suffix.lower() == ".log" and len(raw) > MAX_LOG_BYTES:
                raise ValueError("Log exceeds 8 MiB; review/archive explicitly rather than silently copy or trim: " + name)
        if sum(map(len, payloads.values())) > preparation.MAX_BYTES:
            raise ValueError("Updated repository source exceeds 50 MiB")
        operations = []
        preconditions = {}
        for name in sorted(set(old) | set(payloads)):
            path = checked_file(destination, name)
            if path.exists() and not path.is_file():
                raise ValueError("Managed target is not an ordinary file: " + name)
            if path.is_file():
                if path.stat().st_size > preparation.MAX_BYTES:
                    raise ValueError("Existing managed file is unexpectedly large: " + name)
                current = digest(path.read_bytes())
            else:
                current = None
            expected = digest(payloads[name]) if name in payloads else None
            previous = old.get(name, {}).get("sha256")
            # A previous interrupted update is resumable only when exact new bytes already match.
            if current != previous and current != expected:
                raise ValueError("Managed working file changed outside manifest/source; refusing to overwrite: " + name)
            if name not in old and current is not None and current != expected:
                raise ValueError("Unmanaged target would be overwritten: " + name)
            preconditions[name] = current
            if current != expected:
                operations.append({"path": name, "operation": "remove" if expected is None else "add" if current is None else "replace",
                                   "old_sha256": current, "new_sha256": expected,
                                   "new_bytes": len(payloads[name]) if name in payloads else 0})
        entries = [{"path": name, "bytes": len(raw), "sha256": digest(raw)} for name, raw in sorted(payloads.items())]
        manifest = {"schema_version": 1, "primary_task": task, "formal_ready": False, "submitted": False,
                    "files": entries, "total_bytes": sum(item["bytes"] for item in entries),
                    "manifest_scope": "All repository payload files except repository-manifest.json itself"}
        manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        updates.append({"task": task, "destination": destination, "payloads": payloads,
                        "operations": operations, "preconditions": preconditions,
                        "old_manifest_sha256": digest(old_raw), "manifest_payload": manifest_payload,
                        "new_manifest_sha256": digest(manifest_payload)})
    return updates


def verify_preconditions(updates):
    # Preflight every selected repository before writing anything in any repository.
    for update in updates:
        destination = update["destination"]
        validate_repository(destination, update["task"])
        if digest(checked_file(destination, MANIFEST).read_bytes()) != update["old_manifest_sha256"]:
            raise ValueError("Manifest changed since plan; replan before applying")
        for name, expected in update["preconditions"].items():
            path = checked_file(destination, name)
            actual = digest(path.read_bytes()) if path.is_file() else None
            if actual != expected:
                raise ValueError("Working tree changed since plan; replan before applying: " + name)


def atomic_write(destination, name, payload):
    target = checked_file(destination, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = name + ".sync-temp-" + uuid.uuid4().hex
    temporary = checked_file(destination, temporary_name)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        # Recheck parent/link safety immediately before replacing the managed file.
        checked_file(destination, name)
        os.replace(temporary, target)
    finally:
        if temporary.is_file():
            temporary.unlink()


def apply_updates(updates):
    verify_preconditions(updates)
    for update in updates:
        destination = update["destination"]
        for item in update["operations"]:
            name = item["path"]
            if item["operation"] == "remove":
                checked_file(destination, name).unlink()
            else:
                atomic_write(destination, name, update["payloads"][name])
        # Write manifest last; .git and any unmanaged files remain untouched.
        atomic_write(destination, MANIFEST, update["manifest_payload"])
        for name, payload in update["payloads"].items():
            if digest(checked_file(destination, name).read_bytes()) != digest(payload):
                raise ValueError("Post-update hash mismatch: " + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--repository-root", type=Path, required=True, help="absolute parent of existing three local repositories")
    parser.add_argument("--task", choices=preparation.TASKS, action="append")
    parser.add_argument("--extra-evidence", action="append", default=[], help="explicit existing source-relative evidence/docs/fixtures file")
    parser.add_argument("--apply", action="store_true", help="explicitly update managed working-tree files only")
    parser.add_argument("--normalize-text-lf", action="store_true", help="explicit Sourcefind transport: strict UTF-8 text CRLF->LF; binary unchanged; record both hashes")
    args = parser.parse_args(argv)
    try:
        if not args.repository_root.is_absolute():
            raise ValueError("repository-root must be absolute")
        tasks = tuple(dict.fromkeys(args.task or preparation.TASKS))
        updates = prepare_update(args.source_root.absolute(), args.repository_root, tasks, args.extra_evidence,
                                 normalize_text_lf=args.normalize_text_lf)
        if args.apply:
            apply_updates(updates)
        report = {"mode": "apply" if args.apply else "plan_only", "git_commands_run": False,
                  "git_metadata_modified": False, "commit_created": False, "pushed": False,
                  "submitted": False, "formal_ready": False, "gpu_used": False,
                  "normalize_text_lf": args.normalize_text_lf,
                  "repositories": [{"task": item["task"], "path": str(item["destination"]),
                                    "operations": item["operations"], "files": len(item["payloads"]),
                                    "bytes": sum(map(len, item["payloads"].values())),
                                    "old_manifest_sha256": item["old_manifest_sha256"],
                                    "new_manifest_sha256": item["new_manifest_sha256"]} for item in updates]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc), "git_commands_run": False, "pushed": False, "submitted": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
