#!/usr/bin/env python3
"""Download only the three reviewed immutable checkpoint file sets.

Plan is offline. Apply is restricted to the named SCNet private-data directory.
No framework install, training data, GPU import, or arbitrary URL argument exists.
Copyright scope notice: no new repository-wide open-source license selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/root/private_data/guoxin-candidates")
TASKS = {
    "forcedaligner": "qwen3-forcedaligner-bw-transformers",
    "mimo9b": "mimo-v2.6-distill-qwen9b-bw-vllm",
    "trident": "trident-qwen3vl2b-bw-transformers",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def check_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or ":" in value or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("Unsafe file path")
    return path


def allowed_download_url(url):
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    return (parts.scheme == "https" and parts.username is None and parts.password is None
            and parts.port in (None, 443)
            and any(host == suffix or host.endswith("." + suffix) for suffix in
                    ("huggingface.co", "hf.co", "modelscope.cn", "pythonhosted.org")))


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if not allowed_download_url(new_url):
            raise ValueError("Redirect host is outside the reviewed official download domains")
        return super().redirect_request(request, response, code, message, headers, new_url)


def file_digest(path, deadline=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Download/verification deadline reached")
            value.update(block)
    return value.hexdigest()


def verified_file(path, item, deadline=None):
    path = Path(path)
    return path.is_file() and not path.is_symlink() and path.stat().st_size == item["size"] and file_digest(path, deadline) == item["sha256"]


def download_file(item, destination, url, deadline, opener=None):
    """Resume an owned partial, then verify disk bytes before atomic publication."""
    if not allowed_download_url(url) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise ValueError("Unreviewed source URL or malformed SHA256")
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Destination is a symlink")
    if destination.exists():
        if not verified_file(destination, item, deadline):
            raise ValueError("Existing final file differs; refusing to overwrite")
        return {"path": destination.name, "status": "reused_verified", "sha256": item["sha256"], "size": item["size"]}
    partial = destination.with_name(destination.name + ".part")
    state = destination.with_name(destination.name + ".part.json")
    identity = {"size": item["size"], "sha256": item["sha256"]}
    if partial.is_symlink() or state.is_symlink():
        raise ValueError("Partial download path is a symlink")
    if partial.exists() and (not state.is_file() or read_json(state) != identity):
        raise ValueError("Unowned or mismatched partial file; manual inspection required")
    if state.exists() and read_json(state) != identity:
        raise ValueError("Partial identity differs")
    if not state.exists():
        json_new(state, identity)
    if not partial.exists():
        partial.touch(exist_ok=False)
    initial_offset = partial.stat().st_size
    if initial_offset > item["size"]:
        raise ValueError("Partial exceeds expected size")
    if initial_offset == item["size"]:
        if not verified_file(partial, item, deadline):
            raise ValueError("Completed partial has wrong SHA256; preserved for inspection")
        os.replace(partial, destination)
        state.unlink()
        return {"path": destination.name, "status": "promoted_verified_partial", **identity}
    opener = opener or urllib.request.build_opener(OfficialRedirects())
    for attempt in range(3):
        if time.monotonic() >= deadline:
            raise TimeoutError("Overall download deadline reached; partial preserved")
        offset = partial.stat().st_size
        headers = {"User-Agent": "guoxin-immutable-checkpoint-downloader/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            timeout = min(30.0, max(1.0, deadline - time.monotonic()))
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                if not allowed_download_url(response.geturl()):
                    raise ValueError("Final response host is not reviewed")
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != item["size"] or int(match[2]) >= item["size"]:
                        raise ValueError("Server Content-Range differs from expected checkpoint")
                elif status == 200:
                    # Only our identity-checked partial may be restarted if Range is ignored.
                    offset = 0
                else:
                    raise ValueError("Unexpected HTTP download status")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > item["size"] - offset:
                    raise ValueError("Server payload exceeds expected file size")
                with partial.open("r+b") as handle:
                    if not offset:
                        handle.truncate(0)
                    handle.seek(offset)
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Overall download deadline reached; partial preserved")
                        data = response.read(min(4 * 1024 * 1024, item["size"] - handle.tell() + 1))
                        if not data:
                            break
                        if handle.tell() + len(data) > item["size"]:
                            raise ValueError("Downloaded bytes exceed expected checkpoint size")
                        handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            if partial.stat().st_size != item["size"]:
                raise OSError("Incomplete response; owned partial retained")
            if not verified_file(partial, item, deadline):
                raise ValueError("Downloaded disk bytes failed SHA256; partial preserved")
            if destination.exists():
                raise ValueError("Final file appeared concurrently; refusing replacement")
            os.replace(partial, destination)
            state.unlink()
            return {"path": destination.name, "status": "downloaded_verified", "resumed_from_bytes": initial_offset, **identity}
        except (OSError, urllib.error.URLError, socket.timeout):
            if attempt == 2 or time.monotonic() >= deadline:
                raise
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError("Unreachable download retry state")


def load_files(task, provider):
    slug = TASKS[task]
    directory = ROOT / "tasks" / slug
    original = read_json(directory / "model-lock.json")
    sources = read_json(directory / "download-sources.json")
    if sources["original_huggingface_revision"] != original["revision"] or not re.fullmatch(r"[0-9a-f]{40}", original["revision"]):
        raise ValueError("Checkpoint source revision differs")
    expected = {x["path"]: x for x in original["files"]}
    if len(sources["files"]) != len(expected) or set(x["path"] for x in sources["files"]) != set(expected):
        raise ValueError("Download source file set differs from reviewed model lock")
    rows = []
    for item in sources["files"]:
        name = item["path"]
        check_relative(name)
        if item["size"] != expected[name]["size"] or ("sha256" in expected[name] and item["sha256"] != expected[name]["sha256"]):
            raise ValueError("Download digest/size differs from reviewed model lock")
        key = provider + "_url"
        url = item[key]
        if provider == "modelscope" and not re.fullmatch(r"[0-9a-f]{40}", item["modelscope_revision"]):
            raise ValueError("ModelScope per-file revision must be immutable")
        revision = item["modelscope_revision"] if provider == "modelscope" else original["revision"]
        if "/resolve/" + revision + "/" not in url or not allowed_download_url(url):
            raise ValueError("URL is not pinned to the exact reviewed revision")
        rows.append({**item, "download_url": url})
    return original, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument("--task", choices=tuple(TASKS) + ("all",), default="all")
    parser.add_argument("--provider", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    chosen = list(TASKS) if args.task == "all" else [args.task]
    planned = []
    for task in chosen:
        lock, files = load_files(task, args.provider)
        planned.append({"task": task, "model_id": lock["model_id"], "original_model_revision": lock["revision"],
                        "destination": (BASE / "models" / task).as_posix(), "files": files, "weight_bytes": lock["weight_bytes"]})
    if args.action == "plan":
        print(json.dumps({"mode": "offline_plan", "provider": args.provider, "tasks": planned,
                          "downloads_performed": False, "gpu_imported": False}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != "linux" or not 60 <= args.max_seconds <= 14400:
        parser.error("Download apply requires Linux and max-seconds 60..14400")
    # Inspect every existing ancestor; never follow a user-supplied symlink outside this fixed target.
    if any(path.is_symlink() for path in (BASE, *BASE.parents)):
        raise ValueError("Private-data destination contains a symlink; review storage mapping first")
    BASE.mkdir(parents=True, exist_ok=True)
    lock_file = BASE / ".candidate-model-download.lock"
    token = str(uuid.uuid4())
    descriptor = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "token": token}, handle)
    report = {"run_id": token, "provider": args.provider, "status": "failed", "files": [], "gpu_imported": False}
    deadline = time.monotonic() + args.max_seconds
    try:
        for task in planned:
            destination = Path(task["destination"])
            if destination.is_symlink() or destination.parent.is_symlink():
                raise ValueError("Model directory is a symlink")
            destination.mkdir(parents=True, exist_ok=True)
            needed = 0
            for item in task["files"]:
                final = destination / item["path"]
                partial = final.with_name(final.name + ".part")
                partial_state = final.with_name(final.name + ".part.json")
                present = final.stat().st_size if final.is_file() else 0
                if not final.exists() and partial.is_file() and partial_state.is_file() and read_json(partial_state) == {"size": item["size"], "sha256": item["sha256"]}:
                    present = partial.stat().st_size
                needed += max(0, item["size"] - present)
            if shutil.disk_usage(destination).free < needed + 512 * 1024 * 1024:
                raise ValueError("Insufficient free storage for reviewed checkpoint plus reserve")
            for item in task["files"]:
                relative = check_relative(item["path"])
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                result = download_file(item, target, item["download_url"], deadline)
                report["files"].append({"task": task["task"], **result})
                print(json.dumps({"event": "checkpoint_file_verified", "task": task["task"], **result}), flush=True)
        report["status"] = "completed_all_files_verified"
        return 0
    finally:
        json_new(BASE / f"model-download-{token}.json", report)
        if lock_file.is_file() and read_json(lock_file).get("token") == token:
            lock_file.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
