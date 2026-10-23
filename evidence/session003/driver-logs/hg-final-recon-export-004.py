#!/usr/bin/env python3
"""Export ended Recon development evidence, preserving raw bytes and failures.

Running this script creates a new ZIP (exclusive mode); it never stops services,
rewrites evidence, or submits results. Process/port probes are CPU-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import socket
import sys
import zipfile

EXPECTED_RUN_ID = "b6ba20bae32640c0b924511637d6a222"
RUNTIME_SHA = "66e9a97c63469698bfc09fbc1158a28fbd7a48f52877abb21156785a33ab1221"
DRIVER_SHA = "bc09711bb58fb5a7ae4676a840956a4da1caf49f6e6034b8015c5dd9a5bae14e"
LIMIT = 100 * 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def cleanup_probe(state):
    def matches(kind):
        expected = state.get(kind + "_token")
        return bool(expected) and token(state.get(kind + "_pid", 0)) == expected
    with socket.socket() as probe:
        probe.settimeout(1)
        port_open = probe.connect_ex(("127.0.0.1", 30000)) == 0
    result = {"captured_utc": stamp(), "run_id": state["run_id"], "state_status": state.get("status"),
              "supervisor_same_identity_alive": matches("supervisor"),
              "server_same_identity_alive": matches("server"), "port_30000_open": port_open,
              "platform_shutdown_verified": False,
              "scope": "Read-only owned process start-token and loopback-port observation; platform shutdown is separately verified in UI"}
    result["owned_service_cleanup_verified"] = state.get("status") == "stopped" and not any(
        result[key] for key in ("supervisor_same_identity_alive", "server_same_identity_alive", "port_30000_open"))
    return result


def collect(root, home, expected_run_id, source_update=None, probe=cleanup_probe):
    files, sources, absent_optional = {}, [], []

    def add(path, name, required=True):
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name
                and not any(":" in part for part in relative.parts), "unsafe ZIP path: " + name)
        require(name not in files, "duplicate ZIP entry: " + name)
        if not path.exists():
            require(not required, "missing required export source: " + str(path))
            absent_optional.append(str(path))
            return None
        require(path.is_file() and not path.is_symlink(), "source must be a regular nonsymlink file: " + str(path))
        require(path.stat().st_size < LIMIT, "source exceeds 100 MiB: " + str(path))
        raw = path.read_bytes()
        files[name] = raw
        sources.append({"source": str(path), "path": name, "bytes": len(raw), "sha256": sha(raw)})
        return raw

    def folder(relative, destination=None):
        directory = root / relative
        require(directory.is_dir() and not directory.is_symlink(), "missing/nonregular export directory: " + str(directory))
        for path in sorted(directory.rglob("*")):
            require(not path.is_symlink(), "symlink in export directory: " + str(path))
            if path.is_file():
                name = (PurePosixPath(destination or relative) / path.relative_to(directory).as_posix()).as_posix()
                if name in files:
                    require(any(item["source"] == str(path) and item["path"] == name for item in sources), "duplicate path maps to another source")
                    continue
                add(path, name)

    # Read terminal records first; an incomplete original run is expected and retained.
    original_rel = "evidence/session003/recon-matrix/run.json"
    retry_rel = "evidence/session003/recon-matrix-retry/run.json"
    original = json.loads(add(root / original_rel, original_rel))
    retry = json.loads(add(root / retry_rel, retry_rel))
    require(original.get("ended_utc") and original.get("status") in {"complete", "incomplete_or_failed"}, "original matrix has not ended")
    require(retry.get("ended_utc") and retry.get("status") in {"complete", "incomplete_or_failed"}, "retry has not ended")
    control_rel = "evidence/session003/recon-retry-control/control-events.json"
    control = json.loads(add(root / control_rel, control_rel))
    events = control.get("events")
    require(isinstance(events, list) and events and events[0].get("owned_run_id") == expected_run_id,
            "retry control does not identify the expected owned run")
    runtime = root / "tasks/recon2reason-bw-vllm/runtime"
    state = json.loads(add(runtime / "state.json", "evidence/session003/recon-final-runtime/state.json"))
    require(state.get("run_id") == expected_run_id, "final runtime is another run; refusing to mislabel it")
    require(state.get("status") == "stopped", "final state is not stopped; wait for owned cleanup before export")
    config_path = runtime / ("config-" + expected_run_id + ".json")
    require(Path(state.get("config_path", "")) == config_path, "final state config path differs")
    config = json.loads(add(config_path, "evidence/session003/recon-final-runtime/" + config_path.name))
    require(config.get("run_id") == expected_run_id and Path(config.get("task", "")).resolve() == runtime.parent.resolve(), "final config identity differs")
    require(config.get("port") == 30000 and config.get("served_name") == "recon2reason-reasoning-4b", "final model/port identity differs")
    require(sha((root / "scripts/recon2reason_runtime.py").read_bytes()) == RUNTIME_SHA, "final canonical runtime source differs")
    require(sha((root / "scripts/t02_development_bench.py").read_bytes()) == DRIVER_SHA, "final development driver source differs")

    for relative in ("evidence/session003/recon-matrix", "evidence/session003/recon-matrix-retry",
                     "evidence/session003/recon-original-runtime", "evidence/session003/recon-retry-control"):
        folder(relative)
    update_path = source_update or root / "evidence/session003/source-update-003-004.json"
    add(update_path, "evidence/session003/source-update-003-004.json")
    existing_final = root / "evidence/session003/recon-final-runtime"
    if existing_final.exists():
        # Preserve earlier on-disk snapshots under a distinct prefix; never overwrite them.
        folder("evidence/session003/recon-final-runtime", "evidence/session003/recon-final-runtime-existing")
    for path in (runtime / "state.json", config_path, runtime / "server.log", runtime / "supervisor.log"):
        name = "evidence/session003/recon-final-runtime/" + path.name
        if name not in files:
            add(path, name)
    for relative in ("scripts/recon2reason_runtime.py", "scripts/t02_development_bench.py",
                     "tasks/recon2reason-bw-vllm/run.sh", "tasks/recon2reason-bw-vllm/stop.sh", "tasks/recon2reason-bw-vllm/env.sh"):
        add(root / relative, "evidence/session003/recon-final-runtime/sources/" + relative)
    required_home = ("hg-recon-retry-both-003.py", "hg-recon-original-stop-003.txt",
                     "hg-recon-matrix-retry-003.txt", "hg-recon-retry-helper-launch-003.txt")
    for name in required_home:
        add(home / name, "evidence/session003/driver-logs/" + name)
    require(control.get("helper_sha256") == sha(files["evidence/session003/driver-logs/hg-recon-retry-both-003.py"]),
            "exported retry helper does not match the executed helper hash")
    for name in ("hg-recon-matrix-003.txt", "hg-recon-final-stop-003.txt", "hg-retry-recon-last-row-003.py"):
        add(home / name, "evidence/session003/driver-logs/" + name, required=False)
    add(Path(__file__).resolve(), "evidence/session003/driver-logs/" + Path(__file__).name)

    cleanup = probe(state)
    files["evidence/session003/recon-final-runtime/cleanup-check.json"] = (json.dumps(cleanup, indent=2) + "\n").encode("utf-8")
    changed = []
    for entry in sources:
        path = Path(entry["source"])
        current = sha(path.read_bytes()) if path.is_file() and not path.is_symlink() else None
        if current != entry["sha256"]:
            changed.append({"source": entry["source"], "captured_sha256": entry["sha256"], "after_capture_sha256": current})
    manifest = {"schema_version": 4, "captured_utc": stamp(),
                "scope": "Raw original and separate retry Recon development artifacts; archive export does not establish benchmark completeness",
                "official_acceptance": False, "official_accuracy": False, "official_performance": False,
                "formal_ready": False, "submitted": False, "expected_retry_run_id": expected_run_id,
                "original_reported_status": original.get("status"), "retry_reported_status": retry.get("status"),
                "matrix_completeness_independently_audited": False, "cleanup": cleanup,
                "sources_stable_during_capture": not changed, "changed_sources": changed,
                "source_provenance": sources, "optional_sources_absent": absent_optional,
                "manifest_does_not_list_itself": True,
                "files": [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(files.items())]}
    manifest_path = "evidence/session003/recon-final-export.json"
    require(manifest_path not in files, "manifest path collision")
    files[manifest_path] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return files, manifest


def export(root, home, target, expected_run_id=EXPECTED_RUN_ID, source_update=None, probe=cleanup_probe):
    require(not target.exists() and not target.is_symlink(), "ZIP target exists; never overwrite an earlier export")
    files, manifest = collect(root, home, expected_run_id, source_update, probe)
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
    # Verify the archive we actually wrote, rather than only the in-memory source map.
    with zipfile.ZipFile(target) as archive:
        require(archive.testzip() is None, "created ZIP failed CRC validation")
        require(len(archive.namelist()) == len(files), "created ZIP entry count differs")
        for name, raw in files.items():
            require(archive.read(name) == raw, "created ZIP bytes differ: " + name)
    return {"artifact_exported": True, "path": str(target), "files": len(files), "bytes": target.stat().st_size,
            "sha256": sha(target.read_bytes()), "cleanup": manifest["cleanup"],
            "sources_stable_during_capture": manifest["sources_stable_during_capture"],
            "original_reported_status": manifest["original_reported_status"], "retry_reported_status": manifest["retry_reported_status"],
            "official_acceptance": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/private_data/guoxin-hackathon-20261020-session003"))
    parser.add_argument("--home", type=Path, default=Path("/root"))
    parser.add_argument("--target", type=Path, default=Path("/root/hg-session003-recon-final.zip"))
    parser.add_argument("--expected-run-id", default=EXPECTED_RUN_ID)
    parser.add_argument("--source-update", type=Path, help="explicit source-update004 path if stored elsewhere")
    args = parser.parse_args(argv)
    try:
        result = export(args.root.resolve(), args.home.resolve(), args.target, args.expected_run_id, args.source_update)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["cleanup"]["owned_service_cleanup_verified"] and result["sources_stable_during_capture"] else 1
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"error": str(exc), "official_acceptance": False}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
