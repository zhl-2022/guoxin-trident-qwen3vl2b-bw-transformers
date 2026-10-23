#!/usr/bin/env python3
"""Stdlib-only, bounded SCNet vLLM environment probe; no models or installs."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import traceback


PACKAGES = ("torch", "torchvision", "transformers", "vllm", "vllm-hcu", "triton",
            "flash-attn", "lightop", "aiter")
SWITCHES = ("VLLM_HCU_USE_PD_SPLIT", "VLLM_HCU_USE_CUSTOM_FLASH_ATTN",
            "VLLM_HCU_USE_CUSTOM_OPS", "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM",
            "VLLM_ATTENTION_BACKEND", "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE",
            "VLLM_USE_V1", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_KV_CACHE_LAYOUT",
            "VLLM_TARGET_DEVICE")
CLI_FLAGS = ("--attention-backend", "--mm-encoder-attn-backend", "--enforce-eager",
             "--limit-mm-per-prompt", "--mm-processor-kwargs", "--skip-mm-profiling",
             "--dtype", "--tensor-parallel-size", "--max-model-len",
             "--max-num-seqs", "--gpu-memory-utilization")
MARKER = "SCNET_PROBE_JSON="
CHILD_PROGRAM = r'''
import importlib
import json
import sys
import traceback
sys.dont_write_bytecode = True
kind = sys.argv[1]
result = {"check": kind, "ok": False}
try:
    module = importlib.import_module(kind)
    result.update(version=str(getattr(module, "__version__", "unknown")), file=getattr(module, "__file__", None))
    if kind == "torch":
        result.update(hip=module.version.hip, cuda_build=module.version.cuda)
        available = module.cuda.is_available()
        result["cuda_available"] = available
        if available:
            count = module.cuda.device_count()
            result["device_count"] = count
            result["devices"] = []
            for index in range(min(count, 2)):
                props = module.cuda.get_device_properties(index)
                result["devices"].append({"index": index, "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "gcn_arch_name": getattr(props, "gcnArchName", None),
                    "major": getattr(props, "major", None), "minor": getattr(props, "minor", None)})
            result["additional_devices_omitted"] = max(0, count - 2)
    if kind == "vllm":
        from vllm.platforms import current_platform
        result["current_platform"] = {"class": type(current_platform).__name__,
            "module": type(current_platform).__module__,
            "device_type": getattr(current_platform, "device_type", None),
            "device_name": getattr(current_platform, "device_name", None),
            "dispatch_key": getattr(current_platform, "dispatch_key", None)}
    result["ok"] = True
except BaseException as exc:
    result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                       "traceback": traceback.format_exc()[-5500:]}
print("SCNET_PROBE_JSON=" + json.dumps(result, ensure_ascii=False, default=str), flush=True)
raise SystemExit(0 if result["ok"] else 1)
'''


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class BoundedStream:
    """Drain the whole pipe while retaining only a bounded head and tail."""
    def __init__(self, limit: int = 16384, patterns: tuple = ()):
        self.limit = limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self.error = None
        self.patterns = patterns
        self.matches = {}
        self.search_tail = b""

    def drain(self, pipe) -> None:
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                self.total += len(chunk)
                search = self.search_tail + chunk
                for pattern in self.patterns:
                    if pattern not in self.matches and pattern.encode() in search:
                        self.matches[pattern] = next(
                            line[:500] for line in search.decode("utf-8", errors="replace").splitlines()
                            if pattern in line)
                self.search_tail = search[-128:]
                needed = self.limit // 2 - len(self.head)
                if needed > 0:
                    self.head.extend(chunk[:needed])
                    chunk = chunk[needed:]
                if chunk:
                    self.tail.extend(chunk)
                    del self.tail[:-self.limit // 2]
        except OSError as exc:
            self.error = str(exc)[:500]
        finally:
            pipe.close()

    def result(self) -> dict:
        truncated = self.total > self.limit
        gap = b"\n[... output truncated ...]\n" if truncated else b""
        return {"text": (bytes(self.head) + gap + bytes(self.tail)).decode("utf-8", errors="replace"),
                "total_bytes": self.total, "truncated": truncated, "read_error": self.error,
                "stream_matches": self.matches}


def run_bounded(command: list[str], timeout: float, patterns: tuple = ()) -> dict:
    started = time.monotonic()
    result = {"command": command, "timeout_seconds": timeout, "timed_out": False}
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=os.name != "nt")
    except OSError as exc:
        result.update(returncode=None, launch_error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.monotonic() - started)
        return result
    streams = [BoundedStream(patterns=patterns), BoundedStream()]
    readers = [threading.Thread(target=buffer.drain, args=(pipe,), daemon=True)
               for buffer, pipe in zip(streams, (process.stdout, process.stderr))]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    finally:
        for reader in readers:
            reader.join(timeout=2)
    result.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started,
                  stdout=streams[0].result(), stderr=streams[1].result(),
                  pipe_readers_complete=not any(reader.is_alive() for reader in readers))
    return result


def import_probe(name: str) -> dict:
    result = run_bounded([sys.executable, "-B", "-c", CHILD_PROGRAM, name], 30)
    result["check"] = name
    result["data"] = None
    for line in result.get("stdout", {}).get("text", "").splitlines():
        if line.startswith(MARKER):
            try:
                result["data"] = json.loads(line[len(MARKER):])
            except json.JSONDecodeError:
                result["structured_output_error"] = "truncated or invalid child JSON"
    result["passed"] = bool(result["data"] and result["data"].get("ok")
                            and result.get("returncode") == 0 and not result["timed_out"])
    return result


def source_probe() -> dict:
    """Inspect only metadata-listed platform/env/Qwen3-VL/vision source files."""
    exact = {"envs.py", "qwen3_vl.py", "patch_qwen3_vl.py", "vision.py",
             "mm_encoder_attention.py", "vit_attn_wrappers.py", "arg_utils.py",
             "hcu.py", "platform.py", "platform_hcu.py", "rocm.py"}
    specific = re.compile(r"VLLM_HCU_USE_(?:PD_SPLIT|CUSTOM_FLASH_ATTN|CUSTOM_OPS|CUSTOM_QUANTIZATION_GEMM)"
                          r"|VLLM_HCU_[A-Z0-9_]*(?:VISION|VIT)[A-Z0-9_]*"
                          r"|def get_vit_attn_backend|mm_encoder_attn_backend|FLASH_ATTN_CUSTOM"
                          r"|class MMEncoderAttention|TORCH_SDPA|TRITON_ATTN")
    symbol_pattern = re.compile(r"\bVLLM_HCU_[A-Z0-9_]+\b")
    result = {"files": [], "snippets": [], "hcu_symbols": {}, "errors": [],
              "limits": {"files": 32, "per_file_bytes": 262144, "total_bytes": 2097152,
                         "snippet_windows": 48, "snippet_line_characters": 240},
              "scope": "distribution metadata paths under vllm or vllm_hcu only"}
    candidates = []
    for package, namespace in (("vllm-hcu", "vllm_hcu"), ("vllm", "vllm")):
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            result["errors"].append({"package": package, "error": "distribution missing"})
            continue
        root = Path(dist.locate_file(namespace)).resolve()
        for entry in dist.files or []:
            relative = str(entry).replace("\\", "/")
            path_parts = relative.split("/")
            selected = (Path(relative).name in exact
                        or namespace == "vllm_hcu" and "/platforms/" in relative
                        or namespace == "vllm_hcu" and Path(relative).name.endswith("env.py"))
            if path_parts[0] != namespace or not relative.endswith(".py") or not selected:
                continue
            path = Path(dist.locate_file(entry)).resolve()
            if not path.is_relative_to(root):
                continue
            priority = (0 if Path(relative).name == "envs.py" else
                        1 if namespace == "vllm_hcu" else 2)
            candidates.append((priority, relative, path))
    total = 0
    for _, relative, path in sorted(candidates)[:32]:
        try:
            size = path.stat().st_size
            if size > 262144 or total + size > 2097152:
                result["files"].append({"file": str(path), "skipped": "source byte limit", "bytes": size})
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            total += size
        except OSError as exc:
            result["errors"].append({"file": str(path), "error": str(exc)[:500]})
            continue
        lines = text.splitlines()
        result["files"].append({"file": str(path), "relative": relative, "bytes": size,
                                "lines": len(lines)})
        covered = set()
        file_windows = 0
        for index, line in enumerate(lines):
            for symbol in symbol_pattern.findall(line):
                if symbol not in result["hcu_symbols"] and len(result["hcu_symbols"]) >= 128:
                    continue
                locations = result["hcu_symbols"].setdefault(symbol, [])
                if len(locations) < 4:
                    locations.append({"file": str(path), "line": index + 1})
            if (not specific.search(line) or index in covered or file_windows >= 6
                    or len(result["snippets"]) >= 48):
                continue
            start, end = max(0, index - 1), min(len(lines), index + 4)
            result["snippets"].append({"file": str(path), "matched_line": index + 1,
                                       "lines": [{"line": i + 1, "text": lines[i][:240]}
                                                 for i in range(start, end)]})
            covered.update(range(start, end))
            file_windows += 1
    result["source_bytes_read"] = total
    result["candidate_files"] = len(candidates)
    result["file_limit_reached"] = len(candidates) > 32
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file in an existing directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("output must be new and its parent directory must exist")
    report = {"schema_version": 1, "created_at_utc": stamp(), "status": "running", "stage": "metadata",
              "scope": "metadata, bounded isolated imports/help, selected package sources; no model load/install",
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {}, "known_switches": {name: os.environ.get(name) for name in SWITCHES},
              "imports": [], "official_framework_service": False, "official_accuracy": False}
    code = 0
    with output.open("x", encoding="utf-8") as handle:
        def persist() -> None:
            handle.seek(0)
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.truncate()
            handle.flush()

        try:
            for name in PACKAGES:
                try:
                    report["packages"][name] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    report["packages"][name] = None
            persist()
            report["stage"] = "source_probe"
            report["source"] = source_probe()
            persist()
            report["stage"] = "isolated_imports"
            for name in ("torch", "torchvision", "flash_attn", "vllm"):
                report["active_import"] = name
                persist()
                report["imports"].append(import_probe(name))
                persist()
            report.pop("active_import", None)
            report["stage"] = "api_server_help"
            persist()
            backend_names = ("FLASH_ATTN_CUSTOM", "TORCH_SDPA", "TRITON_ATTN")
            cli = run_bounded([sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server", "--help"],
                              60, patterns=CLI_FLAGS + backend_names)
            mentions = cli.get("stdout", {}).get("stream_matches", {})
            cli["supports"] = {flag: flag in mentions for flag in CLI_FLAGS}
            cli["backend_names_mentioned"] = {name: name in mentions for name in backend_names}
            cli["help_succeeded"] = cli.get("returncode") == 0 and not cli["timed_out"]
            cli["support_caveat"] = "flag mentions scanned across the full drained stdout; verify choices in local source"
            report["api_server_help"] = cli
            report["checks_passed"] = all(item["passed"] for item in report["imports"]) and cli["help_succeeded"]
            report["status"] = "completed" if report["checks_passed"] else "completed_with_issues"
            report["stage"] = "complete"
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "probe_failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()[-7000:]}
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report["finished_at_utc"] = stamp()
        persist()
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
