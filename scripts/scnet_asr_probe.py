#!/usr/bin/env python3
"""Read-only ASR source/codec probe; no model loads, downloads or installs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import sys
import traceback

sys.dont_write_bytecode = True
from scnet_vllm_probe import run_bounded


PACKAGES = ("numpy", "scipy", "soundfile", "librosa", "torchaudio", "python-multipart", "qwen-asr")
SOURCES = (
    "model_executor/models/registry.py",
    "model_executor/models/qwen3_asr.py",
    "transformers_utils/configs/qwen3_asr.py",
    "transformers_utils/processors/qwen3_asr.py",
)
DECLARATIONS = re.compile(r"Qwen3ASRForConditionalGeneration|class Qwen3ASR(?:Config|ThinkerConfig|Processor)"
                          r"|SupportsTranscription|def get_generation_prompt|def get_speech_to_text_config"
                          r"|def post_process_output|sampling_rate.*16000")
MARKER = "SCNET_ASR_JSON="
CHILD_PROGRAM = r'''
import importlib
import io
import json
from pathlib import Path
import sys
import traceback
import wave
sys.dont_write_bytecode = True
name, audio_path = sys.argv[1:3]
result = {"module": name, "ok": False}
try:
    module = importlib.import_module(name)
    result.update(version=str(getattr(module, "__version__", "unknown")), file=getattr(module, "__file__", None))
    if name == "soundfile":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 1600)
        waveform, sr = module.read(io.BytesIO(buffer.getvalue()), dtype="float32")
        if sr != 16000 or waveform.shape != (1600,):
            raise ValueError("In-memory mono 16k PCM WAV decode failed")
        result["synthetic_codec_check"] = {"sample_rate": sr, "samples": len(waveform),
            "dtype": str(waveform.dtype), "asr_accuracy_evidence": False}
        if audio_path:
            path = Path(audio_path)
            if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
                raise ValueError("Optional audio must be an existing nonempty file <=20 MiB")
            info = module.info(str(path))
            if (info.samplerate != 16000 or info.channels != 1 or info.duration > 30
                    or info.frames <= 0 or info.format != "WAV" or info.subtype != "PCM_16"):
                raise ValueError("Optional audio must be mono 16k PCM_16 WAV, nonempty and <=30 seconds; no conversion applied")
            waveform, sr = module.read(str(path), dtype="float32")
            import numpy as np
            if not bool(np.isfinite(waveform).all()):
                raise ValueError("Optional audio has non-finite decoded samples")
            result["audio_read"] = {"path": str(path), "sample_rate": sr, "samples": len(waveform),
                "channels": info.channels, "duration_seconds": info.duration,
                "subtype": info.subtype, "dtype": str(waveform.dtype),
                "peak_abs": float(np.max(np.abs(waveform))), "asr_accuracy_evidence": False}
    result["ok"] = True
except BaseException as exc:
    result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                       "traceback": traceback.format_exc()[-5500:]}
print("SCNET_ASR_JSON=" + json.dumps(result, ensure_ascii=False, default=str), flush=True)
raise SystemExit(0 if result["ok"] else 1)
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_checks() -> dict:
    result = {"files": [], "registry_entry_found": False}
    try:
        distribution = metadata.distribution("vllm")
    except metadata.PackageNotFoundError:
        result["error"] = "vllm distribution is missing"
        return result
    root = Path(distribution.locate_file("vllm")).resolve()
    result["vllm_root"] = str(root)
    for relative in SOURCES:
        path = (root / relative).resolve()
        item = {"file": str(path), "exists": path.is_file()}
        result["files"].append(item)
        if not path.is_relative_to(root):
            item["error"] = "Source path resolves outside the vllm package"
            continue
        if not item["exists"]:
            continue
        size = path.stat().st_size
        item["bytes"] = size
        if size > 262144:
            item["error"] = "Source exceeds 256 KiB read limit"
            continue
        item["sha256"] = sha256(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        item["declarations"] = [{"line": index + 1, "text": line[:240]}
                                for index, line in enumerate(lines) if DECLARATIONS.search(line)][:16]
        if relative.endswith("registry.py"):
            result["registry_entry_found"] = any(
                '"Qwen3ASRForConditionalGeneration"' in line for line in lines)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON file in an existing directory")
    parser.add_argument("--audio", type=Path, help="optional local mono 16k PCM_16 WAV, <=30s and <=20 MiB")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("output must be new and its parent directory must exist")
    audio = args.audio.resolve() if args.audio else None
    report = {"schema_version": 1, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "running", "stage": "metadata", "python": sys.version,
              "executable": sys.executable, "packages": {}, "ffmpeg": shutil.which("ffmpeg"),
              "imports": [], "audio_requested": str(audio) if audio else None,
              "model_loaded": False, "official_accuracy": False,
              "scope": "T04 environment/source/codec checks; no ASR inference or official dataset required"}
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
            report["stage"] = "source_checks"
            report["source"] = source_checks()
            persist()
            report["stage"] = "isolated_imports"
            for name in ("soundfile", "librosa", "torchaudio", "multipart"):
                report["active_import"] = name
                persist()
                check = run_bounded([sys.executable, "-B", "-c", CHILD_PROGRAM, name,
                                     str(audio) if audio else ""], 30)
                check.update(module=name, data=None)
                for line in check.get("stdout", {}).get("text", "").splitlines():
                    if line.startswith(MARKER):
                        try:
                            check["data"] = json.loads(line[len(MARKER):])
                        except json.JSONDecodeError:
                            check["structured_output_error"] = "invalid/truncated child JSON"
                check["passed"] = bool(check["data"] and check["data"].get("ok")
                                       and check.get("returncode") == 0 and not check["timed_out"])
                report["imports"].append(check)
                persist()
            report.pop("active_import", None)
            if audio and audio.is_file() and audio.stat().st_size <= 20 * 1024 * 1024:
                report["audio_sha256"] = sha256(audio)
            files = report["source"].get("files", [])
            report["checks_passed"] = bool(report["source"].get("registry_entry_found")
                                           and len(files) == len(SOURCES)
                                           and all(item.get("sha256") for item in files)
                                           and all(item["passed"] for item in report["imports"]))
            report["status"] = "completed" if report["checks_passed"] else "completed_with_issues"
            report["stage"] = "complete"
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "probe_failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()[-6000:]}
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        persist()
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
