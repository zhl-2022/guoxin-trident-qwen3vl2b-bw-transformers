#!/usr/bin/env python3
"""Explicit bounded WAV transcription diagnostic; never official acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]
BODY_LIMIT = 1024 * 1024
WAV_LIMIT = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, target):
        raise ValueError("redirects are disabled for audio diagnostics")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def preview(text: str, limit: int = 16384) -> dict:
    return {"text": text[:limit], "truncated": len(text) > limit, "characters": len(text)}


def local_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip("/"):
        raise ValueError("base-url must be an HTTP(S) loopback server root without credentials/query/fragment")
    if parsed.hostname != "localhost" and not ipaddress.ip_address(parsed.hostname).is_loopback:
        raise ValueError("audio is sent only to a loopback service")
    parsed.port
    return value.rstrip("/")


def read_wav(path: Path, *, long_audio: bool = False) -> tuple[bytes, dict]:
    limit = 4 * 1024 * 1024 if long_audio else WAV_LIMIT
    max_seconds = 120 if long_audio else 30
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError(f"WAV must be a nonempty local file at most {limit} bytes")
    with wave.open(io.BytesIO(raw), "rb") as audio:
        channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
        if (channels, width, rate, audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("first diagnostic requires mono 16000 Hz uncompressed PCM16 WAV; no automatic conversion")
        if not 0 < frames <= max_seconds * rate:
            raise ValueError(f"diagnostic requires duration greater than zero and at most {max_seconds} seconds")
        if len(audio.readframes(frames)) != frames * channels * width:
            raise ValueError("WAV sample payload is truncated")
    return raw, {
        "path": str(path.resolve()), "sha256": digest(raw), "bytes": len(raw),
        "sample_rate_hz": rate, "channels": channels, "sample_width_bytes": width,
        "sample_frames": frames, "duration_seconds": frames / rate,
        "container": "WAV", "encoding": "PCM16", "resampled_or_trimmed": False,
        "long_audio_explicitly_enabled": long_audio, "duration_limit_seconds": max_seconds,
    }


def multipart_request(model: str, wav: bytes, boundary: str | None = None) -> tuple[bytes, str]:
    if not model or len(model) > 200 or "\r" in model or "\n" in model:
        raise ValueError("invalid served model name")
    boundary = boundary or "guoxin-asr-" + uuid.uuid4().hex
    if not boundary or not all(character.isalnum() or character == "-" for character in boundary):
        raise ValueError("invalid multipart boundary")
    chunks = []
    for key, value in (("model", model), ("response_format", "json"), ("temperature", "0")):
        chunks.extend([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n".encode("ascii"),
            value.encode("utf-8"), b"\r\n",
        ])
    chunks.extend([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("ascii"),
        wav, f"\r\n--{boundary}--\r\n".encode("ascii"),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_json(opener, url: str, timeout: float, body: bytes | None = None, content_type: str = "application/json") -> dict:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(BODY_LIMIT + 1)
    except urllib.error.HTTPError as exc:
        raw = exc.read(2048)
        # Do not log bodies that might echo uploaded audio or other binary data.
        raise ValueError(f"HTTP {exc.code}; error_body_preview_bytes={len(raw)}; error_body_preview_sha256={digest(raw)}") from None
    if len(raw) > BODY_LIMIT:
        raise ValueError("HTTP JSON response exceeds 1 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("error"):
        raise ValueError("HTTP response is not a successful JSON object")
    return value


def transcript_diagnostic(response: dict, expected: str | None = None) -> dict:
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("transcription response has no nonempty text field")
    return {
        "raw_text": preview(text),
        "transcription_nonempty": True,
        "contains_asr_text_marker": "<asr_text>" in text,
        "finish_reason_available": "finish_reason" in response,
        "finish_reason": response.get("finish_reason") if isinstance(response.get("finish_reason"), str) else None,
        "exact_text_match_diagnostic": text.strip() == expected.strip() if expected is not None else None,
        "expected_text_comparison_is_official_metric": False,
        "wer_computed": False, "cer_computed": False, "semantic_accuracy_verified": False,
        "truncation_verified_absent": False,
        "note": "A nonempty text response proves only this API exchange; this route may omit finish reason and token counts.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=ROOT / "tasks/qwen3-asr-bw-vllm/task.json")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--long-audio", action="store_true", help="explicit development diagnostic, up to 120s / 4MiB PCM16")
    parser.add_argument("--expected-text", help="optional local exact-text diagnostic only; never sent to the server")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--log", type=Path, help="new JSONL file; existing files are never overwritten")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if args.expected_text is not None and len(args.expected_text) > 4096:
            raise ValueError("expected-text exceeds 4096 characters")
        if args.task.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("task config exceeds 4 MiB")
        config = json.loads(args.task.read_text(encoding="utf-8"))
        if config.get("model", {}).get("id") != "Qwen/Qwen3-ASR-1.7B":
            raise ValueError("task must identify the original Qwen/Qwen3-ASR-1.7B checkpoint")
        model = args.model or config["service_candidate"]["served_name"]
        base = local_base_url(args.base_url or f"http://127.0.0.1:{config['service_candidate']['port']}")
        wav, audio_info = read_wav(args.audio, long_audio=args.long_audio)
        payload, content_type = multipart_request(model, wav)
        run_id = uuid.uuid4().hex
        output = args.log or args.task.resolve().parent / f"asr-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{run_id[:8]}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as log:
            def record(case, **values):
                event = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "case": case, **values}
                line = json.dumps(event, ensure_ascii=False, allow_nan=False)
                log.write(line + "\n"); log.flush(); print(line, flush=True)

            record("scope", kind="long_wav_api_diagnostic" if args.long_audio else "short_wav_api_diagnostic", official_accuracy=False, official_evaluation="not_run",
                   dry_run=args.dry_run, model=model, base_url=base, audio=audio_info,
                   task_sha256=digest(args.task.read_bytes()), script_sha256=digest(Path(__file__).read_bytes()),
                   fields={"model": model, "response_format": "json", "temperature": "0"},
                   expected_text_local_only=preview(args.expected_text, 4096) if args.expected_text is not None else None,
                   language_or_prompt_sent=False, uploaded_wav_bytes_logged=False)
            if args.dry_run:
                record("summary", dry_run=True, requests_sent=0, official_accuracy=False)
                return 0
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            try:
                models = request_json(opener, base + "/v1/models", args.timeout)
                if model not in {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}:
                    raise ValueError("served model is absent from /v1/models; no audio sent")
                record("models", served_model_present=True)
                started = time.monotonic()
                response = request_json(opener, base + "/v1/audio/transcriptions", args.timeout, payload, content_type)
                result = transcript_diagnostic(response, args.expected_text)
                record("transcription", request_duration_seconds=time.monotonic() - started, **result)
                record("summary", functional_transcription_received=True, semantic_accuracy_verified=False,
                       official_accuracy=False, official_evaluation="not_run")
                return 0
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                record("error", error=preview(str(exc), 2048))
                record("summary", functional_transcription_received=False, official_accuracy=False, official_evaluation="not_run")
                return 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError, wave.Error, EOFError) as exc:
        print(json.dumps({"error": preview(str(exc), 2048), "official_accuracy": False}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
