#!/usr/bin/env python3
"""Explicit local WAV-to-SSE output diagnostic; not real-time audio input or official accuracy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

from qwen3_asr_smoke import (NoRedirect, digest, local_base_url, multipart_request,
                             preview, read_wav, request_json)

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE_BYTES = 65536
MAX_STREAM_BYTES = 1024 * 1024
MAX_TEXT_CHARACTERS = 16384
SOURCE = "https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/entrypoints/openai/speech_to_text/speech_to_text.py"


def streaming_multipart(model, wav):
    boundary = "guoxin-asr-stream-" + uuid.uuid4().hex
    body, content_type = multipart_request(model, wav, boundary)
    ending = f"--{boundary}--\r\n".encode("ascii")
    if not body.endswith(ending):
        raise ValueError("unexpected multipart envelope")
    extra = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode("ascii")
                     for key, value in (("stream", "true"), ("stream_include_usage", "true")))
    return body[:-len(ending)] + extra + ending, content_type


def consume_stream(response, model, timeout, record):
    """Bound SSE lines/events; fail on errors, missing DONE, truncation or empty text."""
    start = time.monotonic()
    deadline = start + timeout
    expired = threading.Event()
    stream_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if not isinstance(stream_socket, socket.socket):
        stream_socket = None
        if not isinstance(response, io.BytesIO):
            raise ValueError("stream response lacks a deadline-controllable socket")

    def check_deadline():
        remaining = deadline - time.monotonic()
        if remaining <= 0 or expired.is_set():
            raise ValueError("stream wall timeout")
        return remaining

    def expire():
        expired.set()
        try:
            stream_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def read_line():
        remaining = check_deadline()
        if stream_socket is not None:
            stream_socket.settimeout(remaining)
        try:
            line = response.readline(MAX_LINE_BYTES + 1)
        except OSError:
            check_deadline()
            raise
        check_deadline()
        return line

    # A socket timeout alone resets on each recv within readline (including HTTP
    # chunk framing). Shutdown at the absolute deadline also bounds slow trickles.
    timer = threading.Timer(check_deadline(), expire) if stream_socket is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        result = _consume_stream(read_line, model, start, check_deadline, record)
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
    check_deadline()
    return result


def _consume_stream(read_line, model, start, check_deadline, record):
    parts, data_lines = [], []
    total_bytes = events = 0
    done = False
    reasons = []
    first_delta_seconds = None
    usage = None
    finished = False

    def event():
        nonlocal done, events, first_delta_seconds, usage, finished
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        data_lines.clear()
        if raw == "[DONE]":
            done = True
            record("stream_done", done_marker_received=True)
            return
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("SSE data must be a JSON object")
        if value.get("error") or value.get("type") == "error":
            raise ValueError("SSE error event received; payload_sha256=" + digest(raw.encode("utf-8")))
        if value.get("model") not in (None, model):
            raise ValueError("SSE response identifies another model")
        choices = value.get("choices")
        if not isinstance(choices, list):
            raise ValueError("SSE transcription event requires choices array")
        if len(choices) > 1:
            raise ValueError("diagnostic supports one transcript only")
        if finished and choices:
            raise ValueError("SSE choice received after terminal finish")
        events += 1
        delta_text = ""
        reason = None
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                raise ValueError("malformed transcription choice")
            delta_text = choice["delta"].get("content")
            if delta_text is None:
                delta_text = ""
            if not isinstance(delta_text, str):
                raise ValueError("SSE content must be text")
            if delta_text and first_delta_seconds is None:
                first_delta_seconds = time.monotonic() - start
            parts.append(delta_text)
            if sum(map(len, parts)) > MAX_TEXT_CHARACTERS:
                raise ValueError("stream text exceeds diagnostic character cap")
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str):
                    raise ValueError("finish reason must be text")
                reasons.append(reason)
                finished = True
        if value.get("usage") is not None:
            if not isinstance(value["usage"], dict):
                raise ValueError("usage must be an object")
            usage = {key: value["usage"][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                     if key in value["usage"]}
        record("stream_event", event_number=events, delta_text=preview(delta_text),
               finish_reason=reason, usage=usage)

    while not done:
        line = read_line()
        if not line:
            break
        total_bytes += len(line)
        if len(line) > MAX_LINE_BYTES or total_bytes > MAX_STREAM_BYTES:
            raise ValueError("SSE response exceeds diagnostic byte cap")
        decoded = line.decode("utf-8", errors="strict").rstrip("\r\n")
        if not decoded:
            event()
        elif decoded.startswith("data:"):
            data_lines.append(decoded[5:].removeprefix(" "))
        elif decoded.startswith(":") or decoded.startswith(("event:", "id:", "retry:")):
            continue
        else:
            raise ValueError("response is not valid SSE framing")
    if data_lines:
        event()
    check_deadline()
    text = "".join(parts)
    if not done:
        raise ValueError("SSE stream ended without [DONE]")
    if not text.strip():
        raise ValueError("SSE stream contains no transcript text")
    transcript_body = text.partition("<asr_text>")[2] if "<asr_text>" in text else text
    if not transcript_body.strip():
        raise ValueError("SSE stream contains no transcript text after ASR prefix")
    if not reasons or any(reason != "stop" for reason in reasons):
        raise ValueError("SSE finish reason missing or not stop: " + repr(reasons))
    return {"raw_text": preview(text), "finish_reasons": reasons,
            "done_marker_received": True, "events": events, "response_bytes": total_bytes,
            "usage": usage, "first_nonempty_delta_seconds": first_delta_seconds,
            "stream_duration_seconds": time.monotonic() - start,
            "contains_asr_text_marker": "<asr_text>" in text,
            "asr_prefix_stripped": False, "transcript_body_nonempty": True,
            "streaming_transcript_received": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--long-audio", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:30002")
    parser.add_argument("--model", default="qwen3-asr-1.7b")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--log", type=Path, help="new JSONL file; required for --run")
    parser.add_argument("--run", action="store_true", help="explicitly send one full WAV to loopback; default no network or writes")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive finite seconds")
        base = local_base_url(args.base_url)
        wav, audio_info = read_wav(args.audio, long_audio=args.long_audio)
        payload, content_type = streaming_multipart(args.model, wav)
        plan = {"schema_version": 1, "kind": "full_wav_sse_output_development_diagnostic",
                "official_accuracy": False, "official_performance": False,
                "realtime_audio_input_verified": False, "audio_uploaded_as_one_file": True,
                "source_reference": SOURCE, "script_sha256": digest(Path(__file__).read_bytes()),
                "audio": audio_info, "model": args.model, "base_url": base,
                "fields": {"stream": True, "stream_include_usage": True,
                           "temperature": 0, "response_format": "json"},
                "language_or_reference_prompt_sent": False,
                "run": args.run, "network_requests": 0}
        if not args.run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if not args.log:
            raise ValueError("--run requires --log naming a new evidence file")
        with args.log.open("x", encoding="utf-8", newline="\n") as log:
            def record(case, **values):
                line = json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "case": case, **values},
                                  ensure_ascii=False, allow_nan=False)
                log.write(line + "\n"); log.flush(); print(line, flush=True)

            record("scope", **plan)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            try:
                models = request_json(opener, base + "/v1/models", args.timeout)
                if args.model not in {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}:
                    raise ValueError("served model absent; no audio sent")
                request = urllib.request.Request(base + "/v1/audio/transcriptions", data=payload,
                                                 headers={"Content-Type": content_type, "Accept": "text/event-stream"})
                with opener.open(request, timeout=args.timeout) as response:
                    if response.headers.get_content_type() != "text/event-stream":
                        raise ValueError("server did not return text/event-stream")
                    result = consume_stream(response, args.model, args.timeout, record)
                record("summary", **result, official_accuracy=False, official_performance=False,
                       realtime_audio_input_verified=False, requests_sent=2)
                return 0
            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                message = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc)
                record("error", error=preview(message, 2048), streaming_transcript_received=False,
                       official_accuracy=False, official_performance=False)
                return 2
    except (OSError, ValueError, TypeError, EOFError, wave.Error) as exc:
        print(json.dumps({"error": str(exc), "official_accuracy": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
