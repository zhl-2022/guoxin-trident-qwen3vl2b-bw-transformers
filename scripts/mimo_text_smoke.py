#!/usr/bin/env python3
"""Bounded text API smoke; preserve checkpoint thinking/sampling, never score accuracy."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
import math
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

MAX_BYTES = 8 * 1024 * 1024
MAX_LINE = 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, target):
        raise ValueError("Smoke redirects are disabled")


def base_url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip("/"):
        raise ValueError("Base URL must be a credential-free loopback HTTP(S) server root")
    if parsed.hostname != "localhost" and not ipaddress.ip_address(parsed.hostname).is_loopback:
        raise ValueError("Smoke only sends to loopback")
    parsed.port
    return value.rstrip("/")


def usage_check(value):
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] <= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError("Missing or invalid raw token usage")
    if value["prompt_tokens"] + value["completion_tokens"] != value["total_tokens"]:
        raise ValueError("Raw token accounting is inconsistent")
    return value


def response_check(value, model):
    if not isinstance(value, dict) or value.get("error") is not None or value.get("model") != model:
        raise ValueError("Response error or model mismatch")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Require exactly one response choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Response truncated or not completed with stop")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
        raise ValueError("Nonempty visible content required; reasoning alone is not an answer")
    usage_check(value.get("usage"))
    return message["content"]


def read_body(response, timeout):
    """A peer sending occasional bytes must not reset the whole-body deadline."""
    deadline = time.monotonic() + timeout
    expired = threading.Event()
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if not isinstance(sock, socket.socket):
        sock = None
        if not isinstance(response, io.BytesIO):
            raise ValueError("HTTP response lacks a deadline-controllable socket")

    def expire():
        expired.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(timeout, expire) if sock is not None else None
    if timer:
        timer.daemon = True
        timer.start()
    try:
        raw = response.read(MAX_BYTES + 1)
        if expired.is_set() or time.monotonic() >= deadline:
            raise ValueError("HTTP body absolute wall timeout")
        return raw
    finally:
        if timer:
            timer.cancel()
            timer.join()


def consume_stream(response, model, timeout, record):
    started = time.monotonic()
    deadline = started + timeout
    expired = threading.Event()
    stream_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if not isinstance(stream_socket, socket.socket):
        stream_socket = None
        if not isinstance(response, io.BytesIO):
            raise ValueError("SSE response lacks a deadline-controllable socket")

    def check():
        remaining = deadline - time.monotonic()
        if remaining <= 0 or expired.is_set():
            raise ValueError("SSE absolute wall timeout")
        return remaining

    def expire():
        expired.set()
        try:
            stream_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(check(), expire) if stream_socket is not None else None
    if timer:
        timer.daemon = True
        timer.start()
    fragments, data = [], []
    total, events = 0, 0
    done = finished = False
    usage = None
    try:
        while not done:
            remaining = check()
            if stream_socket is not None:
                stream_socket.settimeout(remaining)
            try:
                line = response.readline(MAX_LINE + 1)
            except OSError:
                check()
                raise
            check()
            if not line:
                break
            total += len(line)
            if len(line) > MAX_LINE or total > MAX_BYTES:
                raise ValueError("SSE byte cap exceeded")
            decoded = line.decode("utf-8").rstrip("\r\n")
            if decoded.startswith("data:"):
                data.append(decoded[5:].removeprefix(" "))
                continue
            if decoded.startswith(":") or decoded.startswith(("event:", "id:", "retry:")):
                continue
            if decoded:
                raise ValueError("Malformed SSE framing")
            if not data:
                continue
            raw = "\n".join(data)
            data.clear()
            record("stream_raw_event", raw_data=raw)
            if raw == "[DONE]":
                done = True
                continue
            event = json.loads(raw)
            if not isinstance(event, dict) or event.get("error") is not None or event.get("type") == "error":
                raise ValueError("SSE reported an error")
            if event.get("model") not in (None, model):
                raise ValueError("SSE model mismatch")
            choices = event.get("choices")
            if not isinstance(choices, list) or len(choices) > 1 or (finished and choices):
                raise ValueError("SSE choices malformed or received after terminal finish")
            events += 1
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0 or not isinstance(choice.get("delta"), dict):
                    raise ValueError("Malformed SSE choice")
                text = choice["delta"].get("content")
                if text is not None and not isinstance(text, str):
                    raise ValueError("SSE content must be text")
                if text:
                    fragments.append(text)
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason != "stop":
                        raise ValueError("SSE truncated or finish reason not stop")
                    finished = True
            if event.get("usage") is not None:
                if usage is not None or not finished:
                    raise ValueError("Duplicate or premature SSE usage")
                usage = usage_check(event["usage"])
        check()
        if data or not done or not finished or not "".join(fragments).strip():
            raise ValueError("SSE missing framed DONE, stop, or nonempty visible content")
        usage_check(usage)
        return {"raw_text": "".join(fragments), "finish_reason": "stop", "done": True,
                "raw_usage": usage, "events": events, "response_bytes": total,
                "wall_seconds": time.monotonic() - started}
    finally:
        if timer:
            timer.cancel()
            timer.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30109")
    parser.add_argument("--model", default="mimo-v2.6-distill-qwen-9b")
    parser.add_argument("--log", type=Path, default=Path("smoke.log"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0 or not 1 <= args.max_tokens <= 8192:
            raise ValueError("Require finite positive timeout and max-tokens1..8192")
        if not args.model or len(args.model) > 250 or any(c in args.model for c in "\r\n"):
            raise ValueError("Invalid served model name")
        base = base_url(args.base_url)
        scope = {"model": args.model, "base_url": base, "max_tokens": args.max_tokens,
                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "temperature_top_p_top_k_or_thinking_override_sent": False,
                 "reasoning_parser_applied": False, "official_accuracy": False, "official_performance": False,
                 "dry_run": args.dry_run, "planned_checks": ["health", "models", "single", "multiturn_first", "multiturn_followup", "stream"]}
        if args.dry_run:
            print(json.dumps(scope, ensure_ascii=False, indent=2))
            return 0
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        run_id = uuid.uuid4().hex
        # Exclusive log preserves earlier runs and raw failures; --log chooses a fresh path.
        with args.log.open("x", encoding="utf-8", newline="\n") as log:
            def record(case, **values):
                line = json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                                   "case": case, **values}, ensure_ascii=False, allow_nan=False)
                log.write(line + "\n"); log.flush(); print(line, flush=True)

            def get(path):
                with opener.open(base + path, timeout=args.timeout) as response:
                    raw = read_body(response, args.timeout)
                    record("http_raw", path=path, status=response.status, raw_body=raw.decode("utf-8", "replace"))
                    if len(raw) > MAX_BYTES:
                        raise ValueError("HTTP response byte cap")
                    return raw

            def chat(case, messages, stream=False):
                body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens, "stream": stream}
                if stream:
                    body["stream_options"] = {"include_usage": True}
                record(case + "_request", request=body)
                request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode("utf-8"),
                                                 {"Content-Type": "application/json"})
                with opener.open(request, timeout=args.timeout) as response:
                    if stream:
                        if response.headers.get_content_type() != "text/event-stream":
                            raise ValueError("Expected text/event-stream")
                        result = consume_stream(response, args.model, args.timeout, record)
                        record(case, passed=True, **result)
                        return result["raw_text"]
                    raw = read_body(response, args.timeout)
                    record(case + "_raw_response", raw_body=raw.decode("utf-8", "replace"), status=response.status)
                    if len(raw) > MAX_BYTES:
                        raise ValueError("HTTP response byte cap")
                    value = json.loads(raw)
                    text = response_check(value, args.model)
                    record(case, passed=True, raw_response=value)
                    return text

            record("scope", **scope)
            try:
                get("/health")
                record("health", passed=True)
                models = json.loads(get("/v1/models"))
                if not isinstance(models, dict) or not isinstance(models.get("data"), list):
                    raise ValueError("Malformed models JSON object")
                if args.model not in {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}:
                    raise ValueError("Served model absent from models list")
                record("models", passed=True)
                single = [{"role": "user", "content": "Please reply with a short greeting."}]
                chat("single", single)
                first = [{"role": "user", "content": "For this conversation, my name is Xiaohai. Please acknowledge briefly."}]
                answer = chat("multiturn_first", first)
                chat("multiturn_followup", [*first, {"role": "assistant", "content": answer},
                                           {"role": "user", "content": "What name did I ask you to remember?"}])
                chat("stream", single, stream=True)
                record("summary", functional_checks_passed=True, official_accuracy=False, official_performance=False,
                       answer_correctness_scored=False)
                return 0
            except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    record("http_error_raw", status=exc.code, raw_body=exc.read(MAX_BYTES + 1).decode("utf-8", "replace"))
                record("summary", functional_checks_passed=False, error=f"{type(exc).__name__}: {exc}",
                       official_accuracy=False, official_performance=False)
                return 2
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "official_accuracy": False}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
