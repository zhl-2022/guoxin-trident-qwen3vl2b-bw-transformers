#!/usr/bin/env python3
"""Functional multimodal checks only; never an official accuracy evaluation."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
import uuid


def log_payload(value):
    """Keep diagnostics bounded and strip any image data URI echoed by validation errors."""
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    raw = re.sub(r"(data:[^;\s\"]+;base64,)[A-Za-z0-9+/=]+", r"\1<redacted>", raw)
    if len(raw) > 32768:
        return {"preview": raw[:32768], "truncated": True, "original_characters": len(raw)}
    return raw if isinstance(value, str) else json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True,
                        help="repeat for at least two distinct local image files")
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{os.environ.get('PORT', '30000')}")
    parser.add_argument("--model", default=os.environ.get("SERVED_NAME", "recon2reason-reasoning-4b"))
    parser.add_argument("--log", type=Path, default=Path("smoke.log"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_tokens <= 0:
        parser.error("timeout and max-tokens must be positive")
    resolved = [path.resolve() for path in args.image]
    if len(set(resolved)) < 2:
        parser.error("at least two distinct --image paths are required for the multi-image check")
    images = []
    sources = []
    for path in resolved:
        if not path.is_file():
            parser.error(f"image does not exist: {path}")
        if path.stat().st_size > 20 * 1024 * 1024:
            parser.error(f"image exceeds 20 MiB: {path}")
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            parser.error(f"unsupported image type: {path}")
        raw = path.read_bytes()
        if not raw:
            parser.error(f"empty image: {path}")
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}})
    if len({source["sha256"] for source in sources}) < 2:
        parser.error("provide at least two images with different contents")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    base = args.base_url.rstrip("/")
    results = []
    with args.log.open("a", encoding="utf-8") as log:
        def record(case, **values):
            event = {"run_id": run_id, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "case": case, **values}
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(event, ensure_ascii=False), flush=True)

        record("scope", official_accuracy=False, model=args.model, images=sources,
               description="Checks transport and nonempty output; answer correctness is not scored.")

        def get(path):
            with opener.open(base + path, timeout=args.timeout) as response:
                body = response.read().decode("utf-8")
                return {"status": response.status, "body": body}

        def chat(case, messages, stream=False):
            started = time.monotonic()
            body = {"model": args.model, "messages": messages, "temperature": 0,
                    "max_tokens": args.max_tokens, "stream": stream}
            request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode("utf-8"),
                                             {"Content-Type": "application/json"})
            with opener.open(request, timeout=args.timeout) as response:
                if stream:
                    events = []
                    fragments = []
                    reasoning_fragments = []
                    stream_errors = []
                    finish_reasons = []
                    done = False
                    for raw_line in response:
                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                            break
                        event = json.loads(payload)
                        events.append(event)
                        if event.get("error") is not None:
                            stream_errors.append(event["error"])
                        for choice in event.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if isinstance(content, str):
                                fragments.append(content)
                            # Upstream vLLM 0.18 uses reasoning; older DAS forks may use reasoning_content.
                            for field in ("reasoning", "reasoning_content"):
                                if isinstance(delta.get(field), str):
                                    reasoning_fragments.append(delta[field])
                            if choice.get("finish_reason") is not None:
                                finish_reasons.append(choice["finish_reason"])
                    text = "".join(fragments).strip()
                    diagnostics = {"done": done, "events": log_payload(events), "text": log_payload(text),
                                   "reasoning": log_payload("".join(reasoning_fragments)),
                                   "finish_reasons": finish_reasons, "answer_truncated": "length" in finish_reasons,
                                   "stream_errors": log_payload(stream_errors)}
                    error = None
                    if stream_errors or "error" in finish_reasons:
                        error = "stream reported a generation error"
                    elif not done or not text:
                        error = "stream requires nonempty final content and data: [DONE]; reasoning alone is not an answer"
                    if error:
                        record(case + "_response", passed=False, duration=time.monotonic() - started, **diagnostics)
                        raise ValueError(error)
                    record(case, passed=True, duration=time.monotonic() - started, done=done,
                           **{key: value for key, value in diagnostics.items() if key != "done"})
                else:
                    result = json.load(response)
                    choices = result.get("choices", [])
                    text = choices[0].get("message", {}).get("content") if choices else None
                    finish_reason = choices[0].get("finish_reason") if choices else None
                    diagnostics = {"response": log_payload(result), "finish_reason": finish_reason,
                                   "answer_truncated": finish_reason == "length"}
                    if result.get("error") is not None or finish_reason == "error":
                        record(case + "_response", passed=False, duration=time.monotonic() - started, **diagnostics)
                        raise ValueError("nonstreaming response reported a generation error")
                    if not isinstance(text, str) or not text.strip():
                        record(case + "_response", passed=False, duration=time.monotonic() - started, **diagnostics)
                        raise ValueError("nonstreaming response has no nonempty assistant content")
                    record(case, passed=True, duration=time.monotonic() - started, **diagnostics)
            return text

        def check(name, callback):
            try:
                value = callback()
                if name in {"health", "models"}:
                    record(name, passed=True, response=value)
                results.append({"case": name, "passed": True})
                return value
            except Exception as exc:
                diagnostics = {}
                if isinstance(exc, urllib.error.HTTPError):
                    raw = exc.read(65537)
                    diagnostics = {"http_status": exc.code, "response_body": log_payload(raw.decode("utf-8", "replace")),
                                   "http_body_read_truncated": len(raw) > 65536}
                record(name, passed=False, error=f"{type(exc).__name__}: {exc}", **diagnostics)
                results.append({"case": name, "passed": False})
                return None

        check("health", lambda: get("/health"))

        def models():
            result = get("/v1/models")
            parsed = json.loads(result["body"])
            if args.model not in {item.get("id") for item in parsed.get("data", [])}:
                raise ValueError("served model is absent from /v1/models")
            return result

        check("models", models)
        single = [{"role": "user", "content": [images[0], {"type": "text", "text": "请用一句话描述图像中的主要物体。"}]}]
        check("single_image", lambda: chat("single_image", single))
        multiple = [{"role": "user", "content": [*images, {"type": "text", "text": "请简要描述这些图像中的一个相同点和一个不同点。"}]}]
        check("multiple_images", lambda: chat("multiple_images", multiple))
        first = check("multiturn_first", lambda: chat("multiturn_first", single))
        if first is not None:
            dialogue = [*single, {"role": "assistant", "content": first},
                        {"role": "user", "content": "根据刚才的图像描述，选一个主要物体再描述它的位置。"}]
            check("multiturn_followup", lambda: chat("multiturn_followup", dialogue))
        else:
            results.append({"case": "multiturn_followup", "passed": False})
            record("multiturn_followup", passed=False, error="first turn failed")
        check("streaming_image", lambda: chat("streaming_image", single, stream=True))
        passed = all(item["passed"] for item in results)
        record("summary", functional_checks_passed=passed, official_accuracy=False, results=results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
