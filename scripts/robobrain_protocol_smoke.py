#!/usr/bin/env python3
"""Single-image protocol diagnostics only; no semantic accuracy or official acceptance."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import mimetypes
from pathlib import Path
import re
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("general", "grounding", "pointing", "trajectory")
BODY_LIMIT = 1024 * 1024
TEXT_LIMIT = 16384
PATTERNS = {
    "grounding": r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
    "pointing": r"\(\s*\d+\s*,\s*\d+\s*\)",
    "trajectory": r"\d+,\s*\d+,\s*[+-]?\d+\.\d+",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, target):
        raise ValueError("HTTP redirects are disabled for image diagnostics")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bounded_text(value: str, limit: int = TEXT_LIMIT) -> dict:
    value = re.sub(r"data:[^;\s\"]+;base64,[A-Za-z0-9+/=]+", "<image-data-redacted>", value)
    return {"text": value[:limit], "truncated": len(value) > limit, "characters": len(value)}


def local_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base-url must be an HTTP(S) loopback URL without credentials/query/fragment")
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("this diagnostic sends images only to a loopback service")
    if parsed.path.rstrip("/"):
        raise ValueError("base-url must be the server root without /v1")
    parsed.port  # Reject invalid or out-of-range ports before any HTTP request.
    return value.rstrip("/")


def load_configuration(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("task config exceeds 4 MiB")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model", {}).get("id") != "BAAI/RoboBrain2.5-8B-NV":
        raise ValueError("task config must identify BAAI/RoboBrain2.5-8B-NV")
    templates = config["upstream_protocol"]["prompt_templates"]
    for task in TASKS:
        template = templates[task]
        if not isinstance(template, str) or len(template) > 16384:
            raise ValueError("invalid prompt template")
        fields = [(field, spec, conversion) for _, field, spec, conversion in string.Formatter().parse(template) if field is not None]
        if fields != [("text", "", None)]:
            raise ValueError("each template must contain exactly one plain {text} field")
    return config


def image_payload(path: Path) -> tuple[dict, dict]:
    if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
        raise ValueError("image must be a nonempty local file at most 20 MiB")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("use one PNG, JPEG or WebP image")
    raw = path.read_bytes()
    return (
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"}},
        {"path": str(path.resolve()), "sha256": sha256(raw), "bytes": len(raw), "mime": mime},
    )


def build_request(config: dict, task: str, text: str, image: dict, model: str, max_tokens: int) -> dict:
    if task not in TASKS or not text.strip() or len(text) > 4096:
        raise ValueError("task text must be nonempty and at most 4096 characters")
    if max_tokens not in {128, 256}:
        raise ValueError("diagnostic max_tokens must be 128 or 256")
    prompt = config["upstream_protocol"]["prompt_templates"][task].format(text=text)
    return {
        "model": model,
        "messages": [{"role": "user", "content": [image, {"type": "text", "text": prompt}]}],
        "temperature": 0, "max_tokens": max_tokens, "stream": False,
    }


def response_diagnostics(task: str, response: dict) -> dict:
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("server returned an error or a non-object response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response message is missing")
    answer = message.get("content")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("assistant content must be a string or null")
    answer = answer or ""
    reasoning = message.get("reasoning_content", message.get("reasoning", "")) or ""
    if not isinstance(reasoning, str):
        reasoning = "<non-string reasoning omitted>"
    syntax = bool(re.search(PATTERNS[task], answer)) if task in PATTERNS else None
    finished = choice.get("finish_reason") == "stop"
    return {
        "raw_answer": bounded_text(answer),
        "raw_reasoning": bounded_text(reasoning),
        "response_nonempty": bool(answer.strip()),
        "reasoning_only": bool(reasoning.strip()) and not bool(answer.strip()),
        "finish_reason": choice.get("finish_reason"),
        "token_limit_reached": choice.get("finish_reason") == "length",
        "generation_finished": finished,
        "syntax_pattern_observed": syntax,
        "coordinate_range_checked": False,
        "semantic_accuracy_verified": False,
        "functional_protocol_check_passed": bool(answer.strip()) and finished and syntax is not False,
    }


def read_body(response) -> bytes:
    raw = response.read(BODY_LIMIT + 1)
    if len(raw) > BODY_LIMIT:
        raise ValueError("HTTP response exceeds 1 MiB; refusing unbounded diagnostics")
    return raw


def request_json(opener, url: str, timeout: float, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(read_body(response))
    except urllib.error.HTTPError as exc:
        preview = bounded_text(exc.read(TEXT_LIMIT).decode("utf-8", errors="replace"), 2048)
        raise ValueError(f"HTTP {exc.code}: {preview['text']}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=ROOT / "tasks/robobrain-bw-vllm/task.json")
    parser.add_argument("--image", required=True, type=Path, help="exactly one image, reused for four diagnostic requests")
    parser.add_argument("--target", required=True, help="visible object/region description for grounding and pointing")
    parser.add_argument("--trajectory-instruction", required=True, help="image-related manipulation description; text diagnostic only")
    parser.add_argument("--general-text", default="Describe the visible objects and their relative positions.")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--log", type=Path, help="new JSONL file; existing files are never overwritten")
    parser.add_argument("--max-tokens", type=int, choices=(128, 256), default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true", help="write request metadata without making HTTP calls")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        config = load_configuration(args.task)
        model = args.model or config["service_candidate"]["served_name"]
        base = local_base_url(args.base_url or f"http://127.0.0.1:{config['service_candidate']['port']}")
        image, image_info = image_payload(args.image)
        texts = {"general": args.general_text, "grounding": args.target,
                 "pointing": args.target, "trajectory": args.trajectory_instruction}
        bodies = {task: build_request(config, task, texts[task], image, model, args.max_tokens) for task in TASKS}
        run_id = uuid.uuid4().hex
        output = args.log or args.task.resolve().parent / f"protocol-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{run_id[:8]}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as log:
            def record(case: str, **values):
                event = {"run_id": run_id, "utc": datetime.now(timezone.utc).isoformat(), "case": case, **values}
                log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

            record("scope", kind="single_image_functional_protocol_diagnostic", official_accuracy=False,
                   semantic_accuracy_verified=False, official_evaluation="not_run", dry_run=args.dry_run,
                   model=model, base_url=base, image=image_info, log=str(output),
                   script_sha256=sha256(Path(__file__).read_bytes()), task_sha256=sha256(args.task.read_bytes()),
                   prompt_source_sha256=config["upstream"].get("inference_source_sha256"),
                   generation={"temperature": 0, "max_tokens": args.max_tokens, "stream": False},
                   note="Syntax hints do not establish valid coordinates, depth, task correctness or official agreement.")
            if args.dry_run:
                for task, body in bodies.items():
                    record(task, dry_run=True, prompt=body["messages"][0]["content"][1]["text"], image_count=1)
                return 0
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            try:
                models = request_json(opener, base + "/v1/models", args.timeout)
                if model not in {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}:
                    raise ValueError("served model identity is missing from /v1/models; no image request sent")
                record("models", passed=True, served_model_present=True)
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                record("models", passed=False, error=bounded_text(str(exc), 2048))
                record("summary", functional_protocol_checks_passed=False, official_accuracy=False, requests_sent=0)
                return 2
            passed = True
            sent = 0
            for task, body in bodies.items():
                started = time.monotonic()
                sent += 1
                try:
                    response = request_json(opener, base + "/v1/chat/completions", args.timeout, body)
                    result = response_diagnostics(task, response)
                    record(task, prompt=body["messages"][0]["content"][1]["text"],
                           request_duration_seconds=time.monotonic() - started, image_count=1, **result)
                    passed = passed and result["functional_protocol_check_passed"]
                except (ValueError, OSError, TypeError, AttributeError) as exc:
                    record(task, functional_protocol_check_passed=False, error=bounded_text(str(exc), 2048))
                    passed = False
                    break  # Avoid spending further calls after an HTTP or malformed-response failure.
            record("summary", functional_protocol_checks_passed=passed, requests_sent=sent,
                   semantic_accuracy_verified=False, official_accuracy=False, official_evaluation="not_run")
        return 0 if passed else 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(json.dumps({"error": bounded_text(str(exc), 2048), "official_accuracy": False}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
