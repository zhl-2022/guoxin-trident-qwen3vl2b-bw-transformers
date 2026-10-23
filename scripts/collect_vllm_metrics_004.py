#!/usr/bin/env python3
"""Read-only, bounded /metrics capture for one already-running owned vLLM run.

No Torch imports, model requests, service mutations, or acceptance assertions.
Raw metrics retention is first/latest/last-good, not every historical response.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_BODY = 256 * 1024
MAX_JSON = 256 * 1024
MAX_STORAGE = 20_000_000
METRICS = {"vllm:generation_tokens_total", "vllm:request_success_total",
           "vllm:request_generation_tokens_sum", "vllm:request_generation_tokens_count"}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>" + NUMBER + r")(?:\s+" + NUMBER + r")?$" )
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\[\\"n])*)"')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def read_regular(path, limit=MAX_JSON):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("Expected bounded plain file: " + str(path))
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("File changed during read: " + str(path))
    return raw


def process_token(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def json_object(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def runtime_identity(runtime, run_id, token_reader=process_token):
    state_raw = read_regular(runtime / "state.json")
    config_raw = read_regular(runtime / ("config-" + run_id + ".json"))
    state, config = json_object(state_raw), json_object(config_raw)
    if state.get("run_id") != run_id or config.get("run_id") != run_id:
        raise ValueError("Run identity changed or differs from expected run ID")
    if state.get("status") != "ready":
        raise ValueError("Server runtime is not ready: " + str(state.get("status")))
    if state.get("framework") != "vllm" or config.get("framework") != "vllm":
        raise ValueError("Expected explicitly declared vLLM runtime")
    host, port = config.get("host"), config.get("port")
    if not isinstance(host, str) or not ipaddress.ip_address(host).is_loopback or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Runtime endpoint must be numeric loopback")
    bracketed = "[" + host + "]" if ":" in host else host
    url = f"http://{bracketed}:{port}"
    if state.get("url") != url or config.get("url") != url:
        raise ValueError("State/config loopback endpoint mismatch")
    config_path = runtime / ("config-" + run_id + ".json")
    if Path(state.get("config_path", "")).resolve() != config_path.resolve():
        raise ValueError("State config_path mismatch")
    model, served = config.get("model_path"), config.get("served_name")
    if not isinstance(model, str) or not model.startswith("/") or not isinstance(served, str) or not served:
        raise ValueError("Expected explicit absolute model_path and served_name")
    command = config.get("command")
    if not isinstance(command, list) or command != state.get("command") or not all(isinstance(value, str) for value in command):
        raise ValueError("State/config server command mismatch")
    for flag, expected in (("--model", model), ("--served-model-name", served), ("--host", host), ("--port", str(port))):
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command) or command[command.index(flag) + 1] != expected:
            raise ValueError("Server command does not identify configured model/endpoint: " + flag)
    pid, token = state.get("server_pid"), state.get("server_token")
    if type(pid) is not int or pid <= 1 or not isinstance(token, str) or not token.isdigit() or token_reader(pid) != token:
        raise ValueError("Server PID/start token no longer matches")
    identity = {"run_id": run_id, "pid": pid, "start_token": token, "model_path": model, "served_name": served,
                "url": url, "framework": "vllm", "config_sha256": digest(config_raw)}
    return identity, state_raw, config_raw


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError("Metrics redirect refused")


def metrics_request(url, timeout=2):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname or not ipaddress.ip_address(parsed.hostname).is_loopback or parsed.username or parsed.password or parsed.path != "/metrics" or parsed.query or parsed.fragment:
        raise ValueError("Only an explicit loopback HTTP /metrics URL is allowed")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, headers={"Accept": "text/plain", "Accept-Encoding": "identity", "Connection": "close"})
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200 or response.geturl() != url or response.headers.get("Content-Encoding", "identity") not in {"identity", ""}:
            raise ValueError("Metrics endpoint did not return an unredirected plain HTTP 200")
        declared = response.headers.get("Content-Length")
        if declared and (not declared.isdigit() or int(declared) > MAX_BODY):
            raise ValueError("Metrics response exceeds body size limit")
        raw = response.read(MAX_BODY + 1)
    if len(raw) > MAX_BODY:
        raise ValueError("Metrics response exceeds body size limit")
    raw.decode("utf-8", errors="strict")
    return raw


def selected_metrics(raw, identity):
    selected, keys = [], set()
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s{]", line, 1)[0]
        if name not in METRICS:
            continue
        matched = LINE.fullmatch(line)
        if not matched:
            raise ValueError("Selected metric is malformed/nonfinite: " + name)
        labels, block = {}, matched.group("labels")
        if block:
            value = block[1:-1]
            offset = 0
            while offset < len(value):
                label = LABEL.match(value, offset)
                if not label or label.group(1) in labels:
                    raise ValueError("Malformed/duplicate metric labels")
                labels[label.group(1)] = json.loads('"' + label.group(2) + '"')
                offset = label.end()
                if offset != len(value):
                    if value[offset] != ",":
                        raise ValueError("Malformed metric label separator")
                    offset += 1
                    if offset == len(value):
                        raise ValueError("Trailing metric label separator")
        model = labels.get("model_name", labels.get("model"))
        if model is not None and model not in {identity["model_path"], identity["served_name"]}:
            raise ValueError("Metrics model label differs from runtime model")
        number = matched.group("value")
        if not math.isfinite(float(number)) or float(number) < 0:
            raise ValueError("Selected counter must be finite and nonnegative")
        key = (name, tuple(sorted(labels.items())))
        if key in keys:
            raise ValueError("Duplicate selected metric series")
        keys.add(key)
        selected.append({"name": name, "labels": labels, "value_text": number})
        if len(selected) > 64:
            raise ValueError("Too many selected metric series")
    return selected


class Store:
    """Bounded current output files plus a bounded append-only sample log."""
    def __init__(self, folder, limit=MAX_STORAGE):
        self.folder, self.limit, self.sizes = folder, limit, {}
    def reserve(self, name, size, terminal=False):
        reserved = 65536 if terminal else MAX_BODY + MAX_JSON + 65536
        if sum(self.sizes.values()) - self.sizes.get(name, 0) + size > self.limit - reserved:
            raise ValueError("Collector storage budget reached")
    def write(self, name, raw, terminal=False):
        self.reserve(name, len(raw), terminal)
        temporary = self.folder / (name + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(self.folder / name)
        self.sizes[name] = len(raw)
    def append(self, name, raw):
        size = self.sizes.get(name, 0) + len(raw)
        self.reserve(name, size)
        with (self.folder / name).open("ab") as stream:
            stream.write(raw)
            stream.flush()
        self.sizes[name] = size


def collect(runtime, run_id, output, max_seconds=5000, interval_seconds=1, *, token_reader=process_token, request=metrics_request, clock=time.monotonic, sleeper=time.sleep, storage_limit=MAX_STORAGE):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Expected run ID must be 32 lowercase hex characters")
    if not all(math.isfinite(value) and value > 0 for value in (max_seconds, interval_seconds)) or max_seconds > 5000 or interval_seconds < 1:
        raise ValueError("max-seconds must be <=5000 and interval-seconds >=1, both positive finite")
    runtime, output = Path(runtime).absolute(), Path(output).absolute()
    if not runtime.is_dir() or runtime.is_symlink():
        raise ValueError("Runtime must be a plain existing directory")
    output.mkdir(parents=False, exist_ok=False)
    store = Store(output, storage_limit)
    began, count, identity, last_good, last_good_raw = clock(), 0, None, None, None
    record = {"schema": "vllm-metrics-collector-session004-v1", "started_utc": utc(), "expected_run_id": run_id,
              "runtime_dir": str(runtime), "output_dir": str(output), "max_seconds": max_seconds, "interval_seconds": interval_seconds,
              "http_timeout_seconds": 2, "max_raw_bytes": MAX_BODY, "storage_limit_bytes": storage_limit,
              "server_mutated": False, "model_requests_sent": False, "torch_imported_by_collector": False,
              "official_acceptance": False, "per_request_exactness_proven": False, "aggregate_token_comparison_performed": False,
              "raw_retention": "First, latest, and last-good responses only; historical samples retain selected server values and full-response SHA, not full raw responses",
              "boundary": "Server-reported counters are independent of client output_lens fallback, but aggregate equality alone does not prove each request length or isolate benchmark/warmups/other traffic",
              "samples": 0, "reason": "initializing", "identity": None}
    try:
        self_raw = read_regular(Path(__file__).absolute())
        record["collector_sha256"] = digest(self_raw)
        identity, state_raw, config_raw = runtime_identity(runtime, run_id, token_reader)
        record["identity"] = identity
        store.write("runtime-initial-state.json", state_raw)
        store.write("runtime-config.json", config_raw)
        store.write("collector-start.json", encode(record))
        while clock() - began < max_seconds:
            iteration = clock()
            before, state_raw, _ = runtime_identity(runtime, run_id, token_reader)
            if before != identity:
                raise ValueError("Runtime identity changed")
            # One HTTP request per sample; /metrics 200 and runtime ready are the
            # liveness observation. No extra /health or generation HTTP request.
            raw = request(identity["url"] + "/metrics", timeout=2)
            after, after_state, _ = runtime_identity(runtime, run_id, token_reader)
            if after != identity:
                raise ValueError("Runtime identity changed during metrics read")
            selected = selected_metrics(raw, identity)
            sample = {"index": count, "utc": utc(), "elapsed_seconds": round(clock() - began, 6),
                      "raw_bytes": len(raw), "raw_sha256": digest(raw), "identity_sha256": digest(encode(identity)),
                      "state_sha256": digest(after_state), "selected": selected,
                      "selected_families_present": sorted({item["name"] for item in selected})}
            # Reserve the append before changing latest; last-good remains valid
            # if the log budget is exhausted at this sample.
            encoded = encode(sample)
            if len(encoded) > MAX_JSON:
                raise ValueError("Selected sample exceeds metadata size limit")
            store.reserve("samples.jsonl", store.sizes.get("samples.jsonl", 0) + len(encoded))
            store.reserve("metrics-latest.txt", len(raw))
            if count == 0:
                store.write("metrics-first.txt", raw)
                store.write("sample-first.json", encoded)
            store.write("metrics-latest.txt", raw)
            store.write("sample-latest.json", encoded)
            store.write("runtime-last-good-state.json", after_state)
            store.append("samples.jsonl", encoded)
            count += 1
            last_good = sample
            last_good_raw = raw
            remaining = max_seconds - (clock() - began)
            if remaining <= 0:
                break
            sleeper(min(max(0, interval_seconds - (clock() - iteration)), remaining))
        record["reason"] = "deadline_reached"
    except KeyboardInterrupt:
        record["reason"] = "collector_interrupted"
    except (OSError, ValueError, urllib.error.URLError, UnicodeError) as error:
        record["reason"] = "collection_stopped"
        record["error_type"], record["error"] = type(error).__name__, str(error)
    finally:
        record.update(finished_utc=utc(), samples=count, elapsed_seconds=round(clock() - began, 6), last_good=last_good)
        if last_good is not None:
            store.write("metrics-last-good.txt", last_good_raw, terminal=True)
            store.write("sample-last-good.json", encode(last_good), terminal=True)
        record["files"] = [{"name": name, "bytes": size, "sha256": digest(read_regular(output / name, MAX_STORAGE))} for name, size in sorted(store.sizes.items())]
        final = encode(record)
        # A reserved 64KiB terminal record preserves the reason even at log cap.
        if len(final) > 65536 or sum(store.sizes.values()) + len(final) >= storage_limit:
            raise ValueError("Terminal record exceeds reserved collector budget")
        (output / "collector-final.json").write_bytes(final)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory with existing parent")
    parser.add_argument("--max-seconds", type=float, default=5000)
    parser.add_argument("--interval-seconds", type=float, default=1)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Actual PID/start-token collection requires Linux")
    record = collect(args.runtime_dir, args.expected_run_id, args.output_dir, args.max_seconds, args.interval_seconds)
    print(json.dumps({key: record[key] for key in ("reason", "samples", "output_dir", "finished_utc", "official_acceptance")}, indent=2))
    return 0 if record["samples"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error), "error_type": type(error).__name__, "utc": utc()}), file=sys.stderr)
        raise SystemExit(2)
