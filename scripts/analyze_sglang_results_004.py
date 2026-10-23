#!/usr/bin/env python3
"""Read-only post-analysis of archived SGLang004 results; write a NEW directory.

Only the top-level request_rate JSON token Infinity is normalized to "inf".
No source run/native artifact or active producer is changed. Native client
completeness is not official acceptance, semantic accuracy, or exact-token proof.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
APPROVED_DRIVER_SHA256 = "ff8eef852e19f336e99884cc031353627ca304c6a7534c22375116758a850000"
UPSTREAM_BENCH_SHA256 = "129e40b9e361be192306a9df56df41507f0397d5768696db3a9b994421daa259"
SOURCE_FILES = ("__init__.py", "bench_serving.py", "benchmark/datasets/image.py", "benchmark/datasets/common.py", "benchmark/utils.py")
ARRAYS = ("input_lens", "output_lens", "ttfts", "itls", "generated_texts", "errors")
BOUNDARY = ("Native output_lens can come from stream usage or requested-length fallback; raw SSE usage, terminal framing, "
            "and exact 1024 generated tokens are not independently certified by this JSONL. Upstream combines reasoning_content/content.")
LATENCY_FIELDS = ("mean_e2e_latency_ms", "median_e2e_latency_ms", "std_e2e_latency_ms", "p90_e2e_latency_ms", "p99_e2e_latency_ms",
                  "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
                  "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p95_itl_ms", "p99_itl_ms")


@dataclass(frozen=True)
class NonFiniteToken:
    token: str


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def no_links(path):
    for part in (Path(path).absolute(), *Path(path).absolute().parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse path refused: " + str(part))


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Expected archive-relative POSIX file name")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Unsafe archive-relative path")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(raw, permit_request_rate=False):
    normalized = []
    data = json.loads(raw.decode("utf-8"), parse_constant=NonFiniteToken, object_pairs_hook=unique_object)

    def walk(value, path=()):
        if isinstance(value, NonFiniteToken):
            if permit_request_rate and path == ("request_rate",) and value.token == "Infinity":
                normalized.append({"path": "$.request_rate", "original_token": "Infinity", "normalized_value": "inf",
                                   "reason": "upstream json.dumps default for --request-rate inf"})
                return "inf"
            raise ValueError("Non-finite JSON constant refused at " + repr(path) + ": " + value.token)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Overflow/non-finite JSON number refused at " + repr(path))
        if isinstance(value, dict):
            return {key: walk(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, path + (index,)) for index, item in enumerate(value)]
        return value
    return walk(data), normalized


class Archive:
    def __init__(self, root):
        no_links(root)
        self.root = Path(root).resolve()
        self.raw = {}

    def read(self, name):
        name = relative(name)
        if name in self.raw:
            return self.raw[name]
        path = self.root.joinpath(*PurePosixPath(name).parts)
        no_links(path)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Archive path escapes input root")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise ValueError("Not a bounded regular source file: " + name)
        with path.open("rb") as source:
            raw = source.read(MAX_FILE_BYTES + 1)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
            raise ValueError("Input changed during read: " + name)
        self.raw[name] = raw
        if sum(map(len, self.raw.values())) > MAX_TOTAL_BYTES:
            raise ValueError("Source snapshot exceeds200MiB")
        return raw

    def inventory_native(self):
        base = self.root / "native"
        no_links(base)
        if not base.is_dir():
            raise ValueError("Archived native directory missing")
        names = []
        def failure(error):
            raise error
        for directory, dirs, files in os.walk(base, onerror=failure, followlinks=False):
            for name in dirs:
                no_links(Path(directory) / name)
            for name in sorted(files):
                local = (Path(directory) / name).relative_to(self.root).as_posix()
                self.read(local)
                names.append(local)
        return [{"path": name, "bytes": len(self.raw[name]), "sha256": sha(self.raw[name])} for name in sorted(names)]

    def verify_unchanged(self):
        for name, expected in self.raw.items():
            path = self.root / name
            no_links(path)
            if sha(path.read_bytes()) != sha(expected):
                raise ValueError("Archived source changed after snapshot: " + name)


class ExportSeal:
    """Explicit hash-only source proof from a separately checksum-verified export.

    Exporter intentionally omits Python payloads. Its before/after sealed hashes
    can bind the producer/source identities without fabricating source files.
    The caller must supply the independently verified manifest SHA explicitly.
    """
    def __init__(self, path, expected_sha256, archive, run):
        path = Path(path).absolute()
        no_links(path)
        self.archive = Archive(path.parent)
        raw = self.archive.read(path.name)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") or sha(raw) != expected_sha256:
            raise ValueError("Explicit export manifest SHA is missing or differs")
        manifest, _ = parse_json(raw)
        if manifest.get("schema") != "hg-session004-evidence-export-v1" or manifest.get("script_hashes_unchanged") is not True:
            raise ValueError("Export seal schema/script stability differs")
        before, after = manifest.get("script_hashes_before"), manifest.get("script_hashes_after")
        keys = ("archive_path", "source", "size", "sha256")
        if not isinstance(before, list) or not before or not isinstance(after, list) or [{k:item[k] for k in keys} for item in before] != [{k:item[k] for k in keys} for item in after]:
            raise ValueError("Export script hashes differ before/after")
        self.scripts = {}
        for item in before:
            name = relative(item["archive_path"])
            if name in self.scripts or type(item.get("size")) is not int or item["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError("Duplicate or invalid sealed script hash")
            self.scripts[name] = item
        rows = manifest.get("files", [])
        self.files = {relative(item["archive_path"]): item for item in rows}
        if len(self.files) != len(rows):
            raise ValueError("Duplicate sealed payload entry")
        try:
            self.prefix = archive.root.relative_to(path.parent.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("Export manifest is not an ancestor of this archived run") from error
        if not self.prefix.startswith("project/"):
            raise ValueError("Run must map to the export project/ subtree")
        expected_remote_root = str(manifest.get("project", "")).rstrip("/") + "/" + self.prefix[len("project/"):]
        if run.get("output_dir") != expected_remote_root:
            raise ValueError("Export project/run source path differs from producer output_dir")
        self.remote_root = expected_remote_root
        self.summary = {"manifest_path": str(path), "manifest_sha256": sha(raw),
                        "before_after_script_hashes_identical": True, "export_finished_utc": manifest.get("finished_utc"),
                        "capture_semantics": manifest.get("capture_semantics"),
                        "source_payloads_not_reconstructed": True,
                        "scope": "Explicit independently-checksummed export manifest; source content hash proof only when Python payloads were intentionally omitted"}
        self.verify_payload("run.json", archive.read("run.json"))

    def verify_payload(self, name, raw):
        key = self.prefix + "/" + relative(name)
        item = self.files.get(key)
        if not item or item.get("source") != self.remote_root + "/" + name or item.get("sha256") != sha(raw) or type(item.get("size")) is not int or item["size"] != len(raw):
            raise ValueError("Export-sealed payload source/size/SHA differs: " + name)

    def script(self, name):
        key = self.prefix + "/" + relative(name)
        item = self.scripts.get(key)
        if not item or item.get("source") != self.remote_root + "/" + name:
            raise ValueError("Export-sealed script source missing/differs: " + name)
        return {"path": name, "sha256": item["sha256"], "bytes": item["size"],
                "source_payload_archived": False, "proof_type": "export_manifest_before_after_script_hash_only",
                "remote_source": item["source"], "export_manifest_sha256": self.summary["manifest_sha256"]}


def finite_nonnegative(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def native_result(raw, row):
    text = raw.decode("utf-8")
    if len(text.splitlines()) != 1:
        raise ValueError("Native JSONL must contain exactly one original record")
    data, normalizations = parse_json(raw, permit_request_rate=True)
    if not isinstance(data, dict):
        raise ValueError("Native result must be an object")
    expected = row["expected_requests"]
    issues = []
    if type(data.get("completed")) is not int or data["completed"] != expected:
        issues.append("completed differs from expected requests")
    for name in ARRAYS:
        values = data.get(name)
        if not isinstance(values, list) or len(values) != expected:
            issues.append(name + " must contain exactly one entry per request")
            continue
        if name == "generated_texts" and any(not isinstance(item, str) or not item.strip() for item in values):
            issues.append("empty or non-string generated text")
        elif name == "errors" and any(type(item) is not str or item != "" for item in values):
            issues.append("request errors present or wrongly typed")
        elif name == "output_lens" and any(type(item) is not int or item != 1024 for item in values):
            issues.append("client-reported output length differs from1024")
        elif name == "input_lens" and any(type(item) is not int or item <= 0 for item in values):
            issues.append("client-reported input lengths are not positive integers")
        elif name == "ttfts" and any(not finite_nonnegative(item) for item in values):
            issues.append("per-request ttfts are not finite nonnegative numbers")
        elif name == "itls" and any(not isinstance(item, list) or any(not finite_nonnegative(value) for value in item) for item in values):
            issues.append("per-request itls must be lists of finite nonnegative numbers")
    expected_fields = {"backend": "sglang-oai-chat", "dataset_name": "image", "max_concurrency": row["concurrency"],
                       "random_input_len": row["text_target_tokens"], "random_output_len": 1024, "random_range_ratio": 1}
    for key, expected_value in expected_fields.items():
        if type(data.get(key)) is bool or data.get(key) != expected_value:
            issues.append("native configuration differs: " + key)
    if data.get("request_rate") != "inf" or len(normalizations) != 1:
        issues.append("native request_rate must be the positive Infinity token for the archived inf workload")
    for key in ("total_input_tokens", "total_input_text_tokens", "total_input_vision_tokens", "total_output_tokens", "total_output_tokens_retokenized"):
        if type(data.get(key)) is not int or data[key] < 0:
            issues.append("aggregate count must be nonnegative integer: " + key)
    if type(data.get("total_output_tokens")) is int and data["total_output_tokens"] != 1024 * expected:
        issues.append("total_output_tokens differs from expected client-reported lengths")
    if all(type(data.get(key)) is int for key in ("total_input_tokens", "total_input_text_tokens", "total_input_vision_tokens")):
        if data["total_input_tokens"] != data["total_input_text_tokens"] + data["total_input_vision_tokens"]:
            issues.append("total_input text/vision decomposition differs")
    if isinstance(data.get("input_lens"), list) and all(type(item) is int for item in data["input_lens"]):
        if data.get("total_input_tokens") != sum(data["input_lens"]):
            issues.append("total_input_tokens differs from input_lens sum")
    if not finite_nonnegative(data.get("duration")) or data.get("duration", 0) <= 0:
        issues.append("native duration must be positive finite")
    for key in LATENCY_FIELDS + ("request_throughput", "input_throughput", "output_throughput", "total_throughput", "concurrency"):
        if not finite_nonnegative(data.get(key)):
            issues.append("metric must be finite nonnegative: " + key)
    if finite_nonnegative(data.get("duration")) and data["duration"] > 0:
        for key, total in (("request_throughput", expected), ("output_throughput", 1024 * expected),
                           ("input_throughput", data.get("total_input_tokens")),
                           ("total_throughput", data.get("total_input_tokens", 0) + data.get("total_output_tokens", 0)
                            if type(data.get("total_input_tokens")) is int and type(data.get("total_output_tokens")) is int else None)):
            if type(total) in {int, float} and finite_nonnegative(data.get(key)) and not math.isclose(data[key], total / data["duration"], rel_tol=1e-6, abs_tol=1e-6):
                issues.append("throughput/count/duration mismatch: " + key)
    return {"issues": issues, "native_structure_complete": not issues,
            "normalizations": normalizations, "metrics": {key: value for key, value in data.items() if key not in ARRAYS},
            "native_record": data, "exact_output_tokens_independently_verified": False,
            "raw_usage_received_independently_verified": False, "stream_terminal_framing_independently_verified": False,
            "output_length_boundary": BOUNDARY}


def source_provenance(archive, run, export_seal=None):
    issues = []
    if run.get("mode") != "run" or run.get("client_started") is not True or run.get("filesystem_mutated") is not True:
        issues.append("producer is plan-only or lacks actual execution/archive markers")
    identity = run.get("client_identity", {})
    if not isinstance(identity, dict):
        identity = {}
        issues.append("client identity is not an object")
    if run.get("client_type") != "upstream_sglang" or identity.get("upstream_verified") is not True or identity.get("version") != "0.5.12":
        issues.append("not a declared verified upstream SGLang0.5.12 client")
    if run.get("client_backend") != "sglang-oai-chat" or run.get("server_framework") != "vllm":
        issues.append("backend/server declaration differs from SGLang OpenAI client against vLLM")
    identity_process = identity.get("identity_process", {})
    if not isinstance(identity_process, dict):
        identity_process = {}
        issues.append("identity process record is not an object")
    if identity_process.get("process_status") != "exited" or type(identity_process.get("exit_code")) is not int or identity_process["exit_code"] != 0 or identity_process.get("child_reaped") is not True:
        issues.append("client identity process did not exit0 and reap")
    artifacts = []
    def artifact(name):
        try:
            raw = archive.read(name)
            return {"path": name, "sha256": sha(raw), "bytes": len(raw), "source_payload_archived": True, "proof_type": "archived_source_payload"}
        except FileNotFoundError:
            if export_seal is None:
                raise
            return export_seal.script(name)
    try:
        driver = artifact("sources/t02_sglang_bench.py")
        driver_sha = driver["sha256"]
        artifacts.append(driver)
        if run.get("driver_sha256") != driver_sha or driver_sha != APPROVED_DRIVER_SHA256:
            issues.append("archived producer SHA is not the approved004g driver")
    except (ValueError, OSError) as error:
        issues.append("producer archive unavailable: " + str(error))
    entries = identity.get("sources", [])
    if not isinstance(entries, list) or len(entries) != len(SOURCE_FILES) or {item.get("path") for item in entries if isinstance(item, dict)} != set(SOURCE_FILES):
        issues.append("upstream source manifest must have the exact five expected modules")
    else:
        for item in entries:
            try:
                name = "sources/upstream-sglang/" + relative(item["path"])
                proof = artifact(name)
                item_sha = proof["sha256"]
                artifacts.append(proof)
                if type(item.get("bytes")) is not int or item["bytes"] != proof["bytes"] or item.get("sha256") != item_sha:
                    issues.append("archived upstream source SHA/size differs: " + item["path"])
                if item["path"] == "bench_serving.py" and item_sha != UPSTREAM_BENCH_SHA256:
                    issues.append("bench_serving.py differs from independently archived upstreamv0.5.12 reference")
            except (ValueError, OSError) as error:
                issues.append("upstream source unavailable: " + str(error))
    return {"issues": issues, "verified_for_reanalysis": not issues, "artifacts": artifacts,
            "export_source_hash_seal": None if export_seal is None else export_seal.summary,
            "approved_driver_sha256": APPROVED_DRIVER_SHA256, "upstream_bench_reference_sha256": UPSTREAM_BENCH_SHA256,
            "server_framework_verified_from_runtime": run.get("server_framework_verified_from_runtime") is True,
            "official_framework_acceptance_verified": False}


def row_contract(run, row):
    issues = []
    n, c, round_ = row.get("text_target_tokens"), row.get("concurrency"), row.get("round")
    if type(n) is not int or n not in (4096, 16384) or type(c) is not int or c not in (1, 8, 64) or type(round_) is not int or round_ not in (1, 2):
        raise ValueError("Row is not one registered004 load/round")
    name = f"round{round_}_{n}_1024_c{c}"
    if row.get("name") != name or row.get("output_target_tokens") != 1024 or type(row.get("expected_requests")) is not int or row["expected_requests"] != max(8, 2*c):
        issues.append("row identity/output/expected count differs from registered load")
    if row.get("scored_round") is not (round_ == 2):
        issues.append("scored_round differs from round2 rule")
    if row.get("client_backend") != "sglang-oai-chat" or row.get("server_framework") != "vllm":
        issues.append("row backend/server differs")
    old_output = str(run.get("output_dir", "")).replace("\\", "/").rstrip("/")
    original_native = row.get("native_result")
    if not old_output or original_native != old_output + "/native/" + name + ".jsonl":
        issues.append("original native path does not match archived producer output/name")
    command = row.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command) or command[:4] != [run.get("client_python"), "-B", "-m", "sglang.bench_serving"]:
        issues.append("argv prefix is not the original upstream module client")
        return name, issues
    options = {}; index = 4
    while index < len(command):
        key = command[index]
        if not key.startswith("--") or key in options:
            issues.append("unexpected or duplicated argv option"); break
        if key == "--output-details":
            options[key] = True; index += 1
        elif index + 1 < len(command):
            options[key] = command[index+1]; index += 2
        else:
            issues.append("option lacks value"); break
    expected = {"--backend": "sglang-oai-chat", "--host": run.get("host"), "--port": str(run.get("port")),
                "--model": run.get("model_path"), "--served-model-name": run.get("served_model_name"), "--tokenizer": run.get("model_path"),
                "--dataset-name": "image", "--image-count": "1", "--image-resolution": "1024x1024", "--image-content": "random", "--image-format": "jpeg",
                "--random-input-len": str(n), "--random-output-len": "1024", "--random-range-ratio": "1", "--seed": "1",
                "--max-concurrency": str(c), "--num-prompts": str(max(8,2*c)), "--request-rate": "inf", "--warmup-requests": str(row.get("warmup_requests")),
                "--ready-check-timeout-sec": "0", "--output-details": True, "--output-file": original_native}
    if set(options) != set(expected) | {"--extra-request-body"} or any(options.get(key) != value for key, value in expected.items()):
        issues.append("argv flags/values differ from the original004 workload contract")
    try:
        body, _ = parse_json(options.get("--extra-request-body", "").encode())
        if body != {"temperature": 0, "ignore_eos": True, "stream_options": {"include_usage": True}}:
            issues.append("extra request body differs from archived004 contract")
    except (ValueError, UnicodeError) as error:
        issues.append("extra request body invalid: " + str(error))
    if type(row.get("warmup_requests")) is not int or not 0 <= row["warmup_requests"] <= 8:
        issues.append("warmup count invalid")
    return name, issues


def analyze(run_path, analysis_dir, export_manifest=None, expected_export_manifest_sha256=None):
    run_path = Path(run_path).absolute()
    analysis_dir = Path(analysis_dir).absolute()
    no_links(run_path); no_links(analysis_dir)
    root = run_path.parent.resolve()
    if run_path.name != "run.json" or not analysis_dir.is_absolute() or analysis_dir.exists():
        raise ValueError("Use archived run.json and a NEW analysis directory")
    if analysis_dir.resolve().is_relative_to(root) or root.is_relative_to(analysis_dir.resolve()):
        raise ValueError("Analysis output cannot contain/be inside the input archive")
    archive = Archive(root)
    run_raw = archive.read("run.json")
    run, _ = parse_json(run_raw)
    if not isinstance(run, dict) or not isinstance(run.get("rows"), list) or not 1 <= len(run["rows"]) <= 12:
        raise ValueError("run.json must contain1..12 archived004 rows")
    inventory = archive.inventory_native()
    if bool(export_manifest) != bool(expected_export_manifest_sha256):
        raise ValueError("Export manifest and its independently verified SHA must be supplied together")
    export_seal = ExportSeal(export_manifest, expected_export_manifest_sha256, archive, run) if export_manifest else None
    provenance = source_provenance(archive, run, export_seal)
    rows = []; normalized_files = {}; seen = set()
    for producer in run["rows"]:
        if not isinstance(producer, dict):
            raise ValueError("Producer row is not an object")
        name, issues = row_contract(run, producer)
        if name in seen:
            raise ValueError("Duplicate run row: " + name)
        seen.add(name)
        item = {"name": name, "round": producer["round"], "scored_round": producer.get("scored_round"),
                "text_target_tokens": producer["text_target_tokens"], "concurrency": producer["concurrency"],
                "expected_requests": producer.get("expected_requests"), "original_producer_status": producer.get("status"),
                "original_producer_result_error": producer.get("result_error"), "process_status": producer.get("process_status"),
                "exit_code": producer.get("exit_code"), "child_reaped": producer.get("child_reaped"),
                "allocated_timeout_seconds": producer.get("allocated_timeout_seconds"), "client_wall_seconds": producer.get("client_wall_seconds"),
                "source_native_claim": producer.get("native_result"), "analysis_issues": issues,
                "eligible_client_report": False, "official_acceptance": False, "exact_output_tokens_independently_verified": False}
        local = "native/" + name + ".jsonl"
        try:
            raw = archive.read(local)
            if export_seal is not None:
                export_seal.verify_payload(local, raw)
            item.update(native_file=local, native_file_bytes=len(raw), native_file_sha256=sha(raw),
                        producer_claimed_native_sha256=producer.get("native_result_sha256"))
            if producer.get("native_result_sha256") != sha(raw):
                issues.append("producer native SHA missing or differs from archived bytes")
            parsed = native_result(raw, producer)
            item["parsed"] = {key: value for key, value in parsed.items() if key != "native_record"}
            issues.extend(parsed["issues"])
            normalized_files["normalized/" + name + ".json"] = encode(parsed["native_record"])
            item["normalized_native_file"] = "normalized/" + name + ".json"
        except (ValueError, UnicodeError, OSError) as error:
            issues.append("native parse/read failed: " + str(error))
        if producer.get("process_status") != "exited" or type(producer.get("exit_code")) is not int or producer["exit_code"] != 0 or producer.get("child_reaped") is not True:
            issues.append("producer did not complete an exited0/reaped process; timeout/interrupted/native partial never counts")
        if not finite_nonnegative(producer.get("client_wall_seconds")) or producer.get("client_wall_seconds", 0) <= 0:
            issues.append("producer wall time is not positive finite")
        status = producer.get("status")
        if status not in {"complete_client_report", "invalid_native_result"} or (status == "invalid_native_result" and producer.get("result_error") != "Infinity"):
            issues.append("producer state is not a completed report or the specific Infinity parse failure")
        if not provenance["verified_for_reanalysis"]:
            issues.append("upstream/producer provenance not closed; see source_provenance")
        item["eligible_client_report"] = not issues
        item["analysis_status"] = "complete_native_client_report_reanalyzed" if not issues else "not_eligible"
        rows.append(item)
    archive.verify_unchanged()
    if export_seal is not None:
        export_seal.archive.verify_unchanged()
    eligible = [row for row in rows if row["eligible_client_report"]]
    all_loads = {(n,c,r) for n in (4096,16384) for c in (1,8,64) for r in (1,2)}
    report = {"schema_version": 1, "analyzer": "analyze_sglang_results_004", "created_utc": datetime.now(timezone.utc).isoformat(),
              "analyzer_sha256": sha(Path(__file__).read_bytes()), "source_run_path": str(run_path), "source_run_sha256": sha(run_raw),
              "source_run_bytes": len(run_raw), "source_producer_status": run.get("status"), "source_rows": len(rows),
              "normalization_policy": "Only JSON token Infinity at top-level $.request_rate becomes string inf; all other NaN/Inf and overflow refused",
              "native_file_inventory": inventory, "unreferenced_native_files": sorted(set(row["path"] for row in inventory) - {"native/"+name+".jsonl" for name in seen}),
              "source_provenance": provenance, "rows": rows,
              "summary": {"eligible_client_reports": len(eligible), "selected_reports_complete": len(eligible) == len(rows),
                          "full_two_round_matrix_planned": {(row["text_target_tokens"],row["concurrency"],row["round"]) for row in rows} == all_loads,
                          "full_two_round_matrix_complete": {(row["text_target_tokens"],row["concurrency"],row["round"]) for row in eligible} == all_loads,
                          "eligible_client_reported_requests": sum(row["expected_requests"] for row in eligible),
                          "eligible_client_reported_output_tokens": sum(row["parsed"]["metrics"]["total_output_tokens"] for row in eligible),
                          "round2_eligible_reports": sum(row["round"] == 2 for row in eligible),
                          "official_accuracy_verified": False, "official_performance_accepted": False},
              "output_length_boundary": BOUNDARY, "source_files_modified": False, "producer_driver_modified": False,
              "model_or_gpu_execution_performed": False, "source_snapshot_unchanged_at_analysis_end": True}
    payloads = {"analysis.json": encode(report), "source-run.json": run_raw, **normalized_files}
    lines = ["# Archived SGLang native result reanalysis", "", BOUNDARY, "",
             "Only request_rate Infinity is normalized. Original producer statuses and every native file SHA remain in analysis.json.", "",
             "| Load | Producer status | Exit/process | Reanalysis | Completed | Output token/s | Wall seconds |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        metrics = row.get("parsed", {}).get("metrics", {})
        lines.append(f"| {row['name']} | {row['original_producer_status']} | {row['exit_code']}/{row['process_status']} | {row['analysis_status']} | {metrics.get('completed','')} | {metrics.get('output_throughput','')} | {row['client_wall_seconds']} |")
    lines.extend(["", f"Eligible native client reports: {len(eligible)}/{len(rows)}. Official acceptance: false. Exact output token proof: false.", ""])
    payloads["summary.md"] = "\n".join(lines).encode("utf-8")
    # Validate everything before the first output mutation. Existing output is
    # never updated, even when an archived source failed validation.
    no_links(analysis_dir)
    if analysis_dir.exists():
        raise ValueError("Analysis output appeared during analysis; refusing overwrite")
    analysis_dir.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        target = analysis_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as destination:
            destination.write(raw)
        if target.read_bytes() != raw:
            raise ValueError("Analysis output readback differs")
    manifest = {"schema_version": 1, "self_hash_excluded": True,
                "files": [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name,raw in sorted(payloads.items())],
                "source_file_inventory": [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name,raw in sorted(archive.raw.items())]}
    with (analysis_dir/"analysis-manifest.json").open("xb") as destination:
        destination.write(encode(manifest))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Archived run.json next to native/ and sources/")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="New sibling/independent directory; never inside input")
    parser.add_argument("--export-manifest", type=Path, help="Optional verified export-manifest.json ancestor; enables explicit sealed Python hash-only proof")
    parser.add_argument("--expected-export-manifest-sha256", help="Required with --export-manifest; independently verified SHA, not read from a claim inside that file")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.run, args.analysis_dir, args.export_manifest, args.expected_export_manifest_sha256)
        print(json.dumps({"analysis_dir": str(args.analysis_dir), "source_run_sha256": report["source_run_sha256"], "summary": report["summary"]}, indent=2, allow_nan=False))
        return 0 if report["summary"]["selected_reports_complete"] else 1
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
