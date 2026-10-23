"""Collect bounded, non-secret SCNet metadata without importing torch or loading weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
    "F64": 8, "I64": 8, "U64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}
DEFAULT_EXPECTED_ARCHITECTURE = "Qwen3VLForConditionalGeneration"


def architecture_identifier(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value) is None:
        raise ValueError("Expected architecture must be an ASCII class identifier")
    return value


def json_object(raw: str | bytes, label: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate {label} key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError(f"Non-finite number in {label}")

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_checks(model: Path) -> dict:
    config = model / "tokenizer_config.json"
    tokenizer = model / "tokenizer.json"
    vocab = model / "vocab.json"
    merges = model / "merges.txt"
    has_tokenizer_json = tokenizer.exists() or tokenizer.is_symlink()
    result = {
        "tokenizer_present": False,
        "tokenizer_layout": "tokenizer_json" if has_tokenizer_json else "legacy_bpe",
        "tokenizer_json_present": tokenizer.is_file(),
        "legacy_bpe_pair_present": vocab.is_file() and merges.is_file(),
        "tokenizer_validation_scope": "local JSON structure and BPE pair integrity only; tokenizer not executed",
    }
    try:
        selected = (config, tokenizer) if has_tokenizer_json else (config, vocab, merges)
        for path in selected:
            if not path.is_file():
                raise ValueError(f"Required tokenizer file missing: {path.name}")
            if path.resolve().parent != model.resolve():
                raise ValueError("Tokenizer file is outside model directory")
            if not 0 < path.stat().st_size <= 64 * 1024 * 1024:
                raise ValueError(f"Invalid tokenizer metadata size: {path.name}")
        json_object(config.read_text(encoding="utf-8-sig"), "tokenizer config")
        if has_tokenizer_json:
            json_object(tokenizer.read_text(encoding="utf-8-sig"), "tokenizer")
            result["tokenizer_files"] = [config.name, tokenizer.name]
        else:
            vocabulary = json_object(vocab.read_text(encoding="utf-8-sig"), "vocabulary")
            if not vocabulary or any(type(value) is not int or value < 0 for value in vocabulary.values()):
                raise ValueError("Vocabulary must contain nonnegative integer token IDs")
            if len(set(vocabulary.values())) != len(vocabulary):
                raise ValueError("Vocabulary token IDs must be unique")
            pairs = set()
            for line in merges.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip() or line.startswith("#version:"):
                    continue
                pair = tuple(line.split())
                if len(pair) != 2 or any(token not in vocabulary for token in pair):
                    raise ValueError("Each BPE merge must contain two vocabulary tokens")
                if pair in pairs:
                    raise ValueError("Duplicate BPE merge pair")
                pairs.add(pair)
            if not pairs:
                raise ValueError("BPE merges must contain at least one pair")
            result.update(tokenizer_files=[config.name, vocab.name, merges.name],
                          tokenizer_vocab_entries=len(vocabulary), tokenizer_merge_pairs=len(pairs))
        result["tokenizer_present"] = True
    except (OSError, ValueError, TypeError) as exc:
        result["tokenizer_error"] = str(exc)
    return result


def collect(model: Path, hash_weights: bool = False, *, expected_architecture: str = DEFAULT_EXPECTED_ARCHITECTURE) -> dict:
    expected_architecture = architecture_identifier(expected_architecture)
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "sglang", "vllm", "vllm-plugin-das", "sgl-kernel", "triton", "safetensors", "accelerate", "Pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "preflight_only_not_official_evaluation",
        "python": sys.version,
        "system": platform.platform(),
        "packages": packages,
        "tools": {name: bool(shutil.which(name)) for name in ("hy-smi", "rocm-smi", "docker", "podman", "bash")},
        "model_path": str(model),
        "model_exists": model.is_dir(),
        "model_checks": {"expected_architecture_name": expected_architecture},
        "files": [],
    }
    for version_file in (Path("/opt/dtk/.info/version"),):
        if version_file.is_file():
            result["dtk_version_file"] = version_file.read_text(errors="replace")[:4096]
    if shutil.which("hy-smi"):
        try:
            p = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=15)
            result["hy_smi"] = {"returncode": p.returncode, "output": (p.stdout + p.stderr)[:16000]}
        except subprocess.TimeoutExpired:
            result["hy_smi"] = {"error": "timeout"}
    if not model.is_dir():
        result["model_checks"]["complete"] = False
        return result
    config_path = model / "config.json"
    config = json_object(config_path.read_text(encoding="utf-8"), "config") if config_path.is_file() else {}
    checks = result["model_checks"]
    checks["architecture"] = config.get("architectures")
    checks["model_type"] = config.get("model_type")
    checks["dtype"] = config.get("dtype", config.get("torch_dtype"))
    architectures = config.get("architectures", [])
    checks["expected_architecture"] = isinstance(architectures, list) and expected_architecture in architectures
    index_file = model / "model.safetensors.index.json"
    indexed = index_file.exists() or index_file.is_symlink()
    if indexed and (not index_file.is_file() or index_file.resolve().parent != model.resolve()):
        raise ValueError("Index must be a regular file inside model directory")
    index = json_object(index_file.read_text(encoding="utf-8"), "index") if indexed else {}
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in weight_map.items()
    ):
        raise ValueError("Index weight_map must map tensor names to shard names")
    index_metadata = index.get("metadata", {})
    if not isinstance(index_metadata, dict):
        raise ValueError("Index metadata must be an object")
    shard_names = sorted(set(weight_map.values())) if indexed else ["model.safetensors"]
    checks["layout"] = "indexed" if indexed else "single_file"
    checks["index_present"] = indexed
    checks["index_weight_count"] = len(weight_map) if indexed else None
    checks["index_total_size"] = index_metadata.get("total_size") if indexed else None
    for name in shard_names:
        if (model / name).resolve().parent != model.resolve():
            raise ValueError("Weight file is outside model directory")
    # Inspect only this explicit model directory. Never guess a shard set without an index.
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
    candidates = {path.name for path in model.iterdir() if (path.is_file() or path.is_symlink()) and path.suffix in weight_suffixes}
    expected_files = {Path(name).name for name in shard_names}
    extra_files = sorted(candidates - expected_files)
    if extra_files:
        checks["layout_error"] = "additional_weight_files_are_ambiguous"
        checks["additional_weight_files"] = extra_files
    if not indexed:
        other_indexes = sorted(path.name for path in model.glob("*.index.json") if path.is_file() or path.is_symlink())
        if other_indexes:
            checks["layout_error"] = "unsupported_index_present"
            checks["additional_index_files"] = other_indexes
        if not (model / "model.safetensors").is_file():
            checks["layout_error"] = "explicit_model_safetensors_missing"
            checks["layout"] = "unsupported_or_missing"
    checks["shards"] = []
    complete = bool(shard_names) and checks["expected_architecture"] and "layout_error" not in checks
    total_payload_bytes = 0
    for name in shard_names:
        path = (model / name).resolve()
        if path.parent != model.resolve():
            raise ValueError("Index shard is outside model directory")
        shard = {"name": name, "exists": path.is_file()}
        if path.is_file():
            shard["size"] = path.stat().st_size
            try:
                with path.open("rb") as handle:
                    raw = handle.read(8)
                    length = struct.unpack("<Q", raw)[0]
                    if length > 64 * 1024 * 1024 or length > shard["size"] - 8:
                        raise ValueError("Invalid safetensors header size")
                    header = json_object(handle.read(length), "safetensors header")
                tensors = {k: v for k, v in header.items() if k != "__metadata__"}
                if not tensors:
                    raise ValueError("Empty tensor header")
                expected_names = {key for key, value in weight_map.items() if value == name}
                shard["index_keys_match"] = expected_names == set(tensors) if indexed else None
                shard["tensor_count"] = len(tensors)
                payload = shard["size"] - 8 - length
                ranges = []
                dtypes = set()
                for tensor in tensors.values():
                    if not isinstance(tensor, dict):
                        raise ValueError("Tensor metadata must be an object")
                    dtype, shape, offsets = tensor.get("dtype"), tensor.get("shape"), tensor.get("data_offsets")
                    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
                        raise ValueError("Unsupported tensor dtype")
                    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
                        raise ValueError("Invalid tensor shape")
                    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
                        raise ValueError("Tensor offsets must be two integers")
                    start, end = offsets
                    if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
                        raise ValueError("Tensor byte count does not match shape and dtype")
                    ranges.append((start, end))
                    dtypes.add(dtype)
                ranges.sort()
                shard["dtypes"] = sorted(dtypes)
                cursor = 0
                for start, end in ranges:
                    if start != cursor or not start <= end <= payload:
                        raise ValueError("Invalid tensor offsets")
                    cursor = end
                shard["payload_complete"] = cursor == payload
                total_payload_bytes += payload
                if (indexed and not shard["index_keys_match"]) or not shard["payload_complete"]:
                    complete = False
            except (ValueError, KeyError, struct.error, TypeError) as exc:
                shard["error"] = str(exc)
                complete = False
            if hash_weights:
                shard["sha256"] = sha256(path)
        else:
            complete = False
        checks["shards"].append(shard)
    declared_size = checks["index_total_size"]
    checks["payload_bytes"] = total_payload_bytes
    checks["tensor_count"] = sum(shard.get("tensor_count", 0) for shard in checks["shards"])
    checks["index_total_size_matches"] = None if not indexed else (
        declared_size is None or (type(declared_size) is int and declared_size >= 0 and declared_size == total_payload_bytes)
    )
    if indexed:
        complete = complete and checks["index_total_size_matches"]
    for name in ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json"):
        path = model / name
        if path.is_file():
            result["files"].append({"name": name, "size": path.stat().st_size, "sha256": sha256(path)})
    checks.update(tokenizer_checks(model))
    checks["complete"] = complete and checks["tokenizer_present"]
    result["weights_hashed"] = hash_weights
    result["note"] = "Header integrity is not inference correctness. No official baseline or acceptance checked."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--expected-architecture", type=architecture_identifier, default=DEFAULT_EXPECTED_ARCHITECTURE,
                        help="architecture class name to check as plain text (default: %(default)s)")
    args = parser.parse_args()
    result = collect(args.model, args.hash_weights, expected_architecture=args.expected_architecture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "complete": result["model_checks"].get("complete", False), "kind": result["kind"]}))
    return 0 if result["model_checks"].get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
