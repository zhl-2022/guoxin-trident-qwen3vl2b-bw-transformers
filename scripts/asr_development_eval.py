#!/usr/bin/env python3
"""Offline development-only WER/CER scoring and local short-WAV manifest checks."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import wave

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SAMPLES = 10000
MAX_TEXT_CHARACTERS = 16384
MAX_PAIR_CELLS = 4_000_000
MAX_TOTAL_CELLS = 50_000_000
SOURCES = ["https://trec.nist.gov/pubs/trec9/sdrt9_slides/tsld017.htm",
           "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934557"]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON object key")
        obj[key] = value
    return obj


def reject_constant(_value):
    raise ValueError("JSON NaN/Infinity is invalid")


def valid_string(value, field, maximum):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{field} must be a string of at most {maximum} code points")
    value.encode("utf-8", errors="strict")
    return value


def read_jsonl(path, kind):
    with path.open("rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"{kind} must be nonempty and at most {MAX_FILE_BYTES} bytes")
    rows = {}
    # JSON Lines uses LF records. str.splitlines() would split legal U+2028
    # inside a JSON string and silently make Unicode text impossible to score.
    lines = raw.decode("utf-8", errors="strict").split("\n")
    if lines[-1] == "":
        lines.pop()
    if len(lines) > MAX_SAMPLES:
        raise ValueError("Sample limit exceeded")
    fields = {"id", "audio", "sha256"} if kind == "manifest" else {"id", "text"}
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("Unexpected schema; require exactly " + ", ".join(sorted(fields)))
            sample_id = valid_string(row["id"], "id", 128)
            if not sample_id or sample_id.strip() != sample_id or any(ord(c) < 32 or ord(c) == 127 for c in sample_id):
                raise ValueError("id must be nonempty without surrounding whitespace or control characters")
            if sample_id in rows:
                raise ValueError("Duplicate sample id")
            if kind == "manifest":
                audio = valid_string(row["audio"], "audio", 4096)
                if not audio or "://" in audio or audio.startswith(("//", "\\\\")) or "\0" in audio:
                    raise ValueError("audio must name a local path, without URL or UNC network location")
                if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                    raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
            else:
                text = valid_string(row["text"], "text", MAX_TEXT_CHARACTERS)
                if kind == "references" and not text.split():
                    raise ValueError("Empty or whitespace-only references are rejected")
            rows[sample_id] = row
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ValueError(f"Invalid {kind} line {number}: {exc}") from exc
    return rows, {"sha256": digest(raw), "bytes": len(raw), "samples": len(rows)}


def same_ids(expected, actual, label):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        # Do not silently select the intersection or log arbitrary transcript content.
        raise ValueError(f"{label} sample ids differ: missing={len(missing)}, extra={len(extra)}")


def edit_counts(reference, prediction):
    """Unit-cost Levenshtein; deterministic tie priority diagonal, deletion, insertion."""
    # Tuple holds total distance, S, D, I, C; row memory is O(prediction length).
    previous = [(j, 0, 0, j, 0) for j in range(len(prediction) + 1)]
    for i, expected in enumerate(reference, 1):
        current = [(i, 0, i, 0, 0)]
        for j, actual in enumerate(prediction, 1):
            d, s, deleted, inserted, correct = previous[j - 1]
            diagonal = (d, s, deleted, inserted, correct + 1) if expected == actual else (d + 1, s + 1, deleted, inserted, correct)
            d, s, deleted, inserted, correct = previous[j]
            deletion = (d + 1, s, deleted + 1, inserted, correct)
            d, s, deleted, inserted, correct = current[j - 1]
            insertion = (d + 1, s, deleted, inserted + 1, correct)
            current.append(min((diagonal, deletion, insertion), key=lambda value: value[0]))
        previous = current
    distance, substitutions, deletions, insertions, correct = previous[-1]
    return {"reference_units": len(reference), "prediction_units": len(prediction),
            "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "correct": correct, "errors": distance,
            "error_rate": distance / len(reference)}


def scope():
    return {"schema_version": 1, "scope": "development_only", "official_evaluation": False,
            "official_baseline_generated": False, "threshold_applied": False,
            "model_called": False, "network_requests": 0, "gpu_used": False}


def score(references_path, predictions_path, manifest_path=None):
    references, reference_info = read_jsonl(references_path, "references")
    predictions, prediction_info = read_jsonl(predictions_path, "predictions")
    same_ids(references, predictions, "Prediction")
    inputs = {"references": reference_info, "predictions": prediction_info}
    if manifest_path:
        manifest, manifest_info = read_jsonl(manifest_path, "manifest")
        same_ids(references, manifest, "Manifest")
        inputs["manifest"] = manifest_info
    units = []
    total_cells = 0
    for sample_id, row in references.items():
        expected, actual = row["text"], predictions[sample_id]["text"]
        word_ref, word_pred = expected.split(), actual.split()
        cells = len(expected) * len(actual) + len(word_ref) * len(word_pred)
        if max(len(expected) * len(actual), len(word_ref) * len(word_pred)) > MAX_PAIR_CELLS:
            raise ValueError("Per-pair CPU edit-distance limit exceeded; split into shorter utterances explicitly")
        total_cells += cells
        if total_cells > MAX_TOTAL_CELLS:
            raise ValueError("Total CPU edit-distance limit exceeded; use smaller explicit development batches")
        units.append((sample_id, expected, actual, word_ref, word_pred))
    samples = []
    totals = {metric: {name: 0 for name in ("reference_units", "prediction_units", "substitutions", "deletions", "insertions", "correct", "errors")}
              for metric in ("wer", "cer")}
    for sample_id, expected, actual, word_ref, word_pred in units:
        sample = {"id": sample_id, "reference_text_sha256": digest(expected.encode("utf-8")),
                  "prediction_text_sha256": digest(actual.encode("utf-8")),
                  "wer": edit_counts(word_ref, word_pred), "cer": edit_counts(expected, actual)}
        samples.append(sample)
        for metric, counts in totals.items():
            for name in counts:
                counts[name] += sample[metric][name]
    for counts in totals.values():
        counts["error_rate"] = counts["errors"] / counts["reference_units"]
    return {**scope(), "operation": "offline_score", "inputs": inputs,
            "policy": {"normalization": "identity", "text_rewriting": "none",
                       "wer_tokenization": "Python str.split(): Unicode whitespace word boundaries; no Chinese segmentation",
                       "cer_tokenization": "Unicode code points including all whitespace, punctuation and combining marks",
                       "unicode_normalization": "none", "case_folding": False,
                       "number_or_punctuation_mapping": False, "aggregation": "micro: sum(S+D+I) / sum(reference units)",
                       "alignment_cost": "unit insertion/deletion/substitution", "alignment_tie_priority": ["diagonal", "deletion", "insertion"],
                       "empty_predictions": "count as deletions; empty references rejected",
                       "rate_unit": "ratio; may exceed 1.0; multiply by 100 for percent",
                       "manifest_audio_contents_checked": False},
            "definition_sources": SOURCES, "sample_count": len(samples), "totals": totals, "samples": samples}


def check_wav(path, expected_digest):
    with path.open("rb") as handle:
        raw = handle.read(2 * 1024 * 1024 + 1)
    if not raw or len(raw) > 2 * 1024 * 1024:
        raise ValueError("Local diagnostic WAV must be nonempty and at most 2 MiB")
    if digest(raw) != expected_digest:
        raise ValueError("WAV SHA256 does not match manifest")
    with wave.open(io.BytesIO(raw), "rb") as audio:
        channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
        if (channels, width, rate, audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Local diagnostic requires mono 16000 Hz PCM16 WAV; no automatic conversion")
        if not 0 < frames <= 30 * rate:
            raise ValueError("Local diagnostic duration must be >0 and <=30 seconds")
        if len(audio.readframes(frames)) != frames * channels * width:
            raise ValueError("WAV sample payload is truncated")
    return {"sha256": digest(raw), "bytes": len(raw), "channels": channels,
            "sample_width_bytes": width, "sample_rate_hz": rate,
            "frames": frames, "duration_seconds": frames / rate, "resampled_or_trimmed": False}


def manifest_plan(path, check_audio=False):
    rows, info = read_jsonl(path, "manifest")
    samples = []
    for sample_id, row in rows.items():
        sample = {"id": sample_id, "expected_audio_sha256": row["sha256"]}
        if check_audio:
            audio = Path(row["audio"])
            if not audio.is_absolute():
                audio = path.resolve().parent / audio
            sample["audio"] = check_wav(audio, row["sha256"])
        samples.append(sample)
    return {**scope(), "operation": "local_manifest_check" if check_audio else "manifest_plan_only",
            "inputs": {"manifest": info}, "audio_files_opened": check_audio,
            "boundary_scope": "existing short-WAV smoke: mono 16kHz PCM16, >0 and <=30s, <=2MiB; not model maximum or T04 official limits",
            "sample_count": len(samples), "samples": samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score", help="offline references/predictions only; no inference")
    scoring.add_argument("--references", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--manifest", type=Path)
    scoring.add_argument("--normalization", choices=["identity"], default="identity")
    manifest = commands.add_parser("manifest", help="default schema/plan only; no audio file opened")
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--check-audio", action="store_true", help="explicitly read/hash/check the listed local short WAVs")
    for command in (scoring, manifest):
        command.add_argument("--output", type=Path, help="new JSON file; existing files never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise ValueError("Output already exists; use a new development report path")
        report = score(args.references, args.predictions, args.manifest) if args.command == "score" else manifest_plan(args.manifest, args.check_audio)
        report["script_sha256"] = digest(Path(__file__).read_bytes())
        rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(rendered)
        print(rendered, end="")
        return 0
    except (OSError, ValueError, TypeError, UnicodeError, wave.Error, EOFError) as exc:
        print(json.dumps({**scope(), "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
