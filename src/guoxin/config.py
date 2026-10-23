"""Strict configuration loading and explicit local artifact verification."""

import hashlib
import json
import re
from pathlib import Path

from .gates import accuracy_gate, budget_estimate

EXPECTED_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
SUPPORTED_ARCHITECTURES = frozenset({EXPECTED_ARCHITECTURE, "Qwen3ASRForConditionalGeneration"})
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")
IMAGE_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")


def object_value(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def reject_constant(value: str):
    raise ValueError(f"JSON constant {value} is not allowed")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("JSON input exceeds the 4 MiB limit")
    with path.open("r", encoding="utf-8-sig") as handle:
        return object_value(
            json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object),
            "JSON root",
        )


def project_path(project: Path, value: object, *, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a nonempty string")
    base = project.resolve(strict=True)
    unresolved = Path(value)
    target = (unresolved if unresolved.is_absolute() else base / unresolved).resolve(
        strict=must_exist
    )
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("path must stay inside the project, including symlink targets") from exc
    return target


def display_path(project: Path, path: Path) -> str:
    return path.relative_to(project.resolve()).as_posix()


def load_task(project: Path, task_path: str) -> tuple[dict, Path]:
    path = project_path(project, task_path, must_exist=True)
    config = read_json(path)
    if config.get("schema_version") != 1 or isinstance(config.get("schema_version"), bool):
        raise ValueError("schema_version must be 1")
    if not isinstance(config.get("slug"), str) or not SLUG.fullmatch(config["slug"]):
        raise ValueError("slug must be 1-80 lowercase letters, digits, underscores or hyphens")
    for section in ("model", "official", "accuracy", "budget"):
        object_value(config.get(section), section)
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be an array")
    names = set()
    paths = set()
    for artifact in artifacts:
        object_value(artifact, "artifact")
        name = artifact.get("name")
        if not isinstance(name, str) or not SLUG.fullmatch(name) or name in names:
            raise ValueError("each artifact needs a unique valid name")
        names.add(name)
        # Validate even missing files now; traversal cannot become a readiness warning.
        target = project_path(project, artifact.get("path"))
        if target in paths:
            raise ValueError("artifacts must not repeat the same resolved path")
        paths.add(target)
        digest = artifact.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise ValueError("artifact sha256 must be null or 64 hexadecimal characters")
    evidence = config.get("evidence_files", [])
    if not isinstance(evidence, list):
        raise ValueError("evidence_files must be an array of explicit project file paths")
    for value in evidence:
        project_path(project, value)
    model_path = config["model"].get("config_path")
    if model_path is not None:
        project_path(project, model_path)
    expected = config["model"].get("expected_architecture", EXPECTED_ARCHITECTURE)
    if not isinstance(expected, str) or expected not in SUPPORTED_ARCHITECTURES:
        raise ValueError("unsupported expected_architecture; use an explicitly supported model class")
    accuracy = config["accuracy"]
    if not isinstance(accuracy.get("metric"), str) or not accuracy["metric"].strip():
        raise ValueError("accuracy.metric must be a nonempty string")
    # Validate the threshold even while the official baseline score is unavailable.
    sample_baseline = accuracy.get("baseline")
    if sample_baseline is None:
        sample_baseline = 0
    accuracy_gate(
        sample_baseline,
        sample_baseline,
        direction=accuracy.get("direction"),
        max_drop=accuracy.get("max_drop"),
        drop_mode=accuracy.get("drop_mode"),
        full_scale=accuracy.get("full_scale", 1),
        metric=accuracy["metric"],
    )
    budget = config["budget"]
    budget_estimate(
        budget.get("cards"),
        budget.get("wall_hours"),
        budget.get("total_card_hours"),
        budget.get("used_card_hours", 0),
    )
    return config, path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(project: Path, config: dict) -> dict:
    results = []
    for artifact in config["artifacts"]:
        path = project_path(project, artifact["path"])
        expected = artifact.get("sha256")
        row = {
            "name": artifact["name"],
            "path": display_path(project, path),
            "role": artifact.get("role", "unspecified"),
            "verified": False,
        }
        if not path.is_file():
            row["reason"] = "missing_regular_file"
        elif expected is None:
            row["reason"] = "expected_sha256_missing"
        else:
            actual = file_hash(path)
            row.update({"sha256": actual, "expected_sha256": expected.lower()})
            row["verified"] = actual == expected.lower()
            if not row["verified"]:
                row["reason"] = "sha256_mismatch"
        results.append(row)
    return {"verified": bool(results) and all(row["verified"] for row in results), "files": results}


def validate_model(project: Path, config_path: object, expected_architecture: str = EXPECTED_ARCHITECTURE) -> dict:
    if expected_architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError("unsupported expected_architecture")
    if config_path is None:
        return {"valid": False, "reason": "model_config_path_missing"}
    path = project_path(project, config_path)
    if not path.is_file():
        return {"valid": False, "reason": "model_config_file_missing", "path": display_path(project, path)}
    model = read_json(path)
    architectures = model.get("architectures")
    valid = (
        isinstance(architectures, list)
        and all(isinstance(item, str) for item in architectures)
        and expected_architecture in architectures
    )
    result = {
        "valid": valid,
        "path": display_path(project, path),
        "sha256": file_hash(path),
        "expected_architecture": expected_architecture,
        "architectures": architectures,
        "model_type": model.get("model_type"),
        "dtype": model.get("dtype", model.get("torch_dtype")),
        "weights_loaded": False,
    }
    if not valid:
        result["reason"] = "architecture_mismatch"
    return result


def image_reference_valid(value: object) -> bool:
    """Require a named image path and explicit tag/digest, not a UI page/name.

    This is only syntax checking. Registry accessibility and a clean replay
    remain separate checks; the configured string is not runtime attestation.
    """
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value) or "://" in value:
        return False
    if any(ch in value for ch in "?#\\"):
        return False
    if "@" in value:
        if value.count("@") != 1:
            return False
        path, digest = value.split("@")
        if not IMAGE_DIGEST.fullmatch(digest):
            return False
    else:
        path, separator, tag = value.rpartition(":")
        if not separator or not IMAGE_TAG.fullmatch(tag) or "/" in tag:
            return False
    components = path.split("/")
    return (len(components) >= 2 and all(components)
            and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", part) for part in components))


def task_plan(project: Path, config: dict) -> dict:
    artifacts = verify_artifacts(project, config)
    model = validate_model(
        project, config["model"].get("config_path"),
        config["model"].get("expected_architecture", EXPECTED_ARCHITECTURE),
    )
    official = config["official"]
    verified_names = {row["name"] for row in artifacts["files"] if row["verified"]}
    blockers = []
    advisories = []
    claim = official.get("claim")
    if not (
        isinstance(claim, dict)
        and claim.get("confirmed") is True
        and isinstance(claim.get("reference"), str)
        and claim["reference"].strip()
    ):
        advisories.append("task_priority_not_independently_verified")
    for key in ("baseline", "evaluator", "dataset"):
        reference = official.get(key)
        if not isinstance(reference, dict) or reference.get("artifact") not in verified_names:
            blockers.append(f"official_{key}_artifact_not_verified")
    image = official.get("image_reference")
    if not image_reference_valid(image):
        blockers.append("delivery_image_reference_missing_or_invalid")
    digest = official.get("image_digest")
    if digest is None:
        advisories.append("optional_image_digest_missing")
    elif not isinstance(digest, str) or not IMAGE_DIGEST.fullmatch(digest):
        advisories.append("optional_image_digest_invalid")
    if not artifacts["verified"]:
        blockers.append("required_artifacts_not_verified")
    if not model["valid"]:
        blockers.append("model_architecture_not_verified")
    if config["accuracy"].get("baseline") is None:
        blockers.append("official_baseline_metric_missing")
    budget = config["budget"]
    estimate = budget_estimate(
        budget["cards"], budget["wall_hours"], budget["total_card_hours"], budget.get("used_card_hours", 0)
    )
    if not estimate["within_budget"]:
        blockers.append("estimated_card_hours_exceed_remaining_budget")
    return {
        "task": config["slug"],
        "ready": not blockers,
        "readiness_scope": "local prerequisites only; remote launch and official scoring are separate",
        "official_evaluation": "not_run",
        "blockers": blockers,
        "advisories": advisories,
        "image": {"reference": image, "reference_syntax_valid": image_reference_valid(image),
                  "digest": digest, "registry_access_verified": False,
                  "note": "The template requires a complete image address and tag; digest and claim receipt are not mandatory gates."},
        "model": model,
        "artifacts": artifacts,
        "accuracy": {
            key: config["accuracy"].get(key)
            for key in ("metric", "direction", "drop_mode", "max_drop", "full_scale", "baseline")
        },
        "budget": estimate,
    }
