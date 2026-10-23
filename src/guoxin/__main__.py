"""Run with PYTHONPATH=src python -m guoxin; JSON stdout, nonzero on failed gates."""

import argparse
import json
import sys
from pathlib import Path

from .config import EXPECTED_ARCHITECTURE, SUPPORTED_ARCHITECTURES, load_task, project_path, task_plan, validate_model, verify_artifacts
from .gates import accuracy_gate, budget_estimate
from .runtime import collect_environment, export_manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Competition preparation and evidence checks")
    result.add_argument("--project", default=".", help="project root (default: current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("inspect-config", "plan", "verify-artifacts", "export-manifest"):
        command = commands.add_parser(name)
        command.add_argument("--task", required=True, help="explicit task JSON path under project")
        command.add_argument("--output", help="new JSON output path under project; never overwrites")
    model = commands.add_parser("validate-model")
    model.add_argument("--config", required=True, help="local model config.json under project")
    model.add_argument("--expected-architecture", choices=sorted(SUPPORTED_ARCHITECTURES), default=EXPECTED_ARCHITECTURE)
    model.add_argument("--output")
    environment = commands.add_parser("collect-env")
    environment.add_argument("--output")
    accuracy = commands.add_parser("accuracy-gate")
    accuracy.add_argument("--task", help="threshold and baseline configuration")
    accuracy.add_argument("--baseline", help="override numeric baseline")
    accuracy.add_argument("--candidate", required=True)
    accuracy.add_argument("--direction", choices=("higher", "lower"))
    accuracy.add_argument("--drop-mode", choices=("full_scale_points", "relative", "absolute"))
    accuracy.add_argument("--max-drop")
    accuracy.add_argument("--full-scale")
    accuracy.add_argument("--metric")
    accuracy.add_argument("--output")
    budget = commands.add_parser("budget")
    budget.add_argument("--task")
    budget.add_argument("--cards")
    budget.add_argument("--wall-hours")
    budget.add_argument("--total-card-hours")
    budget.add_argument("--used-card-hours")
    budget.add_argument("--output")
    return result


def emit(value: dict, project: Path, output: str | None) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        path = project_path(project, output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    print(payload, end="")


def execute(args: argparse.Namespace) -> tuple[dict, int]:
    project = Path(args.project).resolve(strict=True)
    config = None
    task_path = None
    if getattr(args, "task", None):
        config, task_path = load_task(project, args.task)
    if args.command == "inspect-config":
        return {
            "valid": True,
            "schema_version": config["schema_version"],
            "task": config["slug"],
            "artifact_count": len(config["artifacts"]),
            "evidence_file_count": len(config.get("evidence_files", [])),
            "note": "schema validation does not establish readiness or official acceptance",
        }, 0
    if args.command == "plan":
        result = task_plan(project, config)
        return result, 0 if result["ready"] else 2
    if args.command == "validate-model":
        result = validate_model(project, args.config, args.expected_architecture)
        return result, 0 if result["valid"] else 2
    if args.command == "verify-artifacts":
        result = verify_artifacts(project, config)
        return result, 0 if result["verified"] else 2
    if args.command == "collect-env":
        return collect_environment(), 0
    if args.command == "export-manifest":
        result = export_manifest(project, config, task_path)
        return result, 0 if result["complete"] else 2
    if args.command == "accuracy-gate":
        options = dict(config["accuracy"]) if config else {}
        for key in ("baseline", "direction", "drop_mode", "max_drop", "full_scale", "metric"):
            value = getattr(args, key)
            if value is not None:
                options[key] = value
        options.setdefault("full_scale", 1)
        options.setdefault("metric", "score")
        required = ("baseline", "direction", "max_drop", "drop_mode")
        if any(options.get(key) is None for key in required):
            raise ValueError("accuracy-gate requires baseline, direction, max_drop and drop_mode")
        result = accuracy_gate(candidate=args.candidate, **{key: options[key] for key in (*required, "full_scale", "metric")})
        return result, 0 if result["passed"] else 2
    if args.command == "budget":
        options = dict(config["budget"]) if config else {}
        for key in ("cards", "wall_hours", "total_card_hours", "used_card_hours"):
            value = getattr(args, key)
            if value is not None:
                options[key] = value
        options.setdefault("used_card_hours", 0)
        required = ("cards", "wall_hours", "total_card_hours", "used_card_hours")
        if any(options.get(key) is None for key in required):
            raise ValueError("budget requires cards, wall_hours and total_card_hours")
        result = budget_estimate(**{key: options[key] for key in required})
        return result, 0 if result["within_budget"] else 2
    raise ValueError("unknown command")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result, code = execute(args)
        emit(result, Path(args.project), args.output)
        return code
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
