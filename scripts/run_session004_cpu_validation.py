#!/usr/bin/env python3
"""Run current CPU unittest discovery with raw logs and source-snapshot evidence."""
from __future__ import annotations
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    result = {}
    for folder in ("src", "scripts", "tests", "tasks"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and not set(path.relative_to(ROOT).parts) & {"runtime", "__pycache__", ".git"} and path.suffix in {".py", ".sh", ".json"}:
                result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--exclude-module", action="append", default=[])
    args = parser.parse_args()
    prefix = args.output_prefix.resolve()
    log_path = prefix.with_suffix(".log")
    record_path = prefix.with_suffix(".json")
    if log_path.exists() or record_path.exists():
        parser.error("Validation output exists; choose a new prefix and preserve previous runs")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")])
    before = source_hashes()
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        discovered = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        tests = list(flatten(discovered))
        # Discovery wraps an unavailable import as unittest.loader._FailedTest;
        # its method name retains the original module instead of its first ID component.
        def module_name(test):
            return test.id().rsplit(".", 1)[-1] if test.__class__.__name__ == "_FailedTest" else test.id().split(".")[0]

        excluded = [test.id() for test in tests if module_name(test) in args.exclude_module]
        selected = [test for test in tests if test.id() not in excluded]
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(selected))
    after = source_hashes()
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    record = {"schema_version": 1, "session": "session004", "host": args.host_label,
              "python": sys.executable, "python_version": sys.version, "platform": sys.platform,
              "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": round(time.monotonic() - start, 6), "discovery_pattern": "test_*.py",
              "discovered_tests": len(tests), "selected_test_ids": [test.id() for test in selected],
              "excluded_modules": args.exclude_module, "excluded_test_ids": excluded,
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
              "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
              "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped) - len(result.expectedFailures),
              "successful": result.wasSuccessful(), "full_discovered_suite_selected": not excluded,
              "gpu_run": False, "official_evaluation_run": False,
              "source_hashes_before": before, "source_hashes_after": after,
              "changed_sources_during_run": changed, "source_snapshot_stable": not changed,
              "raw_log": str(log_path), "raw_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()}
    with record_path.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({key: record[key] for key in ("host", "tests_run", "passed", "failures", "errors", "successful", "full_discovered_suite_selected", "source_snapshot_stable")}, ensure_ascii=False))
    print(str(record_path))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
