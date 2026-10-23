import copy
import hashlib
import json
import os
import tempfile
import unittest
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from guoxin.__main__ import main
from guoxin.config import EXPECTED_ARCHITECTURE, image_reference_valid, load_task, project_path, task_plan, validate_model, verify_artifacts
from guoxin.gates import accuracy_gate, budget_estimate
from guoxin.runtime import OUTPUT_LIMIT, export_manifest, probe


class AccuracyTests(unittest.TestCase):
    def gate(self, baseline, candidate, **kwargs):
        options = {"direction": "higher", "max_drop": "0.1", "drop_mode": "full_scale_points"}
        options.update(kwargs)
        return accuracy_gate(baseline, candidate, **options)

    def test_full_scale_points_are_not_baseline_relative(self):
        self.assertTrue(self.gate("0.8", "0.799")["passed"])
        self.assertFalse(self.gate("0.8", "0.799", drop_mode="relative", max_drop="0.001")["passed"])

    def test_decimal_boundary_is_exact(self):
        self.assertTrue(self.gate("0.9", "0.899")["passed"])
        self.assertFalse(self.gate("0.9", "0.8989999999999")["passed"])

    def test_lower_metric_and_improvements(self):
        self.assertTrue(self.gate(20, 20.2, direction="lower", drop_mode="relative", max_drop="0.01")["passed"])
        self.assertFalse(self.gate(20, 20.2001, direction="lower", drop_mode="relative", max_drop="0.01")["passed"])
        self.assertTrue(self.gate("0.8", "0.9")["passed"])

    def test_relative_zero_requires_exact_zero(self):
        for direction in ("higher", "lower"):
            self.assertTrue(self.gate(0, 0, direction=direction, drop_mode="relative")["passed"])
            self.assertFalse(self.gate(0, "0.000000000001", direction=direction, drop_mode="relative")["passed"])
            self.assertFalse(self.gate(0, -1, direction=direction, drop_mode="relative")["passed"])

    def test_invalid_numbers_and_thresholds_rejected(self):
        for bad in ("NaN", "Infinity", float("nan"), float("inf"), True, None, "abc"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.gate(bad, "0.8")
        for options in ({"max_drop": -1}, {"direction": "up"}, {"full_scale": 0}, {"drop_mode": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.gate("0.8", "0.8", **options)
        with self.assertRaises(ValueError):
            self.gate("1.1", "0.9")


class BudgetTests(unittest.TestCase):
    def test_card_hours_and_exhaustion(self):
        result = budget_estimate(8, "1.5", 500, 490)
        self.assertEqual(result["estimated_card_hours"], "12.0")
        self.assertFalse(result["within_budget"])
        self.assertTrue(budget_estimate(2, 5, 500, 490)["within_budget"])

    def test_invalid_resource_estimates(self):
        for args in ((1.5, 1, 500), (0, 1, 500), (1, -1, 500), (True, 1, 500), (1, 1, "NaN"), (1, 1, 5, 6)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                budget_estimate(*args)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.model = self.project / "config.json"
        self.model.write_text(json.dumps({"architectures": [EXPECTED_ARCHITECTURE], "model_type": "qwen3_vl"}), encoding="utf-8")
        self.artifact = self.project / "official.bin"
        self.artifact.write_bytes(b"official test fixture")
        self.config = {
            "schema_version": 1,
            "slug": "qwen3-vl-test",
            "model": {"config_path": "config.json"},
            "artifacts": [{"name": "official", "path": "official.bin", "sha256": hashlib.sha256(self.artifact.read_bytes()).hexdigest()}],
            "official": {
                "claim": {"confirmed": True, "reference": "fixture-only"},
                "baseline": {"artifact": "official"},
                "evaluator": {"artifact": "official"},
                "dataset": {"artifact": "official"},
                "image_reference": "registry.example/native/vllm:dtk26.04",
                "image_digest": "sha256:" + "a" * 64,
            },
            "accuracy": {"metric": "accuracy", "direction": "higher", "drop_mode": "full_scale_points", "max_drop": "0.1", "full_scale": 1, "baseline": "0.8"},
            "budget": {"cards": 1, "wall_hours": 1, "total_card_hours": 500},
            "evidence_files": [],
        }
        self.task = self.project / "task.json"
        self.save_config()

    def save_config(self):
        self.task.write_text(json.dumps(self.config), encoding="utf-8")

    def test_model_architecture_detects_real_family_mismatch(self):
        self.assertTrue(validate_model(self.project, "config.json")["valid"])
        self.model.write_text(json.dumps({"architectures": ["Qwen2_5_VLForConditionalGeneration"]}), encoding="utf-8")
        self.assertFalse(validate_model(self.project, "config.json")["valid"])

    def test_asr_requires_explicit_architecture_and_does_not_pass_as_vl(self):
        architecture = "Qwen3ASRForConditionalGeneration"
        self.model.write_text(json.dumps({"architectures": [architecture], "model_type": "qwen3_asr"}), encoding="utf-8")
        self.assertFalse(validate_model(self.project, "config.json")["valid"])
        self.assertTrue(validate_model(self.project, "config.json", architecture)["valid"])
        self.config["model"]["expected_architecture"] = architecture
        self.save_config()
        config, _ = load_task(self.project, "task.json")
        self.assertTrue(task_plan(self.project, config)["model"]["valid"])
        self.config["model"]["expected_architecture"] = "UnverifiedModelClass"
        self.save_config()
        with self.assertRaises(ValueError):
            load_task(self.project, "task.json")

    def test_artifacts_fail_closed(self):
        self.assertTrue(verify_artifacts(self.project, self.config)["verified"])
        self.artifact.write_bytes(b"modified")
        self.assertFalse(verify_artifacts(self.project, self.config)["verified"])
        self.config["artifacts"][0]["sha256"] = None
        self.assertEqual(verify_artifacts(self.project, self.config)["files"][0]["reason"], "expected_sha256_missing")
        self.artifact.unlink()
        self.assertEqual(verify_artifacts(self.project, self.config)["files"][0]["reason"], "missing_regular_file")

    def test_missing_official_prerequisites_prevent_readiness(self):
        self.assertTrue(task_plan(self.project, self.config)["ready"])
        for key in ("baseline", "evaluator", "dataset", "image_reference"):
            config = copy.deepcopy(self.config)
            config["official"].pop(key)
            with self.subTest(key=key):
                self.assertFalse(task_plan(self.project, config)["ready"])
        self.config["accuracy"]["baseline"] = None
        self.assertFalse(task_plan(self.project, self.config)["ready"])

    def test_claim_receipt_and_digest_are_advisory_with_complete_image_tag(self):
        config = copy.deepcopy(self.config)
        config["official"].pop("claim")
        config["official"].pop("image_digest")
        result = task_plan(self.project, config)
        self.assertTrue(result["ready"])
        self.assertIn("task_priority_not_independently_verified", result["advisories"])
        self.assertIn("optional_image_digest_missing", result["advisories"])
        self.assertFalse(result["image"]["registry_access_verified"])
        self.assertEqual(result["official_evaluation"], "not_run")
        config["official"]["image_digest"] = "wrong"
        self.assertIn("optional_image_digest_invalid", task_plan(self.project, config)["advisories"])

    def test_image_reference_needs_full_path_and_explicit_version(self):
        for value in ("registry.example:5443/hcu/vllm:dtk26.04", "docker.io/org/image@sha256:" + "a" * 64):
            with self.subTest(value=value):
                self.assertTrue(image_reference_valid(value))
        for value in (None, "", "jupyter-vllm:latest", "registry.example/image", "registry.example:5443/image", "https://site/image:tag", "registry.example/image:tag?token=x", "registry.example/image: tag", "registry.example//image:tag", "registry.example/image@sha256:bad"):
            with self.subTest(value=value):
                self.assertFalse(image_reference_valid(value))

    def test_traversal_and_external_paths_rejected(self):
        for value in ("../secret.txt", str(self.project.parent / "secret.txt")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project_path(self.project, value)
        self.config["evidence_files"] = ["../secret.txt"]
        self.save_config()
        with self.assertRaises(ValueError):
            load_task(self.project, "task.json")

    def test_symlink_escape_rejected_where_supported(self):
        with tempfile.TemporaryDirectory() as outside:
            link = self.project / "link"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation unavailable on this host")
            with self.assertRaises(ValueError):
                project_path(self.project, "link/secret.txt")

    def test_manifest_includes_only_explicit_files(self):
        (self.project / "unlisted-secret.txt").write_text("do not collect", encoding="utf-8")
        result = export_manifest(self.project, self.config, self.task)
        self.assertTrue(result["complete"])
        self.assertEqual({row["path"] for row in result["files"]}, {"task.json", "official.bin", "config.json"})
        self.assertNotIn("do not collect", json.dumps(result))
        self.config["evidence_files"] = ["missing.txt"]
        self.assertFalse(export_manifest(self.project, self.config, self.task)["complete"])

    def test_json_nan_and_duplicate_artifacts_rejected(self):
        self.task.write_text('{"score":NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_task(self.project, "task.json")
        self.task.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_task(self.project, "task.json")
        self.config["artifacts"].append(dict(self.config["artifacts"][0]))
        self.save_config()
        with self.assertRaises(ValueError):
            load_task(self.project, "task.json")

    def test_cli_exit_status_and_refuses_evidence_overwrite(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(["--project", str(self.project), "inspect-config", "--task", "task.json"]), 0)
            self.assertEqual(main(["--project", str(self.project), "plan", "--task", "task.json"]), 0)
            self.assertEqual(main(["--project", str(self.project), "budget", "--cards", "8", "--wall-hours", "100", "--total-card-hours", "500"]), 2)
            args = ["--project", str(self.project), "export-manifest", "--task", "task.json", "--output", "evidence.json"]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 1)
        self.config["official"]["claim"]["confirmed"] = False
        self.save_config()
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(["--project", str(self.project), "plan", "--task", "task.json"]), 0)


class EnvironmentTests(unittest.TestCase):
    @patch("guoxin.runtime.shutil.which", return_value=None)
    @patch("guoxin.runtime.subprocess.run")
    def test_missing_probe_never_launches_subprocess(self, run, which):
        self.assertEqual(probe(["missing-tool"])["status"], "unavailable")
        run.assert_not_called()

    @patch("guoxin.runtime.shutil.which", return_value="/bin/tool")
    @patch("guoxin.runtime.subprocess.run", side_effect=subprocess.TimeoutExpired("tool", 2))
    def test_probe_timeout_is_bounded_and_reported(self, run, which):
        self.assertEqual(probe(["tool"], timeout=2)["status"], "timeout")
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        self.assertNotIn("shell", run.call_args.kwargs)

    @patch("guoxin.runtime.shutil.which", return_value="/bin/tool")
    @patch("guoxin.runtime.subprocess.run")
    def test_probe_output_is_truncated(self, run, which):
        run.return_value = subprocess.CompletedProcess(["tool"], 0, "x" * (OUTPUT_LIMIT + 1), "")
        result = probe(["tool"])
        self.assertTrue(result["output_truncated"])
        self.assertEqual(len(result["stdout"]), OUTPUT_LIMIT)


if __name__ == "__main__":
    unittest.main()
