"""Preparation stays local by default and refuses native-stack drift before mutation."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("asr_prepare_under_test", ROOT / "scripts/prepare_qwen_asr_task.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
TASK = ROOT / "tasks/qwen3-asr-bw-vllm"


class ASRPreparationTests(unittest.TestCase):
    def test_default_plan_does_not_install_create_or_inspect_native_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "not-created"
            output = io.StringIO()
            with patch.object(prepare.subprocess, "run", side_effect=AssertionError("No subprocess in plan")), \
                 patch.object(prepare.venv, "EnvBuilder", side_effect=AssertionError("No venv in plan")), \
                 patch.object(prepare.md, "distribution", side_effect=AssertionError("No native inspection in plan")), \
                 redirect_stdout(output):
                self.assertEqual(prepare.main(["--venv", str(target)]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["mode"], "plan_only")
            self.assertFalse(report["service_started"])
            self.assertFalse(report["filesystem_mutated"])
            self.assertFalse(target.exists())

    def test_lock_is_exactly_the_seven_previously_reviewed_wheels(self):
        expected = json.loads((ROOT / "evidence/asr-audio-reviewed-plan-002.json").read_text(encoding="utf-8"))
        actual = [{key: value for key, value in item.items() if key != "requirement"}
                  for item in prepare.read_lock(TASK)]
        self.assertEqual(actual, expected["selected"])
        self.assertEqual(json.loads((TASK / "native-stack.json").read_text()),
                         {name: item["version"] for name, item in expected["protected"].items()})

    def test_drift_is_rejected_before_creating_venv_or_running_pip(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "not-created"
            with patch.object(prepare, "target_platform"), \
                 patch.object(prepare.sys, "prefix", "base"), \
                 patch.object(prepare.sys, "base_prefix", "base"), \
                 patch.object(prepare, "native_snapshot", side_effect=ValueError("Native torch differs")), \
                 patch.object(prepare.venv, "EnvBuilder") as builder, \
                 patch.object(prepare.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "Native torch differs"):
                    prepare.apply(TASK, target)
                builder.assert_not_called()
                run.assert_not_called()
            self.assertFalse(target.exists())

    def test_existing_and_relative_targets_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "new absolute"):
                prepare.check_target(Path(temporary))
        with self.assertRaisesRegex(ValueError, "new absolute"):
            prepare.check_target(Path("relative-audio-venv"))

    def test_rejects_unhashed_or_non_pypi_requirement(self):
        for line in ("librosa==0.11.0", "librosa @ https://example.com/file.whl --hash=sha256:" + "a" * 64):
            with tempfile.TemporaryDirectory() as temporary:
                task = Path(temporary)
                (task / "requirements-extra.txt").write_text(line + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "reviewed PyPI"):
                    prepare.read_lock(task)


if __name__ == "__main__":
    unittest.main()
