"""Run the actual T04 Bash/stdlib bridge in temporary CPU-only projects."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "Linux Bash entry; run in WSL")
class ASROfficialEntryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.task = self.project / "tasks/qwen3-asr-bw-vllm"
        self.task.mkdir(parents=True)
        (self.project / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/recon2reason_runtime.py", self.project / "scripts/recon2reason_runtime.py")
        for name in ("env.sh", "verify.sh", "bench.sh"):
            shutil.copyfile(ROOT / "tasks/qwen3-asr-bw-vllm" / name, self.task / name)
        (self.task / "task.json").write_text('{"track":"T04"}')
        self.evaluator = self.task / "external.py"
        self.evaluator.write_text("import sys\nfrom pathlib import Path\n"
                                  "assert '--selfcheck' in sys.argv\n"
                                  "Path('only-selfcheck-ran').write_text('yes')\n"
                                  "print('external selfcheck original stdout')\n")
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("QWEN_ASR_", "OFFICIAL_"))}
        self.env.update(PYTHONDONTWRITEBYTECODE="1", QWEN_ASR_PYTHON_BIN=sys.executable)

    def run_entry(self, name, *args):
        return subprocess.run(["bash", str(self.task / name), *args], cwd=self.project,
                              env=self.env, capture_output=True, text=True, timeout=15)

    def test_selfcheck_only_executes_external_script_without_venv_baseline_data_or_server(self):
        self.env["QWEN_ASR_OFFICIAL_VERIFY_SCRIPT"] = str(self.evaluator)
        completed = self.run_entry("verify.sh", "--selfcheck")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "selfcheck_executed_uninterpreted")
        self.assertEqual((self.task / "only-selfcheck-ran").read_text(), "yes")
        self.assertFalse((self.task / "verify_result.json").exists())
        records = list((self.task / "runtime").glob("official-selfcheck-*/record.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(len(record["steps"]), 1)
        self.assertFalse(record["service_health_checked"])
        self.assertFalse(record["official_acceptance"])

    def test_missing_materials_are_reported_without_writing_runtime(self):
        for name, args in (("verify.sh", ("--selfcheck", "--dry-run")),
                           ("verify.sh", ("--dry-run",)), ("bench.sh", ("--dry-run",))):
            with self.subTest(name=name, args=args):
                completed = self.run_entry(name, *args)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertTrue(json.loads(completed.stdout)["problems"])
        self.assertFalse((self.task / "runtime").exists())

    def test_full_verify_and_audio_bench_use_explicit_task_scoped_contracts(self):
        baseline = self.task / "baseline.json"
        baseline.write_text('{"unfamiliar":true}')
        data = self.task / "data"
        data.mkdir()
        (data / "manifest.txt").write_text("CPU fixture only")
        self.env.update(QWEN_ASR_OFFICIAL_VERIFY_SCRIPT=str(self.evaluator),
                        QWEN_ASR_OFFICIAL_BASELINE=str(baseline), QWEN_ASR_OFFICIAL_DATA_DIR=str(data),
                        QWEN_ASR_OFFICIAL_VERIFY_ARGS_JSON='["--audio-manifest","explicit.jsonl"]',
                        QWEN_ASR_OFFICIAL_BENCH_SCRIPT=str(self.evaluator),
                        QWEN_ASR_OFFICIAL_BENCH_OUTPUTS_JSON='["audio/result.json"]')
        completed = self.run_entry("verify.sh", "--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["command"][-2:], ["--audio-manifest", "explicit.jsonl"])
        self.assertEqual(plan["selfcheck_command"][-1], "--selfcheck")
        completed = self.run_entry("bench.sh", "--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["expected_outputs"], ["audio/result.json"])
        self.assertNotIn("template_requests_total", plan)
        self.assertFalse((self.task / "runtime").exists())

    def test_sourced_t02_arguments_outputs_and_image_mode_do_not_leak(self):
        keys = ("OFFICIAL_VERIFY_ARGS_JSON", "OFFICIAL_BENCH_ARGS_JSON", "OFFICIAL_SELFCHECK_ARGS_JSON",
                "OFFICIAL_VERIFY_OUTPUTS_JSON", "OFFICIAL_BENCH_OUTPUTS_JSON", "BENCH_MODE")
        self.env.update({key: "inherited-wrong-value" for key in keys})
        code = "import json,os;print(json.dumps({k:os.environ.get(k) for k in " + repr(keys) + "}))"
        completed = subprocess.run(["bash", "-c", 'source "$1"; exec "$2" -c "$3"', "asr-env",
                                    str(self.task / "env.sh"), sys.executable, code], env=self.env,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        values = json.loads(completed.stdout)
        self.assertEqual(values["OFFICIAL_VERIFY_ARGS_JSON"], "[]")
        self.assertEqual(values["OFFICIAL_BENCH_ARGS_JSON"], "[]")
        self.assertEqual(json.loads(values["OFFICIAL_SELFCHECK_ARGS_JSON"]), ["--selfcheck"])
        for key in keys[3:]:
            self.assertIsNone(values[key], key)


if __name__ == "__main__":
    unittest.main()
