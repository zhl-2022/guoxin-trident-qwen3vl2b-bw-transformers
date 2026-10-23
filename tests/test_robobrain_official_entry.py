"""RoboBrain official entry contracts cannot inherit another task's arguments."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/robobrain-bw-vllm"
KEYS = ("OFFICIAL_VERIFY_ARGS_JSON", "OFFICIAL_BENCH_ARGS_JSON", "OFFICIAL_SELFCHECK_ARGS_JSON",
        "OFFICIAL_VERIFY_OUTPUTS_JSON", "OFFICIAL_BENCH_OUTPUTS_JSON", "OFFICIAL_TIMEOUT_SECONDS",
        "OFFICIAL_SELFCHECK_TIMEOUT_SECONDS", "BENCH_MODE")


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "Linux Bash entry; run in WSL")
class RoboOfficialEntryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / "tasks/robobrain-bw-vllm"
        self.task.mkdir(parents=True)
        (self.root / "scripts").mkdir()
        for name in ("env.sh", "verify.sh", "bench.sh"):
            shutil.copyfile(TASK / name, self.task / name)
        shutil.copyfile(ROOT / "scripts/recon2reason_runtime.py", self.root / "scripts/recon2reason_runtime.py")
        (self.task / "task.json").write_text('{"track":"T02"}')
        self.script = self.task / "external.py"
        self.script.write_text("raise RuntimeError('dry-run must not execute me')\n")
        self.baseline = self.task / "baseline.json"
        self.baseline.write_text('{"unknown_metric_schema":true}')
        self.data = self.task / "data"
        self.data.mkdir()
        (self.data / "fixture.txt").write_text("CPU fixture")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ROBOBRAIN_", "OFFICIAL_"))}
        self.env.update({k: "inherited-from-another-task" for k in KEYS})
        self.env.update(ROBOBRAIN_PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1")

    def source_values(self, prefix_source=None):
        code = "import os,json;print(json.dumps({k:os.environ.get(k) for k in " + repr(KEYS) + "}))"
        shell = 'source "$1"; exec "$2" -c "$3"'
        argv = ["bash", "-c", shell, "robo-env", str(self.task / "env.sh"), sys.executable, code]
        if prefix_source:
            argv = ["bash", "-c", 'source "$1"; source "$2"; exec "$3" -c "$4"', "robo-after-recon",
                    str(prefix_source), str(self.task / "env.sh"), sys.executable, code]
        run = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(run.stdout)

    def test_recon_and_generic_evaluator_arguments_do_not_leak(self):
        for prefix in (None, ROOT / "tasks/recon2reason-bw-vllm/env.sh"):
            with self.subTest(prefix=str(prefix)):
                values = self.source_values(prefix)
                self.assertEqual(json.loads(values["OFFICIAL_VERIFY_ARGS_JSON"]), [])
                self.assertEqual(json.loads(values["OFFICIAL_BENCH_ARGS_JSON"]), [])
                self.assertEqual(json.loads(values["OFFICIAL_SELFCHECK_ARGS_JSON"]), ["--selfcheck"])
                self.assertEqual(values["OFFICIAL_TIMEOUT_SECONDS"], "7200")
                self.assertEqual(values["OFFICIAL_SELFCHECK_TIMEOUT_SECONDS"], "120")
                for key in ("OFFICIAL_VERIFY_OUTPUTS_JSON", "OFFICIAL_BENCH_OUTPUTS_JSON", "BENCH_MODE"):
                    self.assertIsNone(values[key])

    def test_explicit_robo_scoped_contracts_survive_source_exactly(self):
        overrides = {"OFFICIAL_VERIFY_ARGS_JSON": '["--split","robo-fixed"]',
                     "OFFICIAL_BENCH_ARGS_JSON": '["--shape","robo-image"]',
                     "OFFICIAL_SELFCHECK_ARGS_JSON": '["--audit"]',
                     "OFFICIAL_VERIFY_OUTPUTS_JSON": '["robo/verify.json"]',
                     "OFFICIAL_BENCH_OUTPUTS_JSON": '["robo/bench.json"]',
                     "OFFICIAL_TIMEOUT_SECONDS": "90", "OFFICIAL_SELFCHECK_TIMEOUT_SECONDS": "10"}
        self.env.update({"ROBOBRAIN_"+k: v for k, v in overrides.items()})
        values = self.source_values(ROOT / "tasks/recon2reason-bw-vllm/env.sh")
        for key, value in overrides.items():
            self.assertEqual(values[key], value, key)
        self.assertIsNone(values["BENCH_MODE"])

    def test_verify_and_bench_wrappers_use_robo_contracts_not_inherited_outputs(self):
        self.env.update(ROBOBRAIN_OFFICIAL_VERIFY_SCRIPT=str(self.script),
                        ROBOBRAIN_OFFICIAL_BENCH_SCRIPT=str(self.script),
                        ROBOBRAIN_OFFICIAL_BASELINE=str(self.baseline),
                        ROBOBRAIN_OFFICIAL_DATA_DIR=str(self.data),
                        ROBOBRAIN_OFFICIAL_VERIFY_ARGS_JSON='["--split","robo-fixed"]')
        for kind in ("verify", "bench"):
            run = subprocess.run(["bash", str(self.task / (kind+".sh")), "--dry-run"], cwd=self.root,
                                 env=self.env, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            plan = json.loads(run.stdout)
            self.assertEqual(plan["problems"], [])
            self.assertEqual(plan["track"], "T02")
            if kind == "verify":
                self.assertEqual(plan["expected_outputs"], ["verify_result.json"])
                self.assertEqual(plan["command"][-2:], ["--split", "robo-fixed"])
            else:
                self.assertEqual(plan["template_requests_total"], 608)
                self.assertEqual(len(plan["expected_outputs"]), 15)
        self.assertFalse((self.task / "runtime").exists())


if __name__ == "__main__":
    unittest.main()
