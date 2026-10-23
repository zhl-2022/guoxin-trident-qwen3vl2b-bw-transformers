"""Offline Bash entry regressions; never import a serving framework or inspect GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/recon2reason-bw-vllm"


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "Linux Bash entry; run in WSL")
class ReconTaskEntryTests(unittest.TestCase):
    def run_entry(self, extra_env=None):
        env = {key: value for key, value in os.environ.items() if not key.startswith("RECON2REASON_")}
        # These represent another sourced task. All are intentionally wrong for Recon.
        env.update(MODEL_ID="Qwen/Qwen3-ASR-1.7B", MODEL_PATH="/another-model", SERVED_NAME="wrong-model",
                   FRAMEWORK="sglang", PRECISION="fp16", CARD="another-card", IMAGE="other:image",
                   PORT="30002", TP="8", MAX_LEN="65536", MEM_FRAC="0.2", LOAD_TIMEOUT="1",
                   MAX_RUNTIME_SECONDS="1", STOP_GRACE_SECONDS="1", EXTRA_ARGS_JSON='["--wrong"]',
                   PYTHON_BIN="/not/a/python", OFFICIAL_VERIFY_SCRIPT="/other/verify.py",
                   OFFICIAL_BENCH_SCRIPT="/other/bench.sh", OFFICIAL_BASELINE="/other/baseline.json",
                   OFFICIAL_DATA_DIR="/other/data", IMAGE_RES="other-resolution",
                   RECON2REASON_PYTHON_BIN=sys.executable)
        env.update(extra_env or {})
        return subprocess.run(["bash", str(TASK / "run.sh"), "--dry-run"], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=15)

    def test_default_command_matches_recorded_success_despite_other_task_environment(self):
        completed = self.run_entry()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        config = json.loads(completed.stdout)
        recorded = json.loads((ROOT / "evidence/vllm-state-001.json").read_text(encoding="utf-8"))
        self.assertEqual(config["command"][1:], recorded["command"][1:])
        self.assertTrue(config["dry_run"])
        self.assertEqual(config["model_id"], "BAAI/Recon2Reason-Reasoning-4B")
        self.assertEqual(config["framework"], "vllm")
        self.assertEqual(config["precision"], "bf16")
        self.assertEqual(config["load_timeout"], 900)
        self.assertEqual(config["max_runtime"], 4500)
        self.assertEqual(config["stop_grace"], 30)

    def test_explicit_task_overrides_work_without_mutating_historical_defaults(self):
        completed = self.run_entry({"RECON2REASON_MODEL_PATH": "/new mount/model",
                                    "RECON2REASON_SERVED_NAME": "recon-test",
                                    "RECON2REASON_PORT": "31000",
                                    "RECON2REASON_MAX_RUNTIME_SECONDS": "600"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        config = json.loads(completed.stdout)
        self.assertEqual(config["model_path"], "/new mount/model")
        self.assertEqual(config["served_name"], "recon-test")
        self.assertEqual(config["url"], "http://127.0.0.1:31000")
        self.assertEqual(config["max_runtime"], 600)
        self.assertEqual(json.loads(config["extra"][-1]), {"image": 2, "video": 0})

    def test_other_tasks_official_paths_and_image_identity_are_not_inherited(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith("RECON2REASON_")}
        env.update(OFFICIAL_VERIFY_SCRIPT="/other/verify.py", OFFICIAL_BENCH_SCRIPT="/other/bench.sh",
                   OFFICIAL_BASELINE="/other/baseline.json", OFFICIAL_DATA_DIR="/other/data",
                   IMAGE_RES="999", IMAGE="other:image", IMAGE_CANDIDATE_NAME="other-display")
        keys = ("OFFICIAL_VERIFY_SCRIPT", "OFFICIAL_BENCH_SCRIPT", "OFFICIAL_BASELINE", "OFFICIAL_DATA_DIR",
                "IMAGE_RES", "IMAGE", "IMAGE_CANDIDATE_NAME")
        code = "import json,os;print(json.dumps({k:os.environ.get(k) for k in " + repr(keys) + "}))"
        completed = subprocess.run(["bash", "-c", 'source "$1"; exec "$2" -c "$3"', "recon-env",
                                    str(TASK / "env.sh"), sys.executable, code], env=env,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        values = json.loads(completed.stdout)
        for key in keys[:-1]:
            self.assertEqual(values[key], "", key)
        self.assertEqual(values["IMAGE_CANDIDATE_NAME"],
                         "jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423")

    def test_core_override_in_extra_array_fails_before_any_launch(self):
        completed = self.run_entry({"RECON2REASON_EXTRA_ARGS_JSON": '["--dtype","float16"]'})
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot override", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
