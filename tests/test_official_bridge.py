"""CPU-only external-material bridge tests using temporary evaluator fixtures."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recon2reason_runtime.py"
SPEC = importlib.util.spec_from_file_location("official_bridge_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class OfficialBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task = Path(self.temporary.name).resolve()
        self.baseline = self.task / "baseline.yaml"
        self.baseline.write_text("unfamiliar_schema: baseline_from_external_source\n", encoding="utf-8")
        self.data = self.task / "data"
        self.data.mkdir()
        (self.data / "fixture.txt").write_text("temporary CPU mock data", encoding="utf-8")
        self.script = self.task / "evaluator.py"
        self.env = {"PATH": os.environ.get("PATH", ""), "MODEL_ID": "fixture/model", "CARD": "BW",
                    "FRAMEWORK": "vllm", "PRECISION": "bf16", "OFFICIAL_VERIFY_SCRIPT": str(self.script),
                    "OFFICIAL_BASELINE": str(self.baseline), "OFFICIAL_DATA_DIR": str(self.data),
                    "OFFICIAL_TIMEOUT_SECONDS": "5", "OFFICIAL_SELFCHECK_TIMEOUT_SECONDS": "5",
                    "PYTHONDONTWRITEBYTECODE": "1"}
        self.make_script()

    def make_script(self, selfcheck="", evaluation=""):
        self.script.write_text("import sys,json,os,time,subprocess\nfrom pathlib import Path\n"
                               "p=Path('events.txt')\n"
                               "phase='selfcheck' if '--selfcheck' in sys.argv or '--audit' in sys.argv else 'evaluation'\n"
                               "with p.open('a') as f: f.write(phase+'\\n')\n"
                               "if phase=='selfcheck':\n" +
                               "\n".join("    " + line for line in (selfcheck or "pass").splitlines()) + "\n"
                               "else:\n" +
                               "\n".join("    " + line for line in (evaluation or
                                   "Path('verify_result.json').write_text(json.dumps({'new_metric':0.75,'passed':True}))").splitlines()) + "\n",
                               encoding="utf-8")

    def execute(self, kind="verify", dry_run=False, selfcheck_only=False):
        self.stdout = io.StringIO()
        with patch.dict(os.environ, self.env, clear=True), patch.object(runtime, "healthy", return_value=True) as health:
            with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(io.StringIO()):
                code = runtime.official(self.task, kind, dry_run, selfcheck_only)
            self.health_calls = health.call_count
        return code

    def record(self):
        records = list((self.task / "runtime").glob("official-*/record.json"))
        self.assertEqual(len(records), 1)
        return json.loads(records[0].read_text(encoding="utf-8"))

    def events(self):
        return (self.task / "events.txt").read_text().splitlines()

    def interrupt_wait_for_phase(self, phase):
        """Interrupt a real CPU child after its deterministic output marker."""
        popen = runtime.subprocess.Popen
        self.interrupted_processes = []

        def start(argv, *args, **kwargs):
            process = popen(argv, *args, **kwargs)
            actual_phase = "selfcheck" if "--selfcheck" in argv else "evaluation"
            if actual_phase != phase:
                return process
            self.interrupted_processes.append(process)
            original_wait = process.wait
            interrupted = False

            def wait(*wait_args, **wait_kwargs):
                nonlocal interrupted
                if not interrupted:
                    marker = self.task / "interrupt.ready"
                    deadline = time.monotonic() + 5
                    while not marker.is_file() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(marker.is_file(), "CPU fixture did not reach interruption marker")
                    interrupted = True
                    raise KeyboardInterrupt
                return original_wait(*wait_args, **wait_kwargs)

            process.wait = wait
            return process

        return patch.object(runtime.subprocess, "Popen", side_effect=start)

    def test_selfcheck_first_opaque_baseline_and_unknown_result_never_become_pass(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.events(), ["selfcheck", "evaluation"])
        record = self.record()
        self.assertEqual(record["status"], "executed_uninterpreted")
        self.assertFalse(record["official_acceptance"])
        self.assertEqual([step["exit_code"] for step in record["steps"]], [0, 0])
        self.assertTrue(all(Path(step["log"]).is_file() for step in record["steps"]))
        self.assertEqual(record["materials"]["baseline"]["sha256"], hashlib.sha256(self.baseline.read_bytes()).hexdigest())
        self.assertEqual(Path(record["outputs"][0]["snapshot"]).read_bytes(), (self.task / "verify_result.json").read_bytes())

    def test_selfcheck_failure_never_evaluates_and_preserves_old_output(self):
        self.make_script(selfcheck="raise SystemExit(7)")
        old = self.task / "verify_result.json"
        old.write_bytes(b"old evidence bytes")
        self.assertEqual(self.execute(), 7)
        self.assertEqual(self.events(), ["selfcheck"])
        record = self.record()
        self.assertEqual(record["status"], "selfcheck_failed")
        self.assertEqual(old.read_bytes(), b"old evidence bytes")
        self.assertEqual(Path(record["previous_outputs"][0]["archive"]).read_bytes(), b"old evidence bytes")

    def test_selfcheck_baseline_drift_stops_before_evaluation(self):
        self.make_script(selfcheck="Path(os.environ['OFFICIAL_BASELINE']).write_text('changed')")
        self.assertEqual(self.execute(), 2)
        self.assertEqual(self.events(), ["selfcheck"])
        self.assertEqual(self.record()["status"], "materials_changed")

    def test_script_drift_is_detected(self):
        self.make_script(selfcheck="Path(__file__).write_text(Path(__file__).read_text()+'\\n# changed\\n')")
        self.assertEqual(self.execute(), 2)
        self.assertEqual(self.events(), ["selfcheck"])
        self.assertEqual(self.record()["status"], "materials_changed")

    def test_dry_run_reads_plan_but_executes_nothing_and_creates_no_runtime(self):
        with patch.object(runtime.subprocess, "Popen") as popen:
            self.assertEqual(self.execute(dry_run=True), 0)
        popen.assert_not_called()
        plan = json.loads(self.stdout.getvalue())
        self.assertEqual(self.health_calls, 0)
        self.assertFalse(plan["filesystem_mutated"])
        self.assertFalse(plan["subprocess_executed"])
        self.assertFalse((self.task / "runtime").exists())
        self.assertFalse((self.task / "events.txt").exists())

    def test_dry_run_missing_materials_is_actionable_without_side_effects(self):
        self.env.pop("OFFICIAL_VERIFY_SCRIPT")
        self.assertEqual(self.execute(dry_run=True), 2)
        plan = json.loads(self.stdout.getvalue())
        self.assertIn("missing OFFICIAL_VERIFY_SCRIPT", plan["problems"])
        self.assertFalse((self.task / "runtime").exists())
        self.assertEqual(self.health_calls, 0)

    def test_cli_dry_run_is_honored_for_verify_and_bench(self):
        for kind in ("verify", "bench"):
            with self.subTest(kind=kind), patch.dict(os.environ, self.env, clear=True):
                with patch.object(sys, "argv", [str(SCRIPT), kind, "--task-dir", str(self.task), "--dry-run"]):
                    with patch.object(runtime, "healthy") as health, patch.object(runtime.subprocess, "Popen") as popen:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            self.assertEqual(runtime.main(), 0 if kind == "verify" else 2)
                        health.assert_not_called()
                        popen.assert_not_called()
        self.assertFalse((self.task / "runtime").exists())

    def test_stale_previous_output_cannot_count_as_new_result(self):
        self.make_script(evaluation="pass")
        (self.task / "verify_result.json").write_text('{"passed":true}')
        self.assertEqual(self.execute(), 2)
        record = self.record()
        self.assertEqual(record["status"], "outputs_incomplete")
        self.assertEqual(len(record["previous_outputs"]), 1)
        self.assertEqual(record["outputs"], [])

    def test_explicit_wrong_identity_is_rejected(self):
        self.make_script(evaluation="Path('verify_result.json').write_text(json.dumps({'MODEL_ID':'other/model','passed':True}))")
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.record()["status"], "identity_mismatch")
        self.assertFalse(self.record()["official_acceptance"])

    def test_reported_failure_is_not_execution_success(self):
        self.make_script(evaluation="Path('verify_result.json').write_text(json.dumps({'passed':False}))")
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.record()["status"], "reported_failure")

    def test_explicit_selfcheck_arguments_are_recorded_separately(self):
        self.env["OFFICIAL_SELFCHECK_ARGS_JSON"] = '["--audit"]'
        self.env["OFFICIAL_VERIFY_ARGS_JSON"] = '["--endpoint","local-fixture"]'
        self.assertEqual(self.execute(), 0)
        record = self.record()
        self.assertEqual(record["steps"][0]["command"][-1], "--audit")
        self.assertEqual(record["steps"][1]["command"][-2:], ["--endpoint", "local-fixture"])
        self.assertEqual(self.events(), ["selfcheck", "evaluation"])

    def test_evaluation_timeout_is_bounded_and_preserves_log(self):
        # Windows cold-start may exceed 100 ms before the fixture prints anything.
        self.env["OFFICIAL_TIMEOUT_SECONDS"] = "1"
        self.make_script(evaluation="print('partial stdout',flush=True)\ntime.sleep(60)")
        self.assertEqual(self.execute(), 124)
        record = self.record()
        self.assertEqual(record["status"], "execution_failed")
        self.assertTrue(record["steps"][-1]["timed_out"])
        self.assertIn("partial stdout", Path(record["steps"][-1]["log"]).read_text())

    def test_evaluation_interrupt_is_terminal_reaps_child_and_preserves_partial_and_old_results(self):
        old = self.task / "verify_result.json"
        old.write_bytes(b"previous report bytes")
        self.make_script(evaluation=
                         "Path('verify_result.json').write_text(json.dumps({'partial':True}))\n"
                         "if os.name=='posix':\n"
                         "    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                         "    Path('child.pid').write_text(str(child.pid))\n"
                         "print('partial interrupted stdout',flush=True)\n"
                         "Path('interrupt.ready').write_text('ready')\ntime.sleep(60)")
        with self.interrupt_wait_for_phase("evaluation"):
            self.assertEqual(self.execute(), 130)
        record = self.record()
        self.assertEqual(record["status"], "interrupted")
        self.assertFalse(record["official_acceptance"])
        self.assertTrue(record["ended_utc"])
        self.assertEqual(self.events(), ["selfcheck", "evaluation"])
        stage = record["steps"][-1]
        self.assertEqual(stage["exit_code"], 130)
        self.assertTrue(stage["interrupted"])
        self.assertTrue(stage["ended_utc"])
        self.assertTrue(stage["log_sha256"])
        self.assertTrue(record["partial_outputs_safe"])
        self.assertIn("partial interrupted stdout", Path(stage["log"]).read_text())
        self.assertEqual(Path(record["previous_outputs"][0]["archive"]).read_bytes(), b"previous report bytes")
        self.assertEqual(json.loads(Path(record["outputs"][0]["snapshot"]).read_text()), {"partial": True})
        self.assertEqual(json.loads(old.read_text()), {"partial": True})
        self.assertTrue(all(process.returncode is not None for process in self.interrupted_processes))
        if sys.platform == "linux":
            child_pid = int((self.task / "child.pid").read_text())
            for _ in range(20):
                if runtime.process_token(child_pid) is None:
                    break
                time.sleep(0.05)
            self.assertIsNone(runtime.process_token(child_pid), "interrupted evaluator descendant remains live")

    def test_selfcheck_interrupt_preserves_old_report_without_relabeling_it_as_new(self):
        old = self.task / "verify_result.json"
        old.write_bytes(b"previous untouched report")
        self.make_script(selfcheck="Path('interrupt.ready').write_text('ready')\ntime.sleep(60)")
        with self.interrupt_wait_for_phase("selfcheck"):
            self.assertEqual(self.execute(), 130)
        record = self.record()
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(self.events(), ["selfcheck"])
        self.assertEqual(record["steps"][0]["exit_code"], 130)
        self.assertTrue(record["steps"][0]["ended_utc"])
        self.assertEqual(record["outputs"], [])
        self.assertEqual(old.read_bytes(), b"previous untouched report")
        self.assertEqual(Path(record["previous_outputs"][0]["archive"]).read_bytes(), b"previous untouched report")
        self.assertTrue(all(process.returncode is not None for process in self.interrupted_processes))

    def test_selfcheck_interrupt_snapshots_changed_partial_report_separately(self):
        old = self.task / "verify_result.json"
        old.write_bytes(b"previous selfcheck report")
        self.make_script(selfcheck="Path('verify_result.json').write_text('{\"partial_selfcheck\":true}')\n"
                                  "Path('interrupt.ready').write_text('ready')\ntime.sleep(60)")
        with self.interrupt_wait_for_phase("selfcheck"):
            self.assertEqual(self.execute(), 130)
        record = self.record()
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(self.events(), ["selfcheck"])
        self.assertEqual(Path(record["previous_outputs"][0]["archive"]).read_bytes(), b"previous selfcheck report")
        self.assertEqual(json.loads(Path(record["outputs"][0]["snapshot"]).read_text()), {"partial_selfcheck": True})
        self.assertTrue(record["partial_outputs_safe"])

    @unittest.skipUnless(sys.platform == "linux", "Linux process-group cleanup")
    def test_selfcheck_timeout_kills_its_child_process_group(self):
        self.env["OFFICIAL_SELFCHECK_TIMEOUT_SECONDS"] = "0.3"
        self.make_script(selfcheck="child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                                  "Path('child.pid').write_text(str(child.pid))\ntime.sleep(60)")
        self.assertEqual(self.execute(), 124)
        self.assertEqual(self.events(), ["selfcheck"])
        pid = int((self.task / "child.pid").read_text())
        for _ in range(20):
            if runtime.process_token(pid) is None:
                break
            time.sleep(0.05)
        self.assertIsNone(runtime.process_token(pid), "timed-out evaluator child remains live")

    def test_bench_dry_run_is_six_loads_two_rounds_608_requests_without_baseline_requirement(self):
        self.env["OFFICIAL_BENCH_SCRIPT"] = str(self.script)
        self.env.pop("OFFICIAL_BASELINE")
        self.env.pop("OFFICIAL_DATA_DIR")
        self.assertEqual(self.execute(kind="bench", dry_run=True), 0)
        plan = json.loads(self.stdout.getvalue())
        loads = plan["t02_template_load_plan"]
        self.assertEqual(len(loads), 12)
        self.assertEqual({item["round"] for item in loads}, {1, 2})
        self.assertEqual(sum(item["requests"] for item in loads), 608)
        self.assertEqual([item["requests"] for item in loads[:6]], [8, 8, 16, 16, 128, 128])
        self.assertEqual(sum(item["scored"] for item in loads), 6)
        self.assertFalse((self.task / "runtime").exists())

    def test_no_script_output_can_escape_task(self):
        self.env["OFFICIAL_VERIFY_OUTPUTS_JSON"] = '["../outside.json"]'
        with self.assertRaisesRegex(ValueError, "stay inside"):
            self.execute(dry_run=True)

    def test_selfcheck_only_needs_no_dataset_baseline_or_service_and_never_evaluates(self):
        self.env.pop("OFFICIAL_BASELINE")
        self.env.pop("OFFICIAL_DATA_DIR")
        self.make_script(selfcheck="print('real selfcheck stdout')\nprint('real selfcheck stderr', file=sys.stderr)",
                         evaluation="raise RuntimeError('full evaluation must not execute')")
        original_script = self.script.read_bytes()
        old = self.task / "verify_result.json"
        old.write_bytes(b"old accuracy report; must survive selfcheck")
        self.assertEqual(self.execute(selfcheck_only=True), 0)
        self.assertEqual(self.health_calls, 0)
        self.assertEqual(self.events(), ["selfcheck"])
        self.assertEqual(old.read_bytes(), b"old accuracy report; must survive selfcheck")
        record = self.record()
        self.assertEqual(record["status"], "selfcheck_executed_uninterpreted")
        self.assertTrue(record["selfcheck_only"])
        self.assertFalse(record["official_acceptance"])
        self.assertFalse(record["service_health_checked"])
        self.assertFalse(record["outputs_required"])
        self.assertEqual(record["expected_outputs"], [])
        self.assertIsNone(record["command"])
        self.assertEqual(record["outputs"], [])
        self.assertEqual(len(record["steps"]), 1)
        self.assertNotIn("baseline", record["materials"])
        saved_script = Path(record["steps"][0]["log"]).parent / "materials/script.py"
        self.assertEqual(saved_script.read_bytes(), original_script)
        raw_log = Path(record["steps"][0]["log"]).read_bytes()
        self.assertIn(b"real selfcheck stdout", raw_log)
        self.assertIn(b"real selfcheck stderr", raw_log)
        self.assertEqual(record["steps"][0]["log_sha256"], hashlib.sha256(raw_log).hexdigest())

    def test_selfcheck_only_dry_run_missing_script_has_no_side_effects(self):
        self.env.pop("OFFICIAL_VERIFY_SCRIPT")
        self.env.pop("OFFICIAL_BASELINE")
        self.env.pop("OFFICIAL_DATA_DIR")
        with patch.object(runtime.subprocess, "Popen") as popen:
            self.assertEqual(self.execute(dry_run=True, selfcheck_only=True), 2)
        popen.assert_not_called()
        plan = json.loads(self.stdout.getvalue())
        self.assertEqual(plan["problems"], ["missing OFFICIAL_VERIFY_SCRIPT"])
        self.assertIsNone(plan["command"])
        self.assertEqual(self.health_calls, 0)
        self.assertFalse((self.task / "runtime").exists())

    def test_selfcheck_only_failure_and_changed_report_do_not_run_evaluation(self):
        self.make_script(selfcheck="Path('verify_result.json').write_text('{\"partial_selfcheck\":true}')\nraise SystemExit(9)")
        self.assertEqual(self.execute(selfcheck_only=True), 9)
        self.assertEqual(self.events(), ["selfcheck"])
        record = self.record()
        self.assertEqual(record["status"], "selfcheck_failed")
        self.assertFalse(record["official_acceptance"])
        self.assertTrue(record["partial_outputs_safe"])
        self.assertEqual(json.loads(Path(record["outputs"][0]["snapshot"]).read_text()), {"partial_selfcheck": True})

    def test_selfcheck_cli_flag_and_alias_only_work_for_verify(self):
        for flag in ("--selfcheck", "--selfcheck-only"):
            with self.subTest(flag=flag), patch.dict(os.environ, self.env, clear=True):
                with patch.object(sys, "argv", [str(SCRIPT), "verify", "--task-dir", str(self.task), flag, "--dry-run"]):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(runtime.main(), 0)
                    self.assertTrue(json.loads(output.getvalue())["selfcheck_only"])
                with patch.object(sys, "argv", [str(SCRIPT), "bench", "--task-dir", str(self.task), flag]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(runtime.main(), 2)

    def test_t04_bench_requires_explicit_audio_script_and_output_contract(self):
        (self.task / "task.json").write_text('{"track":"T04"}')
        self.assertEqual(self.execute(kind="bench", dry_run=True), 2)
        plan = json.loads(self.stdout.getvalue())
        self.assertEqual(set(plan["problems"]), {"missing OFFICIAL_BENCH_SCRIPT", "missing OFFICIAL_BENCH_OUTPUTS_JSON"})
        self.assertEqual(plan["expected_outputs"], [])
        self.assertNotIn("t02_template_load_plan", plan)
        self.assertNotIn("template_requests_total", plan)
        self.assertNotIn("image_resolution_confirmed", plan)
        self.assertFalse((self.task / "runtime").exists())
        self.env["OFFICIAL_BENCH_SCRIPT"] = str(self.script)
        self.assertEqual(self.execute(kind="bench", dry_run=True), 2)
        self.assertEqual(json.loads(self.stdout.getvalue())["problems"], ["missing OFFICIAL_BENCH_OUTPUTS_JSON"])

    def test_t04_bench_executes_audio_contract_without_inherited_image_mode(self):
        (self.task / "task.json").write_text('{"track":"T04"}')
        self.env.update(OFFICIAL_BENCH_SCRIPT=str(self.script),
                        OFFICIAL_BENCH_OUTPUTS_JSON='["audio-bench/summary.json"]', BENCH_MODE="image")
        self.make_script(evaluation="assert 'BENCH_MODE' not in os.environ\n"
                         "Path('audio-bench').mkdir()\nPath('audio-bench/summary.json').write_text('{\"audio_seconds\":123}')")
        self.assertEqual(self.execute(kind="bench"), 0)
        self.assertEqual(self.events(), ["evaluation"])
        record = self.record()
        self.assertEqual(record["track"], "T04")
        self.assertEqual(record["status"], "executed_uninterpreted")
        self.assertFalse(record["official_acceptance"])
        self.assertEqual(record["expected_outputs"], ["audio-bench/summary.json"])
        self.assertEqual(len(record["outputs"]), 1)
        self.assertNotIn("template_requests_total", record)

    def test_t04_bench_rejects_empty_or_unsafe_output_contract_before_execution(self):
        (self.task / "task.json").write_text('{"track":"T04"}')
        self.env["OFFICIAL_BENCH_SCRIPT"] = str(self.script)
        for value in ('[]', '["../outside.json"]', '["evaluator.py"]', '["a.json","a.json"]'):
            self.env["OFFICIAL_BENCH_OUTPUTS_JSON"] = value
            with self.subTest(value=value), patch.object(runtime.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    self.execute(kind="bench")
                popen.assert_not_called()
        self.assertFalse((self.task / "runtime").exists())


if __name__ == "__main__":
    unittest.main()
