"""CPU plan, aggregation, readiness and ownership checks for the session driver."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/session003_robo_asr_checks.py"
SPEC = importlib.util.spec_from_file_location("session_checks", SCRIPT)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)
RUNTIME_SPEC = importlib.util.spec_from_file_location("session_runtime", SCRIPT.parent / "recon2reason_runtime.py")
runtime = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime)


class SessionChecksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.args = types.SimpleNamespace(project=self.project, output=None, asr_venv=None,
                                          timeout_seconds=3600, load_timeout=600,
                                          model_runtime_seconds=1800, base_python="python3", run=False)

    def runner(self):
        return checks.Runner(checks.build_plan(self.args))

    def test_default_plan_never_starts_process_or_mutates_files(self):
        output = io.StringIO()
        with patch.object(checks.subprocess, "Popen") as popen, contextlib.redirect_stdout(output):
            self.assertEqual(checks.main(["--project", str(self.project)]), 0)
        popen.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["mode"], "plan_only")
        self.assertFalse(plan["official_acceptance"])
        self.assertFalse(plan["filesystem_mutated"])
        self.assertEqual(list(self.project.iterdir()), [])

    def test_unsafe_output_and_over_one_hour_are_rejected(self):
        self.args.output = self.project.parent / "outside-evidence"
        with self.assertRaisesRegex(ValueError, "inside"):
            checks.build_plan(self.args)
        self.args.output = None
        self.args.timeout_seconds = 3601
        with self.assertRaisesRegex(ValueError, "3600"):
            checks.build_plan(self.args)

    def test_failed_phase_still_allows_next_after_verified_cleanup_and_records_failure(self):
        runner = self.runner()

        def phase(name):
            runner.report["phases"].append({"name": name, "status": "failed" if name == "robo" else "development_checks_completed",
                                             "cleanup_verified": True})

        with patch.object(checks.sys, "platform", "linux"), patch.object(runner, "phase", side_effect=phase):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(), 1)
        report = json.loads((runner.output / "report.json").read_text())
        self.assertEqual([p["name"] for p in report["phases"]], ["robo", "asr"])
        self.assertEqual(report["status"], "failed_or_incomplete")
        with patch.object(checks.sys, "platform", "linux"):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                runner.run()

    def test_cleanup_does_not_stop_changed_owner_and_marks_unconfirmed(self):
        runner = self.runner()
        runner.output.mkdir(parents=True)
        task = runner.project / "task"
        (task / "runtime").mkdir(parents=True)
        (task / "runtime/state.json").write_text('{"run_id":"someone-else","status":"ready"}')
        phase = {"name": "robo", "port": 30001, "owned_run_id": "ours", "steps": []}
        runner.report["phases"].append(phase)
        with patch.object(runner, "step") as step, patch.object(checks, "port_open", return_value=False):
            runner.cleanup(phase, task, {})
        step.assert_not_called()
        self.assertFalse(phase["cleanup_verified"])
        self.assertIn("ownership changed", phase["stop_error"])

    def test_launch_requires_fresh_ready_state_and_retains_its_identity(self):
        runner = self.runner()
        (runner.output / "robo").mkdir(parents=True)
        task = self.project / "task"
        (task / "runtime").mkdir(parents=True)
        state = task / "runtime/state.json"
        state.write_text('{"run_id":"old","status":"ready"}')
        phase = {"name": "robo", "steps": []}
        runner.report["phases"].append(phase)

        def start(*_args, **_kwargs):
            argv = _args[0]
            state.write_text(json.dumps({"run_id": argv[argv.index("--run-id") + 1], "status": "ready"}))
            return types.SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda **_kwargs: 0)

        with patch.object(checks.subprocess, "Popen", side_effect=start):
            runner.launch(phase, task, {})
        self.assertEqual(phase["owned_run_id"], phase["requested_run_id"])
        self.assertEqual(phase["ready_state"]["run_id"], phase["requested_run_id"])
        self.assertEqual(phase["steps"][0]["status"], "ready")

    def task_state(self, runner, state):
        (runner.output / "robo").mkdir(parents=True)
        task = runner.project / "task"
        (task / "runtime").mkdir(parents=True)
        (task / "runtime/state.json").write_text(json.dumps(state))
        return task

    def test_concurrent_foreign_ready_run_is_never_adopted_or_stopped(self):
        runner = self.runner()
        task = self.task_state(runner, {"run_id": "old", "status": "stopped"})
        phase = {"name": "robo", "port": 30001, "steps": []}
        runner.report["phases"].append(phase)

        def another_launcher_wins(*_args, **_kwargs):
            (task / "runtime/state.json").write_text('{"run_id":"someone-else","status":"ready"}')
            return types.SimpleNamespace(pid=123, returncode=1, poll=lambda: 1, wait=lambda **_kwargs: 1)

        with patch.object(checks.subprocess, "Popen", side_effect=another_launcher_wins):
            with self.assertRaisesRegex(RuntimeError, "before a fresh ready"):
                runner.launch(phase, task, {})
        self.assertNotIn("owned_run_id", phase)
        with patch.object(runner, "step") as stop, patch.object(checks, "port_open", return_value=False), \
                patch.object(checks, "process_matches", return_value=False):
            runner.cleanup(phase, task, {})
        stop.assert_not_called()

    def test_server_alive_without_supervisor_or_port_is_not_verified_clean(self):
        runner = self.runner()
        task = self.task_state(runner, {"run_id": "ours", "status": "starting", "server_pid": 12,
                                        "server_token": "still-loading"})
        phase = {"name": "robo", "port": 30001, "owned_run_id": "ours", "steps": []}
        runner.report["phases"].append(phase)
        with patch.object(runner, "step", side_effect=RuntimeError("recorded server is alive")) as stop, \
                patch.object(checks, "port_open", return_value=False), \
                patch.object(checks, "process_matches", side_effect=lambda state, role="supervisor": role == "server"), \
                patch.object(checks.time, "monotonic", return_value=runner.deadline + 1):
            runner.cleanup(phase, task, {})
        self.assertFalse(phase["cleanup_verified"])
        self.assertIn("server is alive", phase["stop_error"])
        self.assertEqual(stop.call_args.args[2][-2:], ["--run-id", "ours"])

    def test_missing_terminal_state_is_not_verified_clean(self):
        runner = self.runner()
        task = self.task_state(runner, {"run_id": "ours", "status": "starting"})
        phase = {"name": "robo", "port": 30001, "owned_run_id": "ours", "steps": []}
        runner.report["phases"].append(phase)
        with patch.object(runner, "step"), patch.object(checks, "port_open", return_value=False), \
                patch.object(checks, "process_matches", return_value=False):
            runner.cleanup(phase, task, {})
        self.assertFalse(phase["cleanup_verified"])

    def test_timeout_and_interrupt_retain_only_requested_run_and_terminal_step(self):
        for mode in ("timeout", "interrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                args = types.SimpleNamespace(**vars(self.args))
                args.project = Path(temporary)
                runner = checks.Runner(checks.build_plan(args))
                task = self.task_state(runner, {"run_id": "old", "status": "stopped"})
                phase = {"name": "robo", "steps": []}
                runner.report["phases"].append(phase)

                def starting(argv, **_kwargs):
                    (task / "runtime/state.json").write_text(json.dumps({"run_id": argv[-1], "status": "starting"}))
                    return types.SimpleNamespace(pid=123, poll=lambda: None, terminate=lambda: None,
                                                 wait=lambda **_kwargs: 130, kill=lambda: None)

                if mode == "timeout":
                    runner.work_deadline = 0
                sleep = patch.object(checks.time, "sleep", side_effect=KeyboardInterrupt)
                with patch.object(checks.subprocess, "Popen", side_effect=starting), sleep:
                    with self.assertRaises(TimeoutError if mode == "timeout" else KeyboardInterrupt):
                        runner.launch(phase, task, {})
                self.assertEqual(phase["owned_run_id"], phase["requested_run_id"])
                self.assertEqual(phase["steps"][0]["status"], "failed")
                self.assertIn("ended_utc", phase["steps"][0])
                if mode == "interrupt":
                    self.assertTrue(runner.interrupted)
                    self.assertEqual(phase["steps"][0]["exit_code"], 130)
                else:
                    self.assertEqual(phase["steps"][0]["exit_code"], 124)

    def test_process_matches_checks_server_token_and_rejects_recycled_pid(self):
        # /proc fields after ')' start with state; starttime is index19.
        stat = "12 (server name) " + " ".join(["S"] + ["0"] * 18 + ["start-ticks"])
        state = {"server_pid": 12, "server_token": "start-ticks"}
        with patch.object(Path, "read_text", return_value=stat):
            self.assertTrue(checks.process_matches(state, "server"))
            self.assertFalse(checks.process_matches({**state, "server_token": "old-ticks"}, "server"))

    def test_runtime_rejects_invalid_run_ids_before_any_launch(self):
        for value in ("", "../other", "A" * 32, "a" * 31, "g" * 32):
            with self.subTest(value=value), patch.object(runtime, "configuration") as config:
                with self.assertRaisesRegex(ValueError, "run-id"):
                    runtime.launch(self.project, True, value)
                config.assert_not_called()

    def test_runtime_existing_config_is_not_overwritten_and_creation_is_locked(self):
        task = self.project / "runtime-task"
        (task / "runtime").mkdir(parents=True)
        run_id = "a" * 32
        config_path = task / "runtime" / ("config-" + run_id + ".json")
        config_path.write_bytes(b"original-config-bytes")
        locked = []

        @contextlib.contextmanager
        def locked_scope(_runtime):
            locked.append(True)
            try:
                yield
            finally:
                locked.pop()

        original_open = Path.open

        def checked_open(path, *args, **kwargs):
            if path == config_path:
                self.assertTrue(locked, "config must be exclusively created inside control lock")
                self.assertEqual(args[0], "x")
            return original_open(path, *args, **kwargs)

        config = {"model_path": str(task), "url": "http://127.0.0.1:30001"}
        with patch.object(runtime, "configuration", return_value=config), patch.object(runtime.sys, "platform", "linux"), \
                patch.object(runtime, "healthy", return_value=False), patch.object(runtime, "control_lock", locked_scope), \
                patch.object(Path, "open", checked_open), patch.object(runtime.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                runtime.launch(task, False, run_id)
        popen.assert_not_called()
        self.assertEqual(config_path.read_bytes(), b"original-config-bytes")

    def test_runtime_stop_checks_replaced_state_inside_lock_before_any_signal(self):
        task = self.project / "runtime-task"
        (task / "runtime").mkdir(parents=True)
        state = task / "runtime/state.json"
        state.write_text(json.dumps({"run_id": "a" * 32}))

        @contextlib.contextmanager
        def replacement_under_lock(_runtime):
            state.write_text(json.dumps({"run_id": "b" * 32}))
            yield

        with patch.object(runtime, "control_lock", replacement_under_lock), \
                patch.object(runtime, "matches") as matches, patch.object(runtime.os, "kill") as kill, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runtime.stop(task, "a" * 32), 2)
        matches.assert_not_called()
        kill.assert_not_called()

    def test_runtime_default_and_explicit_identity_publish_exact_config_without_real_processes(self):
        for explicit in (None, "b" * 32):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as temporary:
                task = Path(temporary)
                requested = explicit or "a" * 32
                config = {"model_path": str(task), "url": "http://127.0.0.1:30001", "load_timeout": 1, "stop_grace": 1}

                def fake_supervisor(argv, **_kwargs):
                    config_path = Path(argv[-1])
                    self.assertEqual(config_path.name, "config-" + requested + ".json")
                    self.assertEqual(json.loads(config_path.read_text())["run_id"], requested)
                    (task / "runtime/state.json").write_text(json.dumps({"run_id": requested, "status": "ready"}))
                    return types.SimpleNamespace(poll=lambda: None)

                (task / "runtime").mkdir()
                with patch.object(runtime, "configuration", return_value=config), patch.object(runtime.sys, "platform", "linux"), \
                        patch.object(runtime, "control_lock", return_value=contextlib.nullcontext()), \
                        patch.object(runtime, "healthy", return_value=False), \
                        patch.object(runtime.uuid, "uuid4", return_value=types.SimpleNamespace(hex="a" * 32)), \
                        patch.object(runtime.subprocess, "Popen", side_effect=fake_supervisor), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runtime.launch(task, False, explicit), 0)

    def test_runtime_matching_stop_signals_only_within_control_lock(self):
        task = self.project / "runtime-task"
        (task / "runtime").mkdir(parents=True)
        run_id = "a" * 32
        config = task / "runtime" / ("config-" + run_id + ".json")
        config.write_text(json.dumps({"run_id": run_id, "task": str(task), "stop_grace": 1}))
        (task / "runtime/state.json").write_text(json.dumps({"run_id": run_id, "supervisor_pid": 123,
                                                            "supervisor_token": "ticks", "config_path": str(config)}))
        active = []

        @contextlib.contextmanager
        def locked(_runtime):
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def signal_owned(pid, signal):
            self.assertTrue(active)
            self.assertEqual(pid, 123)
            self.assertEqual(signal, runtime.signal.SIGTERM)

        cmdline = b"\0".join([str(runtime.SCRIPT).encode(), b"supervise", str(config).encode()])
        with patch.object(runtime, "control_lock", locked), patch.object(runtime, "matches", side_effect=[True, False, False]), \
                patch.object(Path, "read_bytes", return_value=cmdline), patch.object(runtime.os, "kill", side_effect=signal_owned), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.stop(task, run_id), 0)

    @unittest.skipUnless(checks.sys.platform == "linux", "fcntl locking requires native Linux or WSL")
    def test_runtime_control_lock_refuses_concurrent_controller(self):
        with runtime.control_lock(self.project / "runtime"):
            with self.assertRaises(BlockingIOError):
                with runtime.control_lock(self.project / "runtime"):
                    self.fail("another controller must not enter")


if __name__ == "__main__":
    unittest.main()
