"""CPU-only tests of SHA-bound repair and failure-preserving evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repair_under_test", ROOT / "scripts/repair_sglang_client.py")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class RepairClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.target = self.root / "partial004c"
        (self.target / "bin").mkdir(parents=True)
        (self.target / "bin/python").write_bytes(b"CPU mock interpreter")
        (self.target / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        folder = self.target / "client-bootstrap-evidence"
        folder.mkdir()
        self.record = folder / "record.json"
        self.native = {"torch": {"version": "native", "root": "/native", "metadata_sha256": "a"*64}}
        (folder / "pip.log").write_bytes(b"original pip0")
        (folder / "help.log").write_bytes(b"original missing orjson")
        self.old = {"venv": str(self.target), "native_before": self.native, "native_after": self.native,
                    "native_unchanged": True, "protected_native": {"torch": "native"},
                    "wheel_lock": [{"name": "sglang", "version": "0.5.12", "sha256": repair.SGLANG_WHEEL_SHA}],
                    "steps": [{"label": label, "exit_code": code,
                               "log_sha256": repair.digest((folder / (label+".log")).read_bytes())}
                              for label, code in (("pip", 0), ("help", 1))]}
        self.record.write_text(json.dumps(self.old))
        self.lock = self.root / "cpu.json"
        self.wheels = [{"name": "orjson", "version": "3.11.3", "url": "https://files.pythonhosted.org/packages/orjson-3.11.3-py3-none-any.whl", "sha256": "b"*64}]
        self.lock.write_text(json.dumps(self.wheels))

    def validate(self):
        return repair.validate_inputs(self.record, repair.digest(self.record.read_bytes()), self.lock, repair.digest(self.lock.read_bytes()))

    def test_original_record_log_and_explicit_lock_identity_are_required(self):
        self.assertEqual(self.validate()[2], self.target)
        with self.assertRaisesRegex(ValueError, "SHA"):
            repair.validate_inputs(self.record, "0"*64, self.lock, repair.digest(self.lock.read_bytes()))
        (self.record.parent / "help.log").write_bytes(b"modified original help")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.validate()

    def test_cpu_lock_cannot_reinstall_sglang_torch_or_mismatch_filename(self):
        for name in ("torch", "sglang", "transformers", "orjson-other"):
            entry = {**self.wheels[0], "name": name}
            self.lock.write_text(json.dumps([entry]))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate()
        self.lock.write_text(json.dumps([{**self.wheels[0], "version": "9.9"}]))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.validate()

    def context(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(repair.sys, "platform", "linux"))
        stack.enter_context(patch.object(repair.sys, "version_info", (3, 10, 12)))
        stack.enter_context(patch.object(repair.sys, "prefix", "native"))
        stack.enter_context(patch.object(repair.sys, "base_prefix", "native"))
        stack.enter_context(patch.object(repair, "native_snapshot", return_value=self.native))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def run_fixture(self, help_code=0, reused=False):
        before = {"native": self.native, "packages": {"sglang": {"version": "0.5.12", "root": str(self.target / "lib/site-packages")},
                                                          "orjson": {"version": "existing", "root": "/native"} if reused else None}}
        after = json.loads(json.dumps(before))
        if not reused:
            after["packages"]["orjson"] = {"version": "3.11.3", "root": str(self.target / "lib/site-packages")}
        commands = []

        def fake_run(argv, log, timeout, env):
            self.assertEqual(argv[0], str(self.target / "bin/python"))
            self.assertNotIn("PIP_TARGET", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            commands.append(argv)
            log.write_bytes(b"original CPU mock output")
            return {"argv": argv, "exit_code": help_code if "--help" in argv else 0,
                    "log": log.name, "log_sha256": repair.digest(log.read_bytes())}

        with self.context(), patch.object(repair, "client_snapshot", side_effect=[before, after]), \
                patch.object(repair, "run", side_effect=fake_run), patch.dict(repair.os.environ, {"PIP_TARGET": "/bad-native-target"}):
            code = repair.apply(self.old, self.wheels, self.target, self.record, self.lock)
        return code, commands

    def test_only_missing_cpu_wheel_installs_and_original_record_is_unchanged(self):
        original = self.record.read_bytes()
        code, commands = self.run_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(self.record.read_bytes(), original)
        self.assertEqual(len(commands), 2)
        self.assertIn("--no-deps", commands[0])
        self.assertIn("--require-hashes", commands[0])
        self.assertNotIn("sglang", commands[0])
        new = next((self.target / "client-repair-evidence").glob("*/record.json"))
        report = json.loads(new.read_text())
        self.assertTrue(report["native_unchanged"])
        self.assertTrue(report["compatibility_proven"])
        self.assertEqual(report["status"], "help_complete")

    def test_reused_package_does_not_trigger_pip_or_upgrade(self):
        code, commands = self.run_fixture(reused=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 1)
        self.assertIn("--help", commands[0])
        record = json.loads(next((self.target / "client-repair-evidence").glob("*/record.json")).read_text())
        self.assertEqual(record["selected_missing_wheels"], [])
        self.assertEqual(record["reused_packages"]["orjson"]["version"], "existing")

    def test_each_failed_help_attempt_gets_its_own_terminal_record(self):
        for _ in range(2):
            code, _ = self.run_fixture(help_code=1)
            self.assertEqual(code, 1)
        reports = list((self.target / "client-repair-evidence").glob("*/record.json"))
        self.assertEqual(len(reports), 2)
        for path in reports:
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "help_failed")
            self.assertTrue(record["native_unchanged"])
            self.assertFalse(record["compatibility_proven"])
            self.assertTrue(record["ended_utc"])
            self.assertEqual(record["steps"][-1]["exit_code"], 1)

    def test_client_snapshot_can_import_helper_under_an_arbitrary_filename(self):
        # Include a REAL installed distribution in both the protected set and
        # package loop. An all-missing loop cannot catch a dist variable that
        # accidentally overwrites the imported helper module.
        import shutil
        import sys
        import importlib.metadata as md
        known_distribution = next(md.distributions(), None)
        self.assertIsNotNone(known_distribution, "real metadata regression needs an installed distribution")
        known = known_distribution.metadata["Name"]
        expected = {known: known_distribution.version}
        alias = self.root / "standalone-upload-alias.py"
        shutil.copyfile(ROOT / "scripts/repair_sglang_client.py", alias)
        spec = importlib.util.spec_from_file_location("alias_under_test", alias)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.client_snapshot(sys.executable, expected, [known, "definitely-missing-cpu-mock"], self.root, "alias", {})
        self.assertEqual(result["native"], module.native_snapshot(expected))
        self.assertEqual(result["packages"][known]["version"], known_distribution.version)
        self.assertIsNone(result["packages"]["definitely-missing-cpu-mock"])


if __name__ == "__main__":
    unittest.main()
