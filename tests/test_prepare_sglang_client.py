"""Bootstrap planning and dependency locks never install anything in CPU tests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("client_prepare_test", ROOT / "scripts/prepare_sglang_client.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class SGLangPreparationTests(unittest.TestCase):
    def apply_context(self, target, builder, runner):
        stack = ExitStack()
        stack.enter_context(patch.object(prepare.sys, "platform", "linux"))
        stack.enter_context(patch.object(prepare.sys, "version_info", (3, 10, 12)))
        stack.enter_context(patch.object(prepare.sys, "prefix", "native-base"))
        stack.enter_context(patch.object(prepare.sys, "base_prefix", "native-base"))
        stack.enter_context(patch.object(prepare.platform, "machine", return_value="x86_64"))
        stack.enter_context(patch.object(prepare.platform, "libc_ver", return_value=("glibc", "2.35")))
        native = {"torch": {"version": "mock-native", "root": "/native", "metadata_sha256": "a"*64}}
        stack.enter_context(patch.object(prepare, "snapshot", return_value=native))
        stack.enter_context(patch.object(prepare.venv, "EnvBuilder", builder))
        stack.enter_context(patch.object(prepare.subprocess, "run", runner))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return stack, native

    def test_plan_does_not_install_create_or_inspect_native_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "new-client"
            with patch.object(prepare.venv, "EnvBuilder") as builder, patch.object(prepare.subprocess, "run") as run, \
                    patch.object(prepare, "snapshot") as snapshot, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(prepare.main(["--venv", str(target)]), 0)
                builder.assert_not_called()
                run.assert_not_called()
                snapshot.assert_not_called()
            plan = json.loads(out.getvalue())
            self.assertTrue(plan["system_site_packages"])
            self.assertIn("--no-deps", plan["install_flags"])
            self.assertEqual(plan["wheel_lock"][0]["sha256"], "4b88edfd3870d7dc8dd880f97d104f637baf105538827f968483d6cac56258f8")
            self.assertFalse(plan["compatibility_proven"])
            self.assertFalse(target.exists())

    def test_extra_cpu_wheels_require_hash_and_cannot_replace_native_or_gpu_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cpu.json"
            for name in ("torch", "transformers", "numpy", "triton", "flashinfer-python", "unknown-package"):
                path.write_text(json.dumps([{"name": name, "version": "1.0", "url": "https://files.pythonhosted.org/packages/test.whl", "sha256": "a"*64}]))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    prepare.locks(path, {"torch": "protected"})
            value = {"name": "pybase64", "version": "1.0", "url": "https://files.pythonhosted.org/packages/test.whl", "sha256": "a"*64}
            path.write_text(json.dumps([value]))
            self.assertEqual(prepare.locks(path, {})[1], value)
            value["sha256"] = ""
            path.write_text(json.dumps([value]))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prepare.locks(path, {})

    def test_existing_or_relative_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(prepare.main(["--venv", str(Path(temporary).resolve())]), 2)
            self.assertEqual(prepare.main(["--venv", "relative-client"]), 2)

    def test_apply_uses_native_pip_via_venv_without_ensurepip(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "client004c"
            seen = []

            class NoEnsurepipBuilder:
                def __init__(self, **kwargs):
                    if kwargs.get("with_pip"):
                        raise OSError("this image deliberately has no ensurepip")
                    self.assertions = kwargs
                    seen.append(kwargs)

                def create(self, path):
                    # Evidence must already exist even if creation later fails.
                    assert (path / "client-bootstrap-evidence/record.json").is_file()
                    (path / "bin").mkdir()

            native = {"torch": {"version": "mock-native", "root": "/native", "metadata_sha256": "a"*64}}

            def runner(argv, **kwargs):
                seen.append(argv)
                self.assertEqual(argv[0], str(target / "bin/python"))
                raw = json.dumps(native).encode() if "-c" in argv else b"original CPU mock command log\n"
                return prepare.subprocess.CompletedProcess(argv, 0, stdout=raw, stderr=b"")

            stack, _ = self.apply_context(target, NoEnsurepipBuilder, runner)
            with stack:
                self.assertEqual(prepare.main(["--venv", str(target), "--apply"]), 0)
            self.assertEqual(seen[0], {"with_pip": False, "system_site_packages": True})
            self.assertEqual(seen[1][1:4], ["-m", "pip", "install"])
            record = json.loads((target / "client-bootstrap-evidence/record.json").read_text())
            self.assertEqual(record["status"], "ready")
            self.assertTrue(record["native_unchanged"])
            self.assertTrue(record["compatibility_proven"])
            self.assertFalse(any("ensurepip" in str(arg) for item in seen[1:] for arg in item))

    def test_venv_creation_failure_still_records_original_error_and_native_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "failed004c"

            class FailedBuilder:
                def __init__(self, **kwargs):
                    assert kwargs["with_pip"] is False

                def create(self, path):
                    assert (path / "client-bootstrap-evidence/record.json").is_file()
                    raise OSError("original venv creation fixture failure")

            with patch.object(prepare.subprocess, "run") as runner:
                stack, _ = self.apply_context(target, FailedBuilder, runner)
                with stack:
                    self.assertEqual(prepare.main(["--venv", str(target), "--apply"]), 2)
                runner.assert_not_called()
            record = json.loads((target / "client-bootstrap-evidence/record.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"], "original venv creation fixture failure")
            self.assertEqual(record["failure_type"], "OSError")
            self.assertTrue(record["native_unchanged"])
            self.assertFalse(record["compatibility_proven"])
            self.assertTrue(record["ended_utc"])
            self.assertEqual(record["steps"], [])


if __name__ == "__main__":
    unittest.main()
