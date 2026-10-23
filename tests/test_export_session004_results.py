import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_session004_results.py"
SPEC = importlib.util.spec_from_file_location("export_session004_results", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportSession004Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.candidates = self.root / "candidates"
        self.sglang = self.root / "sglang"
        self.output = self.root / "result.zip"

    def put(self, relative, content="{}", base=None):
        path = (base or self.project) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def run_export(self, **kwargs):
        return exporter.export_results(self.project, self.candidates, self.sglang, self.output, **kwargs)

    def manifest(self):
        with zipfile.ZipFile(self.output) as archive:
            return json.loads(archive.read("export-manifest.json")), archive.namelist()

    def test_exact_allowed_payloads_and_scripts_hash_only(self):
        self.put("evidence/session004/recon/row.json", '{"completed":2}')
        self.put("evidence/session004/recon/stdout.log", "original raw stdout\n")
        self.put("evidence/session003/old.json", "old")
        self.put("evidence/session004/cookies.json", "cookie-secret")
        self.put("evidence/session004/access_token.txt", "token-secret")
        self.put("evidence/session004/token.txt", "token-secret")
        self.put("evidence/session004/weights/model.safetensors", "weights")
        self.put("evidence/session004/venv/site-packages/config.json", "native")
        self.put("scripts/helper.py", "print('source must not be archived')")
        self.put("scripts/arbitrary.json", "not a result")
        self.put("tasks/recon2reason-bw-vllm/task.json", '{"track":"T02"}')
        self.put("tasks/recon2reason-bw-vllm/env.sh", "export TEST=1")
        self.put("tasks/recon2reason-bw-vllm/results/round01.jsonl", '{}\n')
        self.put("tasks/recon2reason-bw-vllm/runtime/server.log", "server raw\n")
        self.put("tasks/recon2reason-bw-vllm/runtime/audio-venv/lib/pkg.json", "package")
        self.put("tasks/qwen3-asr-bw-vllm/runtime/audio-venv/asr-prepare-evidence/validation.stdout", "prepared")
        self.put("fixtures/audio/manifest.json", '{"count":1}')
        self.put("fixtures/audio/private.json", "not manifest")
        self.put("fixtures/audio/speech.wav", "audio")
        self.put("candidate-env-id.json", '{"status":"prepared"}', self.candidates)
        self.put("envs/forced-main/candidate-environment.json", "{}", self.candidates)
        self.put("envs/forced-main/pip-install.log", "pip original", self.candidates)
        self.put("envs/forced-main/partial-repair-evidence/attempt1/repair-record.json", '{"status":"verified"}', self.candidates)
        self.put("envs/forced-main/partial-repair-evidence/attempt1/repair_helper.py", "print('hash only')", self.candidates)
        self.put("envs/forced-main/lib/python3.10/site-packages/token.json", "secret", self.candidates)
        self.put("models/model.safetensors", "weights", self.candidates)
        self.put("results/trident/run-final.json", "{}", self.candidates)
        self.put("client-bootstrap-evidence/bootstrap-record.json", "{}", self.sglang)
        self.put("client-repair-evidence/attempt1/repair_sglang_client.py", "print(1)", self.sglang)
        self.put("client-repair-evidence/attempt1/help.log", "help raw", self.sglang)
        result = self.run_export()
        manifest, names = self.manifest()
        self.assertEqual(result["status"], "exported_and_verified")
        self.assertEqual(result["script_hashes"], 4)
        self.assertTrue(manifest["script_hashes_unchanged"])
        self.assertIn("candidates/envs/forced-main/pip-install.log", names)
        self.assertIn("candidates/envs/forced-main/partial-repair-evidence/attempt1/repair-record.json", names)
        self.assertIn("project/fixtures/audio/manifest.json", names)
        self.assertIn("sglang-client/client-repair-evidence/attempt1/help.log", names)
        self.assertIn("project/tasks/qwen3-asr-bw-vllm/runtime/audio-venv/asr-prepare-evidence/validation.stdout", names)
        for name in names:
            self.assertFalse(name.endswith((".py", ".sh", ".wav", ".safetensors")), name)
            self.assertNotIn("session003", name)
            self.assertNotIn("site-packages", name)
            self.assertNotIn("token", name)
            self.assertNotIn("cookies", name)
        self.assertNotIn("project/scripts-hashes/arbitrary.json", names)
        self.assertEqual(manifest["evaluation_status"], "not_assessed_by_exporter")
        exporter.verify_archive(self.output, manifest)

    def test_raw_state_before_after_are_distinct_snapshots(self):
        task = "recon2reason-bw-vllm"
        source = self.put(f"tasks/{task}/runtime/state.json", '{"status":"ready"}\n')
        original = exporter.read_stable
        calls = 0
        def read_and_change(path, limit):
            nonlocal calls
            result = original(path, limit)
            if path == source:
                calls += 1
                if calls == 1:
                    source.write_text('{"status":"stopped"}\n')
            return result
        with mock.patch.object(exporter, "read_stable", side_effect=read_and_change):
            self.run_export()
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(json.loads(archive.read(f"export-snapshots/before/{task}/state.json"))["status"], "ready")
            self.assertEqual(json.loads(archive.read(f"export-snapshots/after/{task}/state.json"))["status"], "stopped")

    def test_existing_output_never_overwritten(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.run_export()
        self.assertEqual(self.output.read_bytes(), b"keep")

    def test_hard_link_race_refuses_new_existing_destination(self):
        original = os.link
        def raced(source, destination):
            Path(destination).write_bytes(b"other writer")
            return original(source, destination)
        with mock.patch.object(exporter.os, "link", side_effect=raced):
            with self.assertRaises(FileExistsError):
                self.run_export()
        self.assertEqual(self.output.read_bytes(), b"other writer")
        self.assertEqual(list(self.root.glob("*.partial-*")), [])

    def test_file_and_total_limits_fail_without_final_or_partial(self):
        self.put("evidence/session004/large.log", "x" * 2000)
        with self.assertRaisesRegex(ValueError, "File exceeds"):
            self.run_export(max_file_bytes=1000)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("*.partial-*")), [])
        with self.assertRaisesRegex(ValueError, "Total evidence"):
            self.run_export(max_total_bytes=1500)
        self.assertFalse(self.output.exists())

    def test_walk_errors_propagate(self):
        self.put("evidence/session004/row.json")
        def denied(*args, **kwargs):
            kwargs["onerror"](PermissionError("denied test subtree"))
            return iter(())
        with mock.patch.object(exporter.os, "walk", side_effect=denied):
            with self.assertRaisesRegex(PermissionError, "denied test"):
                self.run_export()
        self.assertFalse(self.output.exists())

    def test_mutating_file_read_fails_closed(self):
        source = self.put("evidence/session004/growing.log", "initial")
        original_open = Path.open
        class ChangingRead:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def fileno(self):
                return self.stream.fileno()
            def read(self, limit):
                data = self.stream.read(limit)
                with original_open(source, "ab") as writer:
                    writer.write(b"append during read")
                return data
        def changed_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return ChangingRead(stream) if path == source and args == ("rb",) else stream
        with mock.patch.object(Path, "open", changed_open):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.run_export()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("*.partial-*")), [])

    def test_payload_hash_readback_detects_repacked_tampering(self):
        self.put("evidence/session004/row.json", '{"result":1}')
        self.run_export()
        manifest, names = self.manifest()
        tampered = self.root / "tampered.zip"
        with zipfile.ZipFile(self.output) as source, zipfile.ZipFile(tampered, "w") as target:
            for name in names:
                data = source.read(name)
                if name.endswith("/row.json"):
                    data = b'{"result":2}'
                target.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "integrity check"):
            exporter.verify_archive(tampered, manifest)

    @unittest.skipUnless(sys.platform == "linux", "Linux symlink semantics")
    def test_selected_symlink_and_directory_symlink_rejected(self):
        outside = self.put("outside.json", "private", self.root)
        evidence = self.project / "evidence/session004"
        evidence.mkdir(parents=True)
        link = evidence / "linked.json"
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.run_export()
        link.unlink()
        (evidence / "linked-directory").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.run_export()

    def test_extra_files_allowlisted_and_external_script_hash_only(self):
        log = self.put("manual-help.log", "manual original", self.root)
        script = self.put("t02_sglang_bench_004f.py", "print('remote actual')", self.root)
        self.run_export(extra_files=[log], script_files=[script])
        manifest, names = self.manifest()
        self.assertIn("extra/000-manual-help.log", names)
        self.assertEqual(manifest["script_hashes_before"][0]["source"], str(script.absolute()))
        self.assertFalse(any(name.endswith(".py") for name in names))
        self.output.unlink()
        token = self.put("access_token.txt", "private", self.root)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            self.run_export(extra_files=[token])
        package = self.put("venv/site-packages/config.json", "{}", self.root)
        with self.assertRaisesRegex(ValueError, "environment directory"):
            self.run_export(extra_files=[package])

    def test_missing_explicit_file_and_bad_manifest_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.run_export(extra_files=[self.root / "absent.log"])
        self.run_export()
        manifest, _ = self.manifest()
        manifest["evaluation_status"] = "invented"
        with self.assertRaisesRegex(ValueError, "manifest readback"):
            exporter.verify_archive(self.output, manifest)


if __name__ == "__main__":
    unittest.main()
