"""Small CPU-only frozen stages; never package the actual delivery directory."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package004", ROOT / "scripts/package_session004_repositories.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.stage = self.base / "frozen-stage004"
        self.stage.mkdir()
        self.output = self.base / "new-zips"
        self.sources = {}
        for task in packager.TASKS:
            directory = self.stage / task
            directory.mkdir()
            payloads = {"README.md": ("# " + task + " 中文源码\n").encode(),
                        "run.sh": b"#!/usr/bin/env bash\nexit 2\n",
                        "fixtures/small.txt": b"CPU fixture only\n"}
            for name, raw in payloads.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            rows = [{"path": name, "bytes": len(raw), "sha256": packager.sha(raw)} for name, raw in sorted(payloads.items())]
            manifest = {"schema_version": 1, "session": "session004", "primary_task": task,
                        "files": rows, "payload_bytes": sum(len(raw) for raw in payloads.values()),
                        "manifest_self_hash_excluded": True, "source_bytes_normalized": False}
            (directory / packager.MANIFEST).write_bytes(packager.encoded(manifest))
            self.sources[task] = {**payloads, packager.MANIFEST: (directory / packager.MANIFEST).read_bytes()}

    def test_six_zip_exact_allowlist_readback_crc_sha_and_external_closure(self):
        prepared = packager.plan(self.stage, self.output)
        self.assertFalse(self.output.exists())
        closure = packager.package(prepared)
        self.assertEqual(closure["repository_count"], 6)
        self.assertTrue(closure["all_zip_member_sets_sizes_sha256_crc_verified"])
        self.assertTrue((self.output / "delivery-closure004.json").is_file())
        self.assertFalse(closure["source_or_manifest_bytes_modified"])
        self.assertFalse(closure["git_network_gpu_or_models_used"])
        for item in closure["repositories"]:
            archive_path = Path(item["zip"])
            self.assertEqual(packager.sha(archive_path.read_bytes()), item["zip_sha256"])
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), set(self.sources[item["task"]]))
                self.assertEqual(len(archive.namelist()), item["payload_files"] + 1)
                self.assertNotIn("delivery-closure004.json", archive.namelist())
                for name, raw in self.sources[item["task"]].items():
                    self.assertEqual(archive.read(name), raw)
                    self.assertEqual((self.stage / item["task"] / name).read_bytes(), raw)

    def test_dry_cli_uses_only_stdlib_and_writes_nothing(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/package_session004_repositories.py"),
                                 "--stage-root", str(self.stage), "--output-dir", str(self.output), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "dry_run_no_writes")
        self.assertEqual(len(report["repositories"]), 6)
        self.assertFalse(self.output.exists())
        text = (ROOT / "scripts/package_session004_repositories.py").read_text()
        self.assertNotIn("import torch", text)

    def test_extra_file_directory_or_stage_entry_refused_before_output(self):
        directory = self.stage / packager.TASKS[0]
        (directory / "extra.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "Unlisted source file"):
            packager.plan(self.stage, self.output)
        (directory / "extra.txt").unlink()
        (directory / "empty-extra").mkdir()
        with self.assertRaisesRegex(ValueError, "Unlisted source directory"):
            packager.plan(self.stage, self.output)
        (directory / "empty-extra").rmdir()
        (self.stage / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "exactly the six"):
            packager.plan(self.stage, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_or_changed_frozen_source_refused(self):
        path = self.stage / packager.TASKS[1] / "README.md"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing manifest"):
            packager.plan(self.stage, self.output)
        path.write_bytes(original + b"changed\n")
        with self.assertRaisesRegex(ValueError, "size/SHA"):
            packager.plan(self.stage, self.output)
        self.assertFalse(self.output.exists())

    def test_manifest_traversal_duplicate_alias_total_or_task_mismatch_refused(self):
        path = self.stage / packager.TASKS[0] / packager.MANIFEST
        original = json.loads(path.read_bytes())
        bad = []
        for name in ("../escape.py", "/absolute", "a\\b", "./README.md", ".", "CON.txt", "a:stream"):
            value = json.loads(json.dumps(original))
            value["files"][0]["path"] = name
            bad.append(value)
        duplicate = json.loads(json.dumps(original)); duplicate["files"].append(dict(duplicate["files"][0])); bad.append(duplicate)
        wrong = json.loads(json.dumps(original)); wrong["payload_bytes"] += 1; bad.append(wrong)
        wrong = json.loads(json.dumps(original)); wrong["primary_task"] = "other"; bad.append(wrong)
        wrong = json.loads(json.dumps(original)); wrong["files"][0]["bytes"] = True; bad.append(wrong)
        for value in bad:
            path.write_bytes(packager.encoded(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                packager.plan(self.stage, self.output)
        self.assertFalse(self.output.exists())

    def test_scan_errors_propagate(self):
        original_walk = packager.os.walk
        def fail_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("simulated denied directory"))
            return iter(())
        with patch.object(packager.os, "walk", side_effect=fail_walk):
            with self.assertRaisesRegex(PermissionError, "denied"):
                packager.plan(self.stage, self.output)
        self.assertFalse(self.output.exists())

    def test_source_and_output_ancestor_symlinks_refused(self):
        file = self.stage / packager.TASKS[0] / "README.md"
        external = self.base / "outside.txt"
        file.rename(external)
        try:
            file.symlink_to(external)
        except OSError as error:
            self.skipTest(str(error))
        with self.assertRaisesRegex(ValueError, "Symlink"):
            packager.plan(self.stage, self.output)
        outside = self.base / "outside"
        outside.mkdir()
        link = self.base / "output-link"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            packager.plan(self.stage, link / "new-zips")

    def test_old_stage_overwrite_and_output_inside_input_refused(self):
        with self.assertRaisesRegex(ValueError, "stage003"):
            packager.plan(self.base / "repositories-stage003", self.output)
        with self.assertRaisesRegex(ValueError, "stage003"):
            packager.plan(self.stage, self.base / "repositories-stage003/new")
        with self.assertRaisesRegex(ValueError, "inside"):
            packager.plan(self.stage, self.stage / "new-zips")
        with self.assertRaisesRegex(ValueError, "parents"):
            packager.plan(self.stage, self.output, self.output)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "NEW"):
            packager.plan(self.stage, self.output)

    def test_mutation_between_plan_and_package_refused_before_any_output(self):
        prepared = packager.plan(self.stage, self.output)
        path = self.stage / packager.TASKS[-1] / "README.md"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "size/SHA"):
            packager.package(prepared)
        self.assertFalse(self.output.exists())

    def test_zip_extra_missing_wrong_hash_or_manifest_refused(self):
        prepared = packager.plan(self.stage, self.output)
        repository = prepared["repositories"][0]
        task = repository["task"]
        for case in ("extra", "missing", "changed", "changed-manifest"):
            path = self.base / (case + ".zip")
            with zipfile.ZipFile(path, "x") as archive:
                for name, raw in self.sources[task].items():
                    if case == "missing" and name == "README.md":
                        continue
                    if case == "changed" and name == "README.md" or case == "changed-manifest" and name == packager.MANIFEST:
                        raw = b"changed"
                    archive.writestr(name, raw)
                if case == "extra":
                    archive.writestr("extra.txt", b"unexpected")
            with self.subTest(case=case), self.assertRaises(ValueError):
                packager.verify_zip(path, repository)

    def test_crc_corruption_is_refused_and_closure_can_be_outside_zip_directory(self):
        prepared = packager.plan(self.stage, self.output, self.base / "external-closure.json")
        closure = packager.package(prepared)
        self.assertTrue((self.base / "external-closure.json").is_file())
        self.assertFalse((self.output / "delivery-closure004.json").exists())
        repository = prepared["repositories"][0]
        target = self.base / "corrupted.zip"
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED) as archive:
            for name, raw in self.sources[repository["task"]].items():
                archive.writestr(name, raw)
        raw = target.read_bytes()
        offset = raw.index(b"CPU fixture only\n")
        target.write_bytes(raw[:offset] + b"X" + raw[offset+1:])
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            packager.verify_zip(target, repository)


if __name__ == "__main__":
    unittest.main()
