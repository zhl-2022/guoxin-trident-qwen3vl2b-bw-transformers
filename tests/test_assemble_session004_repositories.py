"""CPU contract tests for the new assembler; build only in fresh temp dirs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("session004_assembler", ROOT / "scripts/assemble_session004_repositories.py")
assembler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assembler
SPEC.loader.exec_module(assembler)


class AssemblerTests(unittest.TestCase):
    def test_archived_asr_metadata_exception_does_not_admit_environment(self):
        prefix = "evidence/session004/export-final/project/tasks/qwen3-asr-bw-vllm/runtime/audio-venv"
        self.assertTrue(assembler.archived_asr_prepare(prefix, directory=True))
        self.assertTrue(assembler.archived_asr_prepare(prefix + "/asr-prepare-evidence", directory=True))
        for leaf in assembler.ASR_PREPARE_FILES:
            name = prefix + "/asr-prepare-evidence/" + leaf
            self.assertFalse(assembler.forbidden(name, archived=True))
            self.assertTrue(assembler.forbidden(name, archived=False))
        for suffix in ("/pyvenv.cfg", "/lib/python/site-packages/package.py",
                       "/asr-prepare-evidence/unlisted.json", "/asr-prepare-evidence/nested/ready.json"):
            self.assertTrue(assembler.forbidden(prefix + suffix, archived=True))

    def test_archived_asr_metadata_tree_accepts_six_and_refuses_venv_config(self):
        folder = "evidence/session004/export-final"
        prefix = folder + "/project/tasks/qwen3-asr-bw-vllm/runtime/audio-venv"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / prefix / "asr-prepare-evidence"
            evidence.mkdir(parents=True)
            for leaf in assembler.ASR_PREPARE_FILES:
                (evidence / leaf).write_text("{}", encoding="utf-8")
            collector = assembler.Collector(root)
            collector.tree(folder, archived=True)
            expected = {prefix + "/asr-prepare-evidence/" + leaf for leaf in assembler.ASR_PREPARE_FILES}
            self.assertEqual(set(collector.payloads), expected)
            self.assertEqual(collector.archived_names, expected)
            (root / prefix / "pyvenv.cfg").write_text("include-system-site-packages = true")
            with self.assertRaisesRegex(ValueError, "Runtime/credential/weight"):
                assembler.Collector(root).tree(folder, archived=True)

    @classmethod
    def setUpClass(cls):
        # Reports now explicitly cite authenticated run exports. Pass these as
        # archived inputs just like the delivery CLI; live runtime stays denied.
        evidence_root = ROOT / "evidence/session004"
        cls.archives = tuple(p.relative_to(ROOT).as_posix() for p in evidence_root.iterdir()
                             if p.is_dir() and (p / "export-manifest.json").is_file())
        cls.sources = assembler.common_sources(ROOT, cls.archives)
        cls.prepared = {item.slug: assembler.task_payloads(cls.sources, item) for item in assembler.REGISTRY}

    def test_registry_six_reports_and_scope_are_explicit(self):
        self.assertEqual(len(assembler.REGISTRY), 6)
        self.assertEqual(sum(spec.service for spec in assembler.REGISTRY), 5)
        self.assertEqual([spec.slug for spec in assembler.REGISTRY if not spec.service],
                         ["qwen3-forcedaligner-bw-transformers"])
        for spec in assembler.REGISTRY:
            files = self.prepared[spec.slug]
            status = json.loads(files["ROOT_TASK.json"])
            self.assertEqual(status["model"], spec.model)
            self.assertFalse(status["formal_ready"])
            self.assertFalse(status["submitted"])
            self.assertEqual(status["standard_service_entry_complete"], spec.service)
            assembler.report_check(files["README.md"])
            if spec.service:
                self.assertIn("verify.py", files)
                self.assertIn("smoke.sh", files)
            else:
                self.assertIn("entry.py", files)
                self.assertNotIn("run.sh", files)
                self.assertNotIn("verify.py", files)
                self.assertIs(status["framework_eligibility_confirmed"], False)

    def test_trident_standard_entries_do_not_upgrade_http_or_official_evidence(self):
        files = self.prepared[assembler.TRIDENT]
        status = json.loads(files["ROOT_TASK.json"])
        for entry in assembler.STANDARD_SHELLS:
            self.assertEqual(files[entry], assembler.wrapper(assembler.TRIDENT, entry))
        self.assertIn("verify.py", files)
        self.assertNotIn("entry.py", files)
        self.assertTrue(status["standard_service_entry_complete"])
        self.assertEqual(status["framework"], "vllm")
        self.assertEqual(status["framework_execution_mode_reported_by_task"], "native_vllm_pooling_conversion")
        self.assertTrue(status["offline_pooling_GPU_verified_reported_by_task"])
        self.assertFalse(status["runtime_evidence_source"]["assembler_independently_verified_GPU_execution"])
        for field in ("HTTP_service_GPU_verified", "official_embedding_evaluator_bridge_implemented",
                      "standard_submission_ready"):
            self.assertIs(status[field], False)
            changed = dict(files)
            changed["ROOT_TASK.json"] = assembler.encoded({**status, field: True})
            with self.assertRaisesRegex(ValueError, "scope differs"):
                assembler.validate_payloads(assembler.TRIDENT, changed, self.sources.archived_names)

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "actual Trident wrapper requires Linux Bash")
    def test_trident_root_selfcheck_reports_missing_bridge_without_evaluation(self):
        task = assembler.TRIDENT
        names = ["verify.py", "verify.sh", f"tasks/{task}/verify.sh", f"tasks/{task}/env.sh",
                 f"tasks/{task}/task.json", "scripts/trident_embedding_runtime_004.py",
                 "scripts/recon2reason_runtime.py", "scripts/trident_vllm_pooling_probe_20261023.py",
                 "scripts/task_expansion_candidates_20261023.py"]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in names:
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.prepared[task][name])
            env = {key: value for key, value in os.environ.items() if not key.startswith("TRIDENT_")}
            env["TRIDENT_PYTHON_BIN"] = sys.executable
            result = subprocess.run([sys.executable, "-B", str(base / "verify.py"), "--selfcheck"],
                                    capture_output=True, text=True, env=env, timeout=15)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            output = json.loads(result.stdout)
            self.assertFalse(output["executed"])
            self.assertFalse(output["official_accuracy_verified"])
            self.assertFalse(any(base.rglob("verify_result.json")))

    def test_report_links_rebase_from_docs_preserves_fences_references_and_suffixes(self):
        raw = ("[fixture](../fixtures/smoke-session004/manifest.json#fixture)\n"
               "[fields]: <stage004-report-fields.json?x=1#f> \"title\"\n"
               "```bash\n[keep](../tasks/x)\nbash run.sh --dry-run\n```\n").encode()
        changed = assembler.rebase_report(ROOT, assembler.REGISTRY[0].report, raw).decode()
        self.assertIn("(fixtures/smoke-session004/manifest.json#fixture)", changed)
        self.assertIn("<docs/stage004-report-fields.json?x=1#f>", changed)
        self.assertIn("[keep](../tasks/x)\nbash run.sh --dry-run", changed)
        self.assertNotIn("../fixtures/", changed)
        escaped = assembler.rebase_report(ROOT, "docs/report.md", b"[space](../docs/test%20file.md#f)").decode()
        self.assertEqual(escaped, "[space](docs/test%20file.md#f)")

    def test_report_rejects_ninth_or_unnumbered_chapter(self):
        valid = "\n".join(f"## {number}. chapter" for number in range(1, 9)).encode()
        assembler.report_check(valid)
        for extra in (b"\n## 9. ninth", b"\n## extra"):
            with self.assertRaisesRegex(ValueError, "eight"):
                assembler.report_check(valid + extra)

    def test_standard_python_and_shell_wrappers_transparently_forward_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for spec in assembler.REGISTRY:
                filename = "verify.py" if spec.service else "entry.py"
                canonical = "verify.sh" if spec.service else "entry.py"
                root_entry = base / spec.slug / filename
                root_entry.parent.mkdir()
                root_entry.write_bytes(self.prepared[spec.slug][filename])
                target = root_entry.parent / "tasks" / spec.slug / canonical
                target.parent.mkdir(parents=True)
                # Intercept the exec boundary; this works without local Bash.
                code = ("import runpy,sys,os,json\n"
                        "def capture(*a):\n print(json.dumps(a)); raise SystemExit(19)\n"
                        "os.execvp=capture; os.execv=capture\n"
                        "sys.argv=[sys.argv[1], '--selfcheck', 'space value', '--dry-run']\n"
                        "runpy.run_path(sys.argv[0],run_name='__main__')\n")
                result = subprocess.run([sys.executable, "-B", "-c", code, str(root_entry)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 19, result.stderr)
                command = json.loads(result.stdout)[1]
                self.assertEqual(command[-3:], ["--selfcheck", "space value", "--dry-run"])
                self.assertEqual(Path(command[1]), target)
                if spec.service:
                    self.assertEqual(command[0], "bash")
                    self.assertIn(b'"$@"', self.prepared[spec.slug]["smoke.sh"])

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "actual root shell semantics require Linux Bash")
    def test_actual_bash_wrapper_and_missing_official_selfcheck_exit_two(self):
        spec = assembler.REGISTRY[3]  # MiMo: no model download or venv preparation.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            minimal = {
                "verify.py": self.prepared[spec.slug]["verify.py"],
                "verify.sh": self.prepared[spec.slug]["verify.sh"],
                f"tasks/{spec.slug}/verify.sh": self.prepared[spec.slug][f"tasks/{spec.slug}/verify.sh"],
                f"tasks/{spec.slug}/env.sh": self.prepared[spec.slug][f"tasks/{spec.slug}/env.sh"],
                f"tasks/{spec.slug}/task.json": self.prepared[spec.slug][f"tasks/{spec.slug}/task.json"],
                "scripts/recon2reason_runtime.py": self.prepared[spec.slug]["scripts/recon2reason_runtime.py"],
            }
            for name, raw in minimal.items():
                file = base / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(raw)
            result = subprocess.run([sys.executable, "-B", str(base / "verify.py"), "--selfcheck", "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(any(base.rglob("verify_result.json")))
            # Check real Bash argv delivery separately without the bridge.
            (base / f"tasks/{spec.slug}/verify.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\nexit 17\n')
            result = subprocess.run(["bash", str(base / "verify.sh"), "--selfcheck", "space value"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 17)
            self.assertEqual(result.stdout.splitlines(), ["--selfcheck", "space value"])

    def test_temp_build_six_manifests_readback_and_existing_directory_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-stage004"
            results = assembler.build(ROOT, output, self.prepared, self.sources.archived_names)
            self.assertEqual(len(results), 6)
            for item in results:
                directory = output / item["task"]
                manifest = json.loads((directory / assembler.MANIFEST).read_bytes())
                self.assertNotIn(assembler.MANIFEST, {row["path"] for row in manifest["files"]})
                for row in manifest["files"]:
                    raw = (directory / row["path"]).read_bytes()
                    self.assertEqual(len(raw), row["bytes"])
                    self.assertEqual(assembler.digest(raw), row["sha256"])
                self.assertFalse((directory / ".git").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                assembler.build(ROOT, output, self.prepared, self.sources.archived_names)

    def test_dry_plan_does_not_create_output_and_rejects_unknown_task_or_old_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-stage004"
            with patch.object(assembler, "common_sources", return_value=self.sources):
                _, prepared = assembler.plan(ROOT, output)
            self.assertEqual(set(prepared), set(assembler.SPECS))
            self.assertFalse(output.exists())
            for bad in (("unknown",), (assembler.REGISTRY[0].slug,) * 2):
                with self.assertRaises(ValueError):
                    assembler.plan(ROOT, output, bad)
        with self.assertRaisesRegex(ValueError, "stage003"):
            assembler.output_check(ROOT, ROOT / "dist/repositories-stage003/new")
        with self.assertRaisesRegex(ValueError, "canonical source"):
            assembler.output_check(ROOT, ROOT / "tasks/no-write-new")

    def test_filesystem_scan_error_raises_instead_of_silent_omission(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "scripts").mkdir()
            collector = assembler.Collector(base)
            def broken_walk(*args, **kwargs):
                kwargs["onerror"](PermissionError("simulated directory access denied"))
                return iter(())
            with patch.object(assembler.os, "walk", side_effect=broken_walk):
                with self.assertRaisesRegex(PermissionError, "denied"):
                    collector.tree("scripts")

    def test_symlink_source_and_output_ancestor_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "source").mkdir()
            (base / "outside").mkdir()
            try:
                (base / "source/link").symlink_to(base / "outside", target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))
            with self.assertRaisesRegex(ValueError, "Symlink"):
                assembler.Collector(base / "source").tree("link")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                assembler.output_check(ROOT, base / "source/link/new-stage")

    def test_unapproved_zip_weight_runtime_credential_and_mutated_template_refused(self):
        for name in ("tasks/x/weights.bin", "tasks/x/runtime/result.json", "scripts/.env", "evidence/secrets.json", "tasks/x/source.zip"):
            with self.assertRaises(ValueError, msg=name):
                assembler.scan(name, b"{}")
        for name, raw in (("evidence/x.log", b"pass" + b'word = "abcdefgh123456"'),
                          ("scripts/x.py", b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----"),
                          (assembler.TEMPLATE, b"changed template")):
            with self.assertRaises(ValueError, msg=name):
                assembler.scan(name, raw)
        assembler.scan("evidence/session004/exported/runtime/server.log", b"raw capture", archived=True)
        with self.assertRaises(ValueError):
            assembler.scan("tasks/x/runtime/server.log", b"raw", archived=True)

    def test_archive_crc_hash_and_fixed_lock_enforced(self):
        for name in assembler.LOCKED_ZIPS:
            lock = self.sources.payloads[str(Path(name).with_name("upstream-lock.json")).replace("\\", "/")]
            raw = self.sources.payloads[name]
            assembler.validate_upstream_archive(name, raw, lock)
            changed = raw[:-1] + bytes([raw[-1] ^ 1])
            with self.assertRaisesRegex(ValueError, "SHA/revision"):
                assembler.validate_upstream_archive(name, changed, lock)
            parsed = json.loads(lock)
            parsed["files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "member SHA"):
                assembler.validate_upstream_archive(name, raw, assembler.encoded(parsed))

    def test_build_policy_failure_happens_before_any_output_creation(self):
        slug = assembler.REGISTRY[0].slug
        # A valid archived payload must pass before testing each mutation;
        # otherwise missing archived_names can mask the intended failure.
        assembler.validate_payloads(slug, self.prepared[slug], self.sources.archived_names)
        for changed in ({**self.prepared[slug], "../escape.py": b"x"},
                        {**self.prepared[slug], "scripts/new.bin": b"bad"},
                        {**self.prepared[slug], "run.sh": b"fake wrapper"}):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "must-not-exist"
                with self.assertRaises(ValueError):
                    assembler.build(ROOT, output, {slug: changed}, self.sources.archived_names)
                self.assertFalse(output.exists())

    def test_no_live_runtime_or_weights_are_collected(self):
        self.assertNotIn("tasks/recon2reason-bw-sglang/task.json", self.sources.payloads)
        for name in self.sources.payloads:
            self.assertFalse(assembler.forbidden(name, archived=name in self.sources.archived_names), name)
            if "/runtime/" in name:
                self.assertTrue(name.startswith("evidence/session004/"), name)
                self.assertIn(name, self.sources.archived_names)


if __name__ == "__main__":
    unittest.main()
