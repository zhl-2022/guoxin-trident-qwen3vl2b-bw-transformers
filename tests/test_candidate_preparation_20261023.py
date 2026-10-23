"""No real network, package installation, checkpoint download, or GPU in tests."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_candidate_models_20261023 as downloader
import bootstrap_candidate_envs_20261023 as bootstrap

URL = "https://modelscope.cn/models/test/model/resolve/" + "a" * 40 + "/model.safetensors"


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}
    def geturl(self):
        return URL


class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
    def open(self, request, timeout=None):
        self.requests.append(request)
        return self.responses.pop(0)


class PreparationContracts(unittest.TestCase):
    def item(self, content=b"abcdef"):
        return {"path": "model.safetensors", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def partial(self, path, data, item):
        path.with_name(path.name + ".part").write_bytes(data)
        path.with_name(path.name + ".part.json").write_text(json.dumps({"size": item["size"], "sha256": item["sha256"]}))

    def test_reviewed_sources_exact_file_set_and_revisions(self):
        with mock.patch.object(downloader.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")):
            for task in downloader.TASKS:
                for provider in ("modelscope", "huggingface"):
                    original, files = downloader.load_files(task, provider)
                    self.assertEqual(len(files), len(original["files"]))
                    self.assertTrue(all("/master/" not in item["download_url"] for item in files))
                    self.assertTrue(all(len(item["sha256"]) == 64 for item in files))

    def test_download_domain_allowlist(self):
        for url in (URL, "https://cdn-lfs.huggingface.co/file", "https://cas-bridge.xethub.hf.co/file", "https://files.pythonhosted.org/a.whl"):
            self.assertTrue(downloader.allowed_download_url(url))
        for url in ("http://modelscope.cn/file", "https://modelscope.cn.evil.example/file", "https://user:pass@huggingface.co/file", "file:///tmp/a", "https://127.0.0.1/file"):
            self.assertFalse(downloader.allowed_download_url(url))

    def test_full_download_sha_and_atomic_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            result = downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([Response(b"abcdef")]))
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(result["status"], "downloaded_verified")
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_range_resume_validates_content_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            self.partial(path, b"abc", self.item())
            opener = Opener([Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "Content-Length": "3"})])
            result = downloader.download_file(self.item(), path, URL, time.monotonic() + 10, opener)
            self.assertEqual(opener.requests[0].get_header("Range"), "bytes=3-")
            self.assertEqual(result["resumed_from_bytes"], 3)
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_ignored_range_restarts_owned_partial_without_append(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            self.partial(path, b"abc", self.item())
            downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([Response(b"abcdef")]))
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_wrong_range_rejected_and_partial_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            self.partial(path, b"abc", self.item())
            with self.assertRaises(ValueError):
                downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([Response(b"def", 206, {"Content-Range": "bytes 0-2/6"})]))
            self.assertFalse(path.exists())
            self.assertEqual(path.with_name(path.name + ".part").read_bytes(), b"abc")

    def test_existing_bad_final_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            path.write_bytes(b"badbad")
            opener = Opener([])
            with self.assertRaises(ValueError):
                downloader.download_file(self.item(), path, URL, time.monotonic() + 10, opener)
            self.assertEqual(path.read_bytes(), b"badbad")
            self.assertEqual(opener.requests, [])

    def test_unowned_partial_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(b"old")
            with self.assertRaises(ValueError):
                downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([]))
            self.assertEqual(partial.read_bytes(), b"old")

    def test_wrong_download_digest_never_publishes_final(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            with self.assertRaises(ValueError):
                downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([Response(b"xxxxxx")]))
            self.assertFalse(path.exists())
            self.assertEqual(path.with_name(path.name + ".part").read_bytes(), b"xxxxxx")

    def test_deadline_stops_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            opener = Opener([])
            with self.assertRaises(TimeoutError):
                downloader.download_file(self.item(), Path(folder) / "model", URL, time.monotonic() - 1, opener)
            self.assertEqual(opener.requests, [])

    def test_valid_existing_final_reused_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            path.write_bytes(b"abcdef")
            result = downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([]))
            self.assertEqual(result["status"], "reused_verified")

    def test_complete_owned_partial_verified_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            self.partial(path, b"abcdef", self.item())
            result = downloader.download_file(self.item(), path, URL, time.monotonic() + 10, Opener([]))
            self.assertEqual(result["status"], "promoted_verified_partial")
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_interrupted_response_resumes_owned_bytes(self):
        class Broken(Response):
            def read(self, amount=-1):
                if self.tell() >= 3:
                    raise OSError("simulated disconnected socket")
                return super().read(3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            opener = Opener([Broken(b"abcdef"), Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "Content-Length": "3"})])
            with mock.patch.object(downloader.time, "sleep"):
                downloader.download_file(self.item(), path, URL, time.monotonic() + 10, opener)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(opener.requests[1].get_header("Range"), "bytes=3-")

    def test_dependency_resolution_prefers_satisfying_native_cpu_package(self):
        native = {"alpha": {"name": "alpha", "version": "1.5", "requires_dist": []}}
        pool = [{"name": "alpha", "version": "2.0", "requires_dist": []}]
        values = bootstrap.resolve_packages(["alpha>=1"], pool, native.get)
        self.assertEqual(values[0]["mode"], "reuse_native")
        self.assertEqual(values[0]["version"], "1.5")

    def test_exact_shadow_and_protected_native_dependency(self):
        native = {"transformers": {"name": "transformers", "version": "5.5.0", "requires_dist": []},
                  "torch": {"name": "torch", "version": "2.10.0+das.test", "requires_dist": []}}
        pool = [{"name": "transformers", "version": "4.57.6", "requires_dist": ["torch>=2.6"]}]
        values = bootstrap.resolve_packages(["transformers==4.57.6"], pool, native.get)
        self.assertEqual({x["name"]: x["mode"] for x in values}, {"torch": "reuse_native", "transformers": "install_cpu_wheel"})

    def test_gpu_wheel_and_unreviewed_dependency_fail_closed(self):
        with self.assertRaises(ValueError):
            bootstrap.resolve_packages(["torch>=2"], [{"name": "torch", "version": "2.10.0"}], lambda name: None)
        with self.assertRaises(ValueError):
            bootstrap.resolve_packages(["missing>=1"], [], lambda name: None)
        with self.assertRaises(ValueError):
            bootstrap.resolve_packages(["alpha @ https://example.com/x.whl"], [], lambda name: None)

    def test_full_forced_and_trident_pool_closure_without_install(self):
        captured = json.loads((ROOT / "tasks/qwen3-forcedaligner-bw-transformers/env-plan.json").read_text())["protected_native_expectations"]
        native = {name: {"name": name, "version": version, "root": "/native", "requires_dist": []} for name, version in captured.items()}
        for task in ("forcedaligner", "trident"):
            plan = bootstrap.build_plan(task, "publisher-pinned", native.get)
            self.assertGreater(plan["install_count"], 0)
            self.assertFalse(any(bootstrap.canonical(x["name"]) in bootstrap.PROTECTED and x["mode"] != "reuse_native" for x in plan["selected"]))

    def test_trident_native_first_reuses_existing_tf5(self):
        captured = json.loads((ROOT / "tasks/trident-qwen3vl2b-bw-transformers/env-plan.json").read_text())["protected_native_expectations"]
        native = {name: {"name": name, "version": version, "root": "/native", "requires_dist": []} for name, version in captured.items()}
        native["transformers"] = {"name": "transformers", "version": "5.5.0", "root": "/native", "requires_dist": []}
        plan = bootstrap.build_plan("trident", "native-first", native.get)
        item = next(x for x in plan["selected"] if bootstrap.canonical(x["name"]) == "transformers")
        self.assertEqual(item["version"], "5.5.0")
        self.assertEqual(item["mode"], "reuse_native")

    def test_apply_invokes_pip_only_with_new_venv_python_and_hash_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "candidates"
            wheel = {"name": "alpha", "version": "1.0", "mode": "install_cpu_wheel", "filename": "alpha-1.0-py3-none-any.whl",
                     "url": "https://files.pythonhosted.org/alpha.whl", "sha256": "a" * 64, "size": 1, "requires_dist": []}
            plan = {"selected": [wheel], "protected_native_expectations": {"torch": "native"}, "framework_eligibility": "pending"}
            result = mock.Mock(returncode=0, stdout='{"gpu_initialized":false}', stderr="")
            def create(destination):
                Path(destination).mkdir(parents=True)
            builder = mock.Mock()
            builder.create.side_effect = create
            argv = ["bootstrap", "apply", "--task", "forcedaligner"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "platform", "linux"), mock.patch.object(sys, "version_info", (3, 10, 0)), mock.patch.object(bootstrap.platform, "machine", return_value="x86_64"), mock.patch.object(bootstrap.platform, "libc_ver", return_value=("glibc", "2.35")), mock.patch.object(bootstrap, "BASE", base), mock.patch.object(bootstrap, "build_plan", return_value=plan), mock.patch.object(bootstrap, "native_snapshot", return_value={"torch": {"version": "native", "root": "/native"}}), mock.patch.object(bootstrap, "download_file"), mock.patch.object(bootstrap.venv, "EnvBuilder", return_value=builder) as factory, mock.patch.object(bootstrap.subprocess, "run", return_value=result) as execute, mock.patch("builtins.print"):
                self.assertEqual(bootstrap.main(), 0)
            factory.assert_called_once_with(with_pip=False, system_site_packages=True, clear=False)
            first = execute.call_args_list[0].args[0]
            self.assertEqual(first[0], str(base / "envs/forcedaligner-001/bin/python"))
            for flag in ("--no-index", "--no-deps", "--ignore-installed", "--require-hashes", "--only-binary=:all:"):
                self.assertIn(flag, first)
            self.assertNotEqual(first[0], sys.executable)


if __name__ == "__main__":
    unittest.main()
