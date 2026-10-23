"""CPU-only contracts: none of these tests load weights, Torch, or a GPU."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expansion_candidates", ROOT / "scripts/task_expansion_candidates_20261023.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ExpansionContracts(unittest.TestCase):
    def test_offline_plan_all_three_no_torch_import_or_download(self):
        for task in runner.SLUGS:
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "tasks" / task / "entry.py"), "plan"], capture_output=True, text=True, check=True)
            data = json.loads(completed.stdout)
            self.assertFalse(data["gpu_imported"])
            self.assertFalse(data["downloads_performed"])
            self.assertFalse(data["submission_ready"])
            self.assertEqual(data["task"], task)
        self.assertLess(sum(json.loads((ROOT / "tasks" / slug / "model-lock.json").read_text())["weight_bytes"] for slug in runner.SLUGS), 25_000_000_000)

    def test_missing_checkpoint_cannot_pass(self):
        result = runner.verify_checkpoint(ROOT / "this-checkpoint-does-not-exist", {"files": []})
        self.assertFalse(result["complete"])

    def test_checkpoint_metadata_and_weights_must_both_match(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            weights, config = b"tiny fake test weights", b'{"model_type":"test"}'
            (base / "model.safetensors").write_bytes(weights)
            (base / "config.json").write_bytes(config)
            lock = {"model_id": "test", "revision": "test", "files": [
                {"path": "model.safetensors", "size": len(weights), "sha256": hashlib.sha256(weights).hexdigest()},
                {"path": "config.json", "size": len(config), "git_blob_sha1": hashlib.sha1(f"blob {len(config)}\0".encode() + config).hexdigest()}]}
            self.assertFalse(runner.verify_checkpoint(base, lock)["complete"])
            self.assertTrue(runner.verify_checkpoint(base, lock, True)["complete"])
            (base / "config.json").write_bytes(config.replace(b"test", b"evil"))
            self.assertFalse(runner.verify_checkpoint(base, lock, True)["complete"])

    def test_unsafe_manifest_paths_rejected(self):
        for name in ("../x", "/root/x", "C:/x", "a\\b", "a//b", "./x"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.safe_relative(name)

    def test_publisher_source_archive_extract_and_tamper_detection(self):
        task = ROOT / "tasks" / runner.SLUGS[0]
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "new"
            self.assertTrue(runner.extract_source(task, destination)["complete"])
            api = destination / "qwen_asr/inference/qwen3_forced_aligner.py"
            api.write_bytes(api.read_bytes() + b"# changed\n")
            self.assertFalse(runner.verify_source(destination, runner.read_json(task / "upstream-lock.json"))["complete"])
            with self.assertRaises(ValueError):
                runner.extract_source(task, destination)

    def test_publisher_source_rejects_untracked_import_shadow(self):
        task = ROOT / "tasks" / runner.SLUGS[0]
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "new"
            runner.extract_source(task, destination)
            (destination / "torch.py").write_text("# unexpected shadow")
            result = runner.verify_source(destination, runner.read_json(task / "upstream-lock.json"))
            self.assertFalse(result["complete"])
            self.assertEqual(result["extra_executable_sources"], ["torch.py"])

    def test_alignment_preserves_chinese_and_english_units(self):
        items = [{"text": "你", "start_time": 0.0, "end_time": 0.2}, {"text": "好", "start_time": 0.2, "end_time": 0.4}]
        self.assertTrue(runner.validate_alignment(items, "你好！", 1)["text_units_preserved"])
        english = [{"text": "It's", "start_time": 0.0, "end_time": 0.3}, {"text": "fine", "start_time": 0.4, "end_time": 0.6}]
        self.assertTrue(runner.validate_alignment(english, "It's fine.", 1)["text_units_preserved"])

    def test_alignment_rejects_empty_changed_nan_unordered_outside(self):
        bad = [[], [{"text": "x", "start_time": 0, "end_time": 1}],
               [{"text": "a", "start_time": float("nan"), "end_time": 1}],
               [{"text": "a", "start_time": 0.4, "end_time": 0.3}],
               [{"text": "a", "start_time": 0, "end_time": 1.5}],
               [{"text": "a", "start_time": True, "end_time": 1}]]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_alignment(value, "a", 1)

    def test_alignment_order_rejects_backtracking(self):
        items = [{"text": "a", "start_time": 0.5, "end_time": 0.8}, {"text": "b", "start_time": 0.3, "end_time": 0.7}]
        with self.assertRaises(ValueError):
            runner.validate_alignment(items, "ab", 1)

    def test_audio_contract_checks_actual_wav_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sound.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            self.assertEqual(runner.audio_contract(path)["duration_seconds"], 1.0)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            with self.assertRaises(ValueError):
                runner.audio_contract(path)

    def test_embedding_similarity_is_real_dot_product(self):
        result = runner.validate_embeddings([[1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
        self.assertEqual(result["cosine_scores"], [[0.0, 1.0, -1.0]])
        self.assertEqual(result["ranked_document_indices"], [[1, 0, 2]])
        self.assertFalse(result["retrieval_accuracy_against_qrels_evaluated"])

    def test_embedding_rejects_nonunit_mismatched_nan_vectors(self):
        for docs in ([[2.0, 0.0]], [[1.0]], [[float("nan"), 0.0]], [[True, 0.0]], []):
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                runner.validate_embeddings([[1.0, 0.0]], docs)

    def test_mimo_completion_requires_stop_model_visible_text_and_usage(self):
        response = {"model": "mimo", "choices": [{"message": {"content": "<think>15 percent</think>36"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}}
        self.assertTrue(runner.validate_mimo_response(response, "mimo")["raw_thinking_preserved"])
        for wrong in (
            {**response, "model": "other"},
            {**response, "choices": [{"message": {"content": "36"}, "finish_reason": "length"}]},
            {**response, "choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            {**response, "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 99}},
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                runner.validate_mimo_response(wrong, "mimo")

    def test_mimo_worker_preserves_thinking_and_never_replaces_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(task=runner.SLUGS[1], port=30109, model_dir=folder, max_seconds=900)
            output = Path(folder)
            child = mock.Mock(pid=12345)
            child.poll.side_effect = [None, None, 0]
            child.wait.return_value = 0
            child.returncode = 0
            response = {"model": "mimo-v2.6-distill-qwen-9b-candidate", "choices": [{"message": {"content": "36"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
            calls = []
            def fake_request(url, payload=None, timeout=10):
                calls.append((url, payload))
                return {"data": [{"id": response["model"]}]} if payload is None else response
            with mock.patch.object(runner.socket, "socket"), mock.patch.object(runner.subprocess, "Popen", return_value=child) as popen, mock.patch.object(runner, "request_json", side_effect=fake_request):
                runner.worker(args, ROOT / "tasks" / args.task, output)
            command = popen.call_args.args[0]
            self.assertNotIn("--tokenizer", command)
            self.assertNotIn("--reasoning-parser", command)
            payload = calls[-1][1]
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(payload["max_tokens"], 2048)
            self.assertNotIn("temperature", payload)
            child.terminate.assert_called_once()

    def test_trident_worker_calls_embedding_backend_with_distinct_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            request_path = directory / "requests.json"
            request_path.write_text(json.dumps({"queries": [{"text": "red", "images": []}], "documents": [{"text": "red", "images": []}]}))
            calls = []
            class Array:
                def tolist(self): return [[1.0, 0.0]]
            class Backend:
                def __init__(self, args, device): self.args_dict = args
                def load(self): calls.append("load")
                def default_instruction(self, role): return "query instruction" if role == "query" else "document instruction"
                def process(self, inputs, role):
                    calls.append((inputs, role))
                    return Array(), None
            common = types.ModuleType("embedding_backends.common")
            common.ensure_tokenizers_thread_safety = lambda: None
            common.ensure_transformers_special_tokens_compat = lambda: None
            backend_module = types.ModuleType("embedding_backends.trident_qwen3vl")
            backend_module.TridentQwen3VLBackend = Backend
            args = argparse.Namespace(task=runner.SLUGS[2], upstream_root=str(directory), model_dir=str(directory), requests=str(request_path))
            with mock.patch.dict(sys.modules, {"embedding_backends": types.ModuleType("embedding_backends"), "embedding_backends.common": common, "embedding_backends.trident_qwen3vl": backend_module}), mock.patch.object(runner, "verify_source", return_value={"complete": True}):
                result = runner.worker(args, ROOT / "tasks" / args.task, directory)
            self.assertEqual(result["mode"], "publisher_embedding_backend")
            self.assertEqual([x[1] for x in calls[1:]], ["query", "doc"])
            self.assertNotEqual(calls[1][0][0]["instruction"], calls[2][0][0]["instruction"])

    def test_fixture_paths_resolve_from_request_file_and_reject_urls(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "sound.wav").write_bytes(b"fixture")
            self.assertEqual(runner.fixture_path("sound.wav", base / "requests.json"), base.resolve() / "sound.wav")
            with self.assertRaises(ValueError):
                runner.fixture_path("https://example.com/audio.wav", base / "requests.json")

    def test_forced_worker_uses_exact_publisher_api_and_known_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            audio_path = directory / "audio.wav"
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            requests = directory / "requests.json"
            requests.write_text(json.dumps([{"audio": "audio.wav", "text": "a", "language": "English"}]))
            calls = []
            class FakeAligner:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls.append((path, kwargs))
                    return cls()
                def align(self, **kwargs):
                    calls.append(kwargs)
                    return [[types.SimpleNamespace(text="a", start_time=0.0, end_time=0.5)]]
            torch = types.ModuleType("torch")
            torch.bfloat16 = "fake_bf16"
            package = types.ModuleType("qwen_asr")
            package.Qwen3ForcedAligner = FakeAligner
            expected_api = directory / "qwen_asr/inference/qwen3_forced_aligner.py"
            args = argparse.Namespace(task=runner.SLUGS[0], upstream_root=str(directory), model_dir=str(directory), requests=str(requests))
            with mock.patch.dict(sys.modules, {"torch": torch, "qwen_asr": package}), mock.patch.object(runner, "verify_source", return_value={"complete": True}), mock.patch("inspect.getfile", return_value=str(expected_api)):
                result = runner.worker(args, ROOT / "tasks" / args.task, directory)
            self.assertEqual(calls[0][1]["attn_implementation"], "sdpa")
            self.assertEqual(calls[1]["text"], "a")
            self.assertEqual(calls[1]["language"], "English")
            self.assertTrue(result["results"][0]["validation"]["text_units_preserved"])

    def test_outer_deadline_signals_only_the_owned_process_group(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-output"
            child = mock.Mock(pid=98765, returncode=-15)
            child.wait.side_effect = [subprocess.TimeoutExpired("test", 60), -15]
            argv = ["candidate", runner.SLUGS[1], "run", "--output", str(output), "--max-seconds", "60"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "platform", "linux"), mock.patch.object(runner, "verify_checkpoint", return_value={"complete": True}), mock.patch.object(runner.subprocess, "Popen", return_value=child), mock.patch.object(runner.os, "killpg", create=True) as kill, mock.patch.object(runner.signal, "SIGKILL", 9, create=True), mock.patch.object(runner.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                code = runner.main()
            self.assertEqual(code, 1)
            self.assertEqual([call.args[0] for call in kill.call_args_list], [98765, 98765])
            result = runner.read_json(output / "run-final.json")
            self.assertEqual(result["status"], "timeout")
            self.assertFalse(result["submission_ready"])


if __name__ == "__main__":
    unittest.main()
