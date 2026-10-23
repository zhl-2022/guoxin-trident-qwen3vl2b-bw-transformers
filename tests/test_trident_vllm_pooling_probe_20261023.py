import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import trident_vllm_pooling_probe_20261023 as probe

EVIDENCE = ROOT / "evidence/trident-vllm-probe-20261023"


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}

    def decode(self, ids, **kwargs):
        return "x" * len(ids)


class FakeIds(list):
    def tolist(self):
        return list(self)


class Processor:
    tokenizer = Tokenizer()

    def apply_chat_template(self, conv, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return json.dumps(conv) + "<|im_start|>assistant\n"

    def __call__(self, **kwargs):
        assert kwargs["do_resize"] is False
        return {"input_ids": [FakeIds([1, 2, 3])]}


class PoolingProbeTests(unittest.TestCase):
    def setUp(self):
        self.mmemb = probe.read_json(EVIDENCE / "trident-mmemb-model.json")

    def test_exact_official_config_difference_is_only_registration_and_cache(self):
        left = probe.read_json(EVIDENCE / "trident-config.json")
        right = probe.read_json(EVIDENCE / "qwen-official-embedding-config.json")
        differences = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
        self.assertEqual(differences, {"architectures", "use_cache"})

    def test_all_tensor_keys_follow_vllm_mapper_without_projection(self):
        header = probe.read_json(EVIDENCE / "trident-safetensors-header.json")
        keys = [k for k in header if k != "__metadata__"]
        self.assertEqual(len(keys), 625)
        self.assertTrue(all(k.startswith(("model.language_model.", "model.visual.")) for k in keys))
        self.assertEqual(sum((v["data_offsets"][1] - v["data_offsets"][0]) for k,v in header.items() if k != "__metadata__"), 4255064064)

    def test_header_is_bounded_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            raw = (EVIDENCE / "trident-safetensors-header.json").read_bytes()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            self.assertEqual(probe.inspect_header(path)["tensor_count"], 625)
            path.write_bytes(struct.pack("<Q", 2_000_001))
            with self.assertRaises(ValueError):
                probe.inspect_header(path)

    def test_conversion_uses_last_fp32_normalization_not_chat(self):
        config = probe.engine_kwargs("/tmp/checkpoint")
        self.assertEqual(config["runner"], "pooling")
        self.assertEqual(config["convert"], "embed")
        self.assertEqual(config["pooler_config"], {"seq_pooling_type": "LAST", "use_activation": True})
        self.assertEqual(config["hf_overrides"]["head_dtype"], "float32")
        self.assertFalse(config["trust_remote_code"])
        self.assertNotIn("normalize", config["pooler_config"])

    def test_role_instructions_and_image_text_order_match_publisher(self):
        image = str((ROOT / "tasks/trident-qwen3vl2b-bw-transformers/fixtures/red-rectangle.png").resolve())
        row = {"text": "hello", "images": [image]}
        for role, instruction in (("query", "Represent the query for retrieving relevant content."),
                                  ("doc", "Represent the candidate content for retrieval.")):
            result = probe.conversation(row, role, Tokenizer(), self.mmemb)
            self.assertEqual(result[0]["content"][0]["text"], instruction)
            self.assertEqual([x["type"] for x in result[1]["content"]], ["image", "text"])
            self.assertEqual(result[1]["content"][0]["min_pixels"], 262144)
            self.assertEqual(result[1]["content"][0]["max_pixels"], 1310720)

    def test_text_truncates_before_template_not_at_image_placeholders(self):
        result = probe.conversation({"text": "a" * 5000, "images": []}, "query", Tokenizer(), self.mmemb)
        self.assertEqual(len(result[1]["content"][0]["text"]), 4096)

    def test_no_silent_video_or_remote_image_support(self):
        for row in ({"text": "x", "videos": ["test.mp4"]},
                    {"text": "x", "images": ["https://example.com/x.png"]},
                    {"text": "x", "instruction": "wrong"}):
            with self.assertRaises(ValueError):
                probe.normalize_inputs({"queries": [row], "documents": [{"text": "x"}]}, "unused.json")

    def test_cpu_prompt_contract_has_expected_ids_and_resized_image(self):
        class Image:
            size = (512, 512)
        seen = []
        def vision(conv, **kwargs):
            seen.append(kwargs)
            return [Image()], None, {}
        image = str((ROOT / "tasks/trident-qwen3vl2b-bw-transformers/fixtures/red-rectangle.png").resolve())
        inputs = {"queries": [{"text": "q", "images": []}], "documents": [{"text": None, "images": [image]}]}
        prompts, audit = probe.build_prompts(inputs, Processor(), vision, self.mmemb)
        self.assertNotIn("multi_modal_data", prompts[0])
        self.assertEqual(audit[1]["expected_prompt_token_ids"], [1, 2, 3])
        self.assertEqual(audit[1]["resized_image_sizes"], [[512, 512]])
        self.assertEqual(seen, [{"image_patch_size": 16, "return_video_metadata": True, "return_video_kwargs": True}])

    def test_reference_same_inputs_required_and_threshold_is_not_contest(self):
        vector = [1.0] + [0.0] * 2047
        embeddings = {"queries": [vector], "documents": [vector]}
        inputs = {"queries": [{"text": "q", "images": []}], "documents": [{"text": "d", "images": []}]}
        reference = {"mode": "publisher_embedding_backend", "inputs": inputs, "embeddings": embeddings}
        result = probe.compare_reference(embeddings, reference, inputs)
        self.assertTrue(result["engineering_agreement_passed"])
        self.assertFalse(result["contest_accuracy_measurement"])
        reference["inputs"] = {}
        with self.assertRaises(ValueError):
            probe.compare_reference(embeddings, reference, inputs)

    def test_plan_uses_no_heavy_import_and_flags_remain_false(self):
        process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/trident_vllm_pooling_probe_20261023.py"), "plan"],
                                 capture_output=True, text=True, check=True)
        result = json.loads(process.stdout)
        self.assertFalse(result["submission_ready"])
        self.assertFalse(result["official_accuracy_verified"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)

    def test_reference_snapshot_hash_and_missing_runtime_image_identity_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker-result.json"
            value = {"mode": "publisher_embedding_backend", "inputs": {}, "embeddings": {}}
            path.write_text(json.dumps(value), encoding="utf-8")
            identity = probe.reference_provenance(path)
            self.assertEqual(identity["reference_sha256"], probe.digest(path))
            self.assertFalse(identity["reference_input_image_bytes_at_runtime_verified"])
            self.assertFalse(identity["reference_source_identity_at_runtime_verified"])
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(identity["artifacts"][path.name]["content"], value)

    def test_wrong_reference_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker-result.json"
            path.write_text("{}", encoding="utf-8")
            (path.parent / "checkpoint-verification.json").write_text(json.dumps({"model_id": "wrong", "complete": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.reference_provenance(path)

    def test_worker_preflight_failure_preserves_structured_error_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            output.mkdir()
            arguments = ["probe", "run", "--_worker", "--model-dir", str(Path(directory)/"missing"), "--output", str(output)]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(sys, "platform", "linux"):
                with self.assertRaisesRegex(ValueError, "Checkpoint identity"):
                    probe.main()
            self.assertFalse(probe.read_json(output / "checkpoint-verification.json")["complete"])
            self.assertEqual(probe.read_json(output / "worker-error.json")["type"], "ValueError")
            self.assertNotIn("torch", sys.modules)

    def test_timeout_records_terminal_state_and_signals_only_owned_group(self):
        class Child:
            pid = 424242
            returncode = None
            calls = 0
            def wait(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("owned worker", 60)
                self.returncode = -15
                return -15
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            arguments = ["probe", "run", "--model-dir", str(Path(directory)/"missing"), "--output", str(output), "--max-seconds", "60"]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(sys, "platform", "linux"), \
                 mock.patch.object(probe.subprocess, "Popen", return_value=Child()) as popen, \
                 mock.patch.object(probe.os, "killpg", create=True) as killpg, \
                 mock.patch.object(probe.signal, "SIGKILL", 9, create=True), mock.patch.object(probe.time, "sleep"):
                self.assertEqual(probe.main(), 1)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual([call.args[0] for call in killpg.call_args_list], [424242, 424242])
            self.assertEqual(probe.read_json(output / "run-final.json")["status"], "timeout")
            self.assertTrue(probe.read_json(output / "run-start.json")["deadline_includes_checkpoint_hash_semantics_inputs_imports_and_gpu_worker"])


if __name__ == "__main__":
    unittest.main()
