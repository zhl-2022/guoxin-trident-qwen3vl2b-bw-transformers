"""Exercise model header checks with tiny fixtures; never import real torch."""

import importlib.util
import contextlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scnet_preflight.py"
SPEC = importlib.util.spec_from_file_location("scnet_preflight_under_test", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def tensor(offsets, *, shape=None, dtype="BF16"):
    return {"dtype": dtype, "shape": [2] if shape is None else shape, "data_offsets": offsets}


class SCNetPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (self.model / name).write_text("{}", encoding="utf-8")
        (self.model / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3VLForConditionalGeneration"], "torch_dtype": "bfloat16"}),
            encoding="utf-8",
        )
        self.index = {
            "metadata": {"total_size": 8},
            "weight_map": {"layer.a": "model-1.safetensors", "layer.b": "model-2.safetensors"},
        }
        self.write_index()
        self.write_shard("model-1.safetensors", {"layer.a": tensor([0, 4])}, b"\0" * 4)
        self.write_shard("model-2.safetensors", {"layer.b": tensor([0, 4])}, b"\0" * 4)
        self.addCleanup(patch.stopall)
        patch.object(preflight.shutil, "which", return_value=None).start()
        patch.object(preflight.metadata, "version", return_value="fixture-no-import").start()

    def write_index(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps(self.index), encoding="utf-8")

    def write_shard(self, name, header, payload):
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)
        path = self.model / name
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        return path

    def collect(self):
        return preflight.collect(self.model)

    def assert_incomplete(self):
        result = self.collect()
        self.assertFalse(result["model_checks"]["complete"])
        return result

    def test_two_shards_are_complete_without_loading_torch(self):
        with patch("builtins.__import__", wraps=__import__) as importer:
            result = self.collect()
        self.assertTrue(result["model_checks"]["complete"])
        self.assertEqual(result["model_checks"]["layout"], "indexed")
        self.assertTrue(result["model_checks"]["index_present"])
        self.assertTrue(result["model_checks"]["expected_architecture"])
        self.assertEqual(result["model_checks"]["expected_architecture_name"], "Qwen3VLForConditionalGeneration")
        self.assertEqual(result["model_checks"]["index_weight_count"], 2)
        self.assertEqual(result["model_checks"]["payload_bytes"], 8)
        self.assertEqual(result["model_checks"]["shards"][0]["dtypes"], ["BF16"])
        self.assertFalse(result["weights_hashed"])
        self.assertFalse(any(call.args and call.args[0] == "torch" for call in importer.call_args_list))

    def test_asr_passes_only_when_its_architecture_is_explicit(self):
        architecture = "Qwen3ASRForConditionalGeneration"
        (self.model / "config.json").write_text(
            json.dumps({"architectures": [architecture], "model_type": "qwen3_asr", "torch_dtype": "bfloat16"}),
            encoding="utf-8",
        )
        default_result = self.collect()
        self.assertFalse(default_result["model_checks"]["expected_architecture"])
        self.assertFalse(default_result["model_checks"]["complete"])
        result = preflight.collect(self.model, expected_architecture=architecture)
        self.assertEqual(result["model_checks"]["expected_architecture_name"], architecture)
        self.assertTrue(result["model_checks"]["expected_architecture"])
        self.assertTrue(result["model_checks"]["complete"])
        self.assertEqual(result["model_checks"]["model_type"], "qwen3_asr")
        self.assertEqual(result["model_checks"]["payload_bytes"], 8)
        self.assertTrue(result["model_checks"]["index_total_size_matches"])

    def test_expected_architecture_rejects_expressions_paths_and_nonidentifiers(self):
        for value in (None, 1, "", "module.Model", "../Model", "Model()", "__import__('torch')", "Model Class", "类名"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ASCII class identifier"):
                    preflight.collect(self.model, expected_architecture=value)

    def test_cli_forwards_the_selected_architecture_and_keeps_vl_default(self):
        architecture = "Qwen3ASRForConditionalGeneration"
        (self.model / "config.json").write_text(json.dumps({"architectures": [architecture]}), encoding="utf-8")
        for explicit, expected_code in ((True, 0), (False, 2)):
            with self.subTest(explicit=explicit):
                output = self.model / f"preflight-{explicit}.json"
                arguments = [str(SCRIPT), "--model", str(self.model), "--output", str(output)]
                if explicit:
                    arguments.extend(["--expected-architecture", architecture])
                with patch.object(preflight.sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(preflight.main(), expected_code)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(result["model_checks"]["expected_architecture"], explicit)
                self.assertEqual(result["model_checks"]["complete"], explicit)

    def test_truncated_prefix_header_and_payload_fail(self):
        path = self.model / "model-1.safetensors"
        for invalid in (b"\0" * 7, struct.pack("<Q", 1024) + b"{}"):
            with self.subTest(invalid_length=len(invalid)):
                path.write_bytes(invalid)
                self.assert_incomplete()
        self.write_shard(path.name, {"layer.a": tensor([0, 4])}, b"\0" * 2)
        self.assert_incomplete()

    def test_bad_offsets_and_shape_bytes_fail(self):
        for offsets in ([1, 5], [-4, 0], [4, 0], [0, 5], [0, 4.0], [False, 4], [0, 4, 8]):
            with self.subTest(offsets=offsets):
                self.write_shard("model-1.safetensors", {"layer.a": tensor(offsets)}, b"\0" * 4)
                self.assert_incomplete()
        for shape in ([3], [-1], [True]):
            with self.subTest(shape=shape):
                self.write_shard("model-1.safetensors", {"layer.a": tensor([0, 4], shape=shape)}, b"\0" * 4)
                self.assert_incomplete()

    def test_offset_gap_and_overlap_fail(self):
        self.index["weight_map"]["layer.b"] = "model-1.safetensors"
        self.write_index()
        (self.model / "model-2.safetensors").unlink()
        for offsets in ([5, 9], [2, 6]):
            with self.subTest(offsets=offsets):
                self.write_shard("model-1.safetensors", {"layer.a": tensor([0, 4]), "layer.b": tensor(offsets)}, b"\0" * 8)
                self.assert_incomplete()

    def test_index_keys_and_total_size_must_match(self):
        self.write_shard("model-1.safetensors", {"unindexed": tensor([0, 4])}, b"\0" * 4)
        result = self.assert_incomplete()
        self.assertFalse(result["model_checks"]["shards"][0]["index_keys_match"])
        self.write_shard("model-1.safetensors", {"layer.a": tensor([0, 4])}, b"\0" * 4)
        self.index["metadata"]["total_size"] = 999
        self.write_index()
        self.assertFalse(self.assert_incomplete()["model_checks"]["index_total_size_matches"])

    def test_index_path_escape_is_rejected_before_reading(self):
        for value in ("../outside.safetensors", str(self.model.parent / "outside.safetensors")):
            with self.subTest(value=value):
                self.index["weight_map"]["layer.a"] = value
                self.write_index()
                with self.assertRaisesRegex(ValueError, "outside model directory"):
                    self.collect()

    def test_symlink_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "external.safetensors"
            target.write_bytes(b"not a model")
            link = self.model / "external.safetensors"
            try:
                os.symlink(target, link)
            except OSError:
                self.skipTest("symlink creation unavailable")
            self.index["weight_map"]["layer.a"] = link.name
            self.write_index()
            with self.assertRaisesRegex(ValueError, "outside model directory"):
                self.collect()

    def test_nonobject_header_fails_closed(self):
        self.write_shard("model-1.safetensors", [], b"\0" * 4)
        self.assert_incomplete()

    def test_invalid_index_map_is_rejected(self):
        self.index["weight_map"] = ["not", "a", "map"]
        self.write_index()
        with self.assertRaisesRegex(ValueError, "weight_map"):
            self.collect()

    def test_tokenizer_and_architecture_are_required(self):
        (self.model / "tokenizer.json").unlink()
        self.assert_incomplete()
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        # A string containing the class name is not an architectures list.
        (self.model / "config.json").write_text(json.dumps({"architectures": "Qwen3VLForConditionalGeneration"}), encoding="utf-8")
        self.assert_incomplete()

    def use_single_file(self):
        (self.model / "model.safetensors.index.json").unlink()
        (self.model / "model-1.safetensors").unlink()
        (self.model / "model-2.safetensors").unlink()
        return self.write_shard("model.safetensors", {"layer.a": tensor([0, 4]), "layer.b": tensor([4, 8])}, b"\0" * 8)

    def test_explicit_single_file_passes_without_fabricating_index_claims(self):
        self.use_single_file()
        result = self.collect()
        checks = result["model_checks"]
        self.assertTrue(checks["complete"])
        self.assertEqual(checks["layout"], "single_file")
        self.assertFalse(checks["index_present"])
        self.assertIsNone(checks["index_weight_count"])
        self.assertIsNone(checks["index_total_size_matches"])
        self.assertIsNone(checks["shards"][0]["index_keys_match"])
        self.assertEqual(checks["tensor_count"], 2)
        self.assertEqual(checks["payload_bytes"], 8)
        self.assertTrue(checks["shards"][0]["payload_complete"])

    def test_single_file_truncation_and_malformed_header_fail(self):
        path = self.use_single_file()
        good = path.read_bytes()
        for bad in (good[:-1], b"\0" * 7, struct.pack("<Q", 1000) + b"{}"):
            with self.subTest(size=len(bad)):
                path.write_bytes(bad)
                self.assert_incomplete()
        self.write_shard(path.name, [], b"\0" * 8)
        self.assert_incomplete()
        self.write_shard(path.name, {"layer.a": tensor([0, 8])}, b"\0" * 8)
        self.assert_incomplete()  # BF16 shape [2] cannot claim eight payload bytes.

    def test_single_file_still_requires_tokenizer_and_explicit_asr_architecture(self):
        self.use_single_file()
        (self.model / "config.json").write_text(json.dumps({"architectures": ["Qwen3ASRForConditionalGeneration"]}), encoding="utf-8")
        self.assert_incomplete()
        selected = lambda: preflight.collect(self.model, expected_architecture="Qwen3ASRForConditionalGeneration")
        self.assertTrue(selected()["model_checks"]["complete"])
        (self.model / "tokenizer.json").unlink()
        self.assertFalse(selected()["model_checks"]["complete"])

    def test_single_file_rejects_ambiguous_extra_weights_and_unindexed_shards(self):
        self.use_single_file()
        for name in ("adapter_model.safetensors", "pytorch_model.bin", "other.pt"):
            with self.subTest(name=name):
                extra = self.model / name
                extra.write_bytes(b"extra weights")
                result = self.assert_incomplete()
                self.assertEqual(result["model_checks"]["layout_error"], "additional_weight_files_are_ambiguous")
                extra.unlink()
        (self.model / "model.safetensors").rename(self.model / "model-00001-of-00002.safetensors")
        self.assert_incomplete()  # Do not guess a missing index for shard-looking files.

    def test_indexed_layout_never_falls_back_to_an_extra_single_file(self):
        self.write_shard("model.safetensors", {"layer.a": tensor([0, 4])}, b"\0" * 4)
        result = self.assert_incomplete()
        self.assertEqual(result["model_checks"]["layout"], "indexed")
        self.assertIn("model.safetensors", result["model_checks"]["additional_weight_files"])

    def use_legacy_bpe(self):
        (self.model / "tokenizer.json").unlink()
        (self.model / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "ab": 2}), encoding="utf-8")
        (self.model / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")

    def test_legacy_bpe_pair_is_valid_without_fabricating_tokenizer_json(self):
        self.use_legacy_bpe()
        result = self.collect()
        checks = result["model_checks"]
        self.assertTrue(checks["complete"])
        self.assertEqual(checks["tokenizer_layout"], "legacy_bpe")
        self.assertFalse(checks["tokenizer_json_present"])
        self.assertTrue(checks["legacy_bpe_pair_present"])
        self.assertEqual(checks["tokenizer_merge_pairs"], 1)
        self.assertEqual(checks["tokenizer_vocab_entries"], 3)
        self.assertFalse((self.model / "tokenizer.json").exists())
        self.assertTrue({"vocab.json", "merges.txt"}.issubset({row["name"] for row in result["files"]}))

    def test_legacy_bpe_missing_half_or_config_fails(self):
        self.use_legacy_bpe()
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
            with self.subTest(name=name):
                path = self.model / name
                raw = path.read_bytes(); path.unlink()
                self.assertFalse(self.assert_incomplete()["model_checks"]["tokenizer_present"])
                path.write_bytes(raw)

    def test_legacy_bpe_corrupt_vocabulary_or_merges_fails(self):
        self.use_legacy_bpe()
        vocab = self.model / "vocab.json"
        for raw in ('[]', '{"a":true}', '{"a":0,"b":0}', '{broken'):
            with self.subTest(vocab=raw):
                vocab.write_text(raw, encoding="utf-8")
                self.assertFalse(self.assert_incomplete()["model_checks"]["tokenizer_present"])
        vocab.write_text(json.dumps({"a": 0, "b": 1, "ab": 2}), encoding="utf-8")
        merges = self.model / "merges.txt"
        for raw in (b"", b"#version: 0.2\n", b"a\n", b"a b c\n", b"a missing\n", b"a b\na b\n", b"\xff\xfe"):
            with self.subTest(merges=raw):
                merges.write_bytes(raw)
                self.assertFalse(self.assert_incomplete()["model_checks"]["tokenizer_present"])

    def test_existing_tokenizer_json_layout_is_preserved_and_not_bypassed(self):
        checks = self.collect()["model_checks"]
        self.assertTrue(checks["tokenizer_present"])
        self.assertEqual(checks["tokenizer_layout"], "tokenizer_json")
        (self.model / "vocab.json").write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
        (self.model / "merges.txt").write_text("a b\n", encoding="utf-8")
        (self.model / "tokenizer.json").write_text("[]", encoding="utf-8")
        self.assertFalse(self.assert_incomplete()["model_checks"]["tokenizer_present"])


if __name__ == "__main__":
    unittest.main()
