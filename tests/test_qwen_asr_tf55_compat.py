"""Regression checks for nested config availability during parent validation."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("asr_compat_test", ROOT / "scripts/qwen_asr_tf55_compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class ASRConfigCompatibilityTests(unittest.TestCase):
    def fake_module(self):
        module = types.ModuleType("vllm.transformers_utils.configs.qwen3_asr")
        class Child:
            def __init__(self, **values):
                self.values = values
        class Config:
            def __init__(self, thinker_config=None, support_languages=None, **kwargs):
                # Models the actual new parent validation which accesses the
                # subclass child before the original constructor assigns it.
                self.parent_saw_child = dict(self.thinker_config.values)
                self.thinker_config = Child(**(thinker_config or {}))
                self.support_languages = support_languages
                self.other = kwargs
        module.Qwen3ASRConfig = Config
        module.Qwen3ASRThinkerConfig = Child
        return module

    def test_parent_sees_exact_child_and_original_arguments_are_preserved(self):
        module = self.fake_module()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "captured.py"
            source.write_bytes(b"captured")
            module.__file__ = str(source)
            import hashlib
            modules = {"vllm": types.ModuleType("vllm"),
                       "vllm.transformers_utils": types.ModuleType("vllm.transformers_utils"),
                       "vllm.transformers_utils.configs": types.ModuleType("vllm.transformers_utils.configs"),
                       module.__name__: module}
            with patch.dict(sys.modules, modules), patch.object(compat.importlib.metadata, "version", return_value="5.5.0"), patch.object(compat, "EXPECTED_CONFIG_SHA256", hashlib.sha256(b"captured").hexdigest()), patch.object(compat.inspect, "getsourcefile", return_value=str(source)):
                with self.assertRaises(AttributeError):
                    module.Qwen3ASRConfig(thinker_config={"text_config": {"vocab_size": 42}})
                compat.register()
                child = {"text_config": {"vocab_size": 42}, "audio_token_id": 12}
                config = module.Qwen3ASRConfig(child, ["Chinese"], eos_token_id=3)
                self.assertEqual(config.parent_saw_child, child)
                self.assertEqual(config.thinker_config.values, child)
                self.assertEqual(config.support_languages, ["Chinese"])
                self.assertEqual(config.other, {"eos_token_id": 3})
                first = module.Qwen3ASRConfig.__init__
                compat.register()
                self.assertIs(first, module.Qwen3ASRConfig.__init__)
                with self.assertRaises(TypeError):
                    module.Qwen3ASRConfig("bad")

    def test_rejects_other_transformers_version_before_patch(self):
        module = self.fake_module()
        modules = {"vllm": types.ModuleType("vllm"),
                   "vllm.transformers_utils": types.ModuleType("vllm.transformers_utils"),
                   "vllm.transformers_utils.configs": types.ModuleType("vllm.transformers_utils.configs"),
                   module.__name__: module}
        original = module.Qwen3ASRConfig.__init__
        with patch.dict(sys.modules, modules), patch.object(compat.importlib.metadata, "version", return_value="4.57.6"):
            with self.assertRaises(RuntimeError):
                compat.register()
        self.assertIs(original, module.Qwen3ASRConfig.__init__)
