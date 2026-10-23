"""Opt-in vLLM plugin for the captured TF 5.5 ASR config init-order mismatch.

The native config constructor, checkpoint, tokenizer and model are unchanged.
Only pre-initialize its nested config before TF 5.5 validates token IDs.
Load through a task-local vllm.general_plugins entry point, never sitecustomize.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

EXPECTED_CONFIG_SHA256 = "56fd1f1ea939dff47533b052962d294375e85482ed310e48ccad402cf5699818"


def register():
    import vllm.transformers_utils.configs.qwen3_asr as module
    if importlib.metadata.version("transformers") != "5.5.0":
        raise RuntimeError("ASR init-order candidate requires captured Transformers 5.5.0")
    path = Path(inspect.getsourcefile(module))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("ASR config source hash differs; re-review compatibility")
    cls = module.Qwen3ASRConfig
    original = cls.__init__
    if getattr(original, "_qwen_asr_tf55_compat", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        thinker = bound.arguments.get("thinker_config")
        if thinker is not None and not isinstance(thinker, dict):
            raise TypeError("Captured ASR constructor expects thinker_config dict or None")
        # Preserve all native defaults and values. Native __init__ reconstructs
        # the same child after super(); preinitialization only satisfies its
        # get_text_config call during the new parent validation.
        self.thinker_config = module.Qwen3ASRThinkerConfig(**(thinker or {}))
        original(self, *args, **kwargs)

    initialize._qwen_asr_tf55_compat = True
    cls.__init__ = initialize
    print("QWEN_ASR_TF55_COMPAT " + json.dumps({
        "event": "installed", "config_sha256": actual,
        "scope": "nested config available during parent token-ID validation",
        "weights_changed": False, "official_accuracy_verified": False,
    }), flush=True)
