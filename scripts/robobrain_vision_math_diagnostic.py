"""Opt-in, project-local vLLM worker diagnostic for the captured DAS image.

Load only through ``--worker-extension-cls
robobrain_vision_math_diagnostic.VisionMathDiagnostic`` with this directory on
PYTHONPATH. It changes SDPA selection only for the RoboBrain fixture shapes;
it is not a general model fix or an official accuracy/performance result.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path


EXPECTED_SOURCE_HASHES = {
    "vllm.model_executor.layers.attention.mm_encoder_attention":
        "ab688ca30965d042f51c010f58c9b46d5b9c29bd3b92137e56f00854541f14f9",
    "vllm.v1.attention.ops.vit_attn_wrappers":
        "99373a69178219e3d36c596da714176dad44f5820e638ebef1f810797f91c677",
}
FIXTURE_SEQUENCE_LENGTHS = frozenset({384, 768})
COUNTS = {"seen": 0, "eligible": 0, "completed": 0, "unchanged": 0}


def _emit(event: str, **fields: object) -> None:
    print(
        "ROBOBRAIN_VISION_MATH_DIAGNOSTIC "
        + json.dumps({"event": event, "pid": os.getpid(), **fields},
                     ensure_ascii=True, sort_keys=True, default=str),
        flush=True,
    )


def _source_guard(module: object, expected_sha256: str) -> str:
    source_file = inspect.getsourcefile(module)
    if not source_file:
        raise RuntimeError("Diagnostic requires a readable captured Python source")
    actual_sha256 = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Diagnostic source mismatch for {getattr(module, '__name__', module)}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def _tensor_metadata(tensor: object) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _make_wrapper(original, torch_module, sdp_backend, sdpa_kernel):
    @functools.wraps(original)
    def diagnostic(self, query, key, value, cu_seqlens=None):
        COUNTS["seen"] += 1
        eligible = (
            self.head_size == 72
            and query.dim() in (3, 4)
            and query.size(1) in FIXTURE_SEQUENCE_LENGTHS
        )
        if not eligible:
            COUNTS["unchanged"] += 1
            return original(self, query, key, value, cu_seqlens)

        COUNTS["eligible"] += 1
        report_call = COUNTS["eligible"] == 1 or COUNTS["eligible"] % 27 == 0
        try:
            with sdpa_kernel(sdp_backend.MATH):
                if report_call:
                    backend = torch_module.backends.cuda
                    _emit(
                        "eligible_enter", counts=dict(COUNTS),
                        head_size=self.head_size,
                        attention_backend=str(getattr(self, "attn_backend", None)),
                        query=_tensor_metadata(query), key=_tensor_metadata(key),
                        value=_tensor_metadata(value),
                        sdpa_flags={
                            "math": backend.math_sdp_enabled(),
                            "flash": backend.flash_sdp_enabled(),
                            "efficient": backend.mem_efficient_sdp_enabled(),
                            "cudnn": backend.cudnn_sdp_enabled(),
                        },
                    )
                output = original(self, query, key, value, cu_seqlens)
            COUNTS["completed"] += 1
            if report_call:
                _emit("eligible_complete", counts=dict(COUNTS))
            return output
        except BaseException as exc:
            _emit("eligible_error", counts=dict(COUNTS),
                  error_type=type(exc).__name__, error=str(exc)[:800])
            raise

    diagnostic._robobrain_vision_math_diagnostic = True
    return diagnostic


class VisionMathDiagnostic:
    """Empty worker mixin; module import installs the scoped diagnostic."""


def _install() -> None:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from vllm.model_executor.custom_op import get_oot_class_by_name
    from vllm.model_executor.layers.attention import mm_encoder_attention as mm
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import vit_attn_wrappers as vit

    verified = {
        module.__name__: _source_guard(module, EXPECTED_SOURCE_HASHES[module.__name__])
        for module in (mm, vit)
    }
    if not current_platform.is_rocm() or current_platform.is_out_of_tree():
        raise RuntimeError("Diagnostic requires the captured ROCm-dispatch HCU platform")
    if get_oot_class_by_name("MMEncoderAttention") is not None:
        raise RuntimeError("Diagnostic refuses an OOT MMEncoderAttention replacement")
    original = mm.MMEncoderAttention._forward_sdpa
    if getattr(original, "_robobrain_vision_math_diagnostic", False):
        raise RuntimeError("Diagnostic was already installed under another module name")
    mm.MMEncoderAttention._forward_sdpa = _make_wrapper(
        original, torch, SDPBackend, sdpa_kernel
    )
    _emit(
        "installed", source_sha256=verified,
        platform=f"{type(current_platform).__module__}.{type(current_platform).__name__}",
        head_size=72, sequence_lengths=sorted(FIXTURE_SEQUENCE_LENGTHS),
        scope="only matching MMEncoderAttention SDPA calls; other shapes unchanged",
    )
    atexit.register(lambda: _emit("process_exit", counts=dict(COUNTS)))


_install()
