"""Opt-in head-72 vision SDPA compatibility candidate for the captured DAS image.

Load via --worker-extension-cls robobrain_vision_math_compat.VisionMathCompat.
Query chunks retain every key/value of the current image. The captured vLLM
torch_sdpa_wrapper still owns cu_seqlens splitting and contiguous conversion.
This implements the same noncausal attention equation, not bitwise equivalence.
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
MAX_QUERY_CHUNK = 128
# At most 64 MiB for one FP32 score tensor, not a total allocator/peak-memory cap.
MAX_SCORE_ELEMENTS = 16_777_216
COUNTS = {"seen": 0, "eligible": 0, "completed": 0, "unchanged": 0,
          "chunks": 0, "largest_score_elements": 0}
_ORIGINAL_APPLY_SDPA = None


def _emit(event: str, **fields: object) -> None:
    print("ROBOBRAIN_VISION_MATH_COMPAT " + json.dumps(
        {"event": event, "pid": os.getpid(), **fields},
        ensure_ascii=True, sort_keys=True, default=str), flush=True)


def _source_guard(module: object, expected_sha256: str) -> str:
    filename = inspect.getsourcefile(module)
    if not filename:
        raise RuntimeError("Compatibility candidate requires captured Python sources")
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"Compatibility source mismatch for {module.__name__}: "
                           f"expected {expected_sha256}, got {actual}")
    return actual


def _attention_geometry(q, k, v, enable_gqa):
    """Return conservative broadcast batch/head sizes before budgeting scores."""
    if q.size(-1) != k.size(-1) or k.size(1) != v.size(1):
        raise RuntimeError("Compatibility candidate requires equal Q/K features and K/V lengths")
    batches = [t.size(0) for t in (q, k, v)]
    batch = max(batches)
    if any(size not in (1, batch) for size in batches):
        raise RuntimeError("Compatibility candidate requires broadcastable batch dimensions")
    heads = [t.size(2) for t in (q, k, v)]
    if enable_gqa:
        query_heads, key_heads, value_heads = heads
        if key_heads != value_heads or query_heads % key_heads:
            raise RuntimeError("Compatibility candidate requires Hq divisible by equal GQA K/V heads")
        score_heads = query_heads
    else:
        score_heads = max(heads)
        if any(size not in (1, score_heads) for size in heads):
            raise RuntimeError("Compatibility candidate requires broadcastable attention heads")
    return batch, score_heads


def _make_wrapper(original, torch_module, sdp_backend, sdpa_kernel, *,
                  max_query_chunk=MAX_QUERY_CHUNK,
                  max_score_elements=MAX_SCORE_ELEMENTS):
    if max_query_chunk < 1 or max_score_elements < 1:
        raise ValueError("Chunk and score-element limits must be positive")

    @functools.wraps(original)
    def apply_sdpa(q, k, v, scale=None, enable_gqa=False):
        COUNTS["seen"] += 1
        # Retain the original contract and all non-target/empty-input behavior.
        if (any(t.dim() != 4 for t in (q, k, v)) or q.size(-1) != 72
                or any(t.numel() == 0 for t in (q, k, v))):
            COUNTS["unchanged"] += 1
            return original(q, k, v, scale=scale, enable_gqa=enable_gqa)

        COUNTS["eligible"] += 1
        q_length, k_length = q.size(1), k.size(1)
        # Hq=1 may broadcast to Hkv>1 even when GQA is disabled. Counting Hq
        # alone underestimates the score tensor and can violate the chunk cap.
        batch, score_heads = _attention_geometry(q, k, v, enable_gqa)
        elements_per_query = batch * score_heads * k_length
        if elements_per_query > max_score_elements:
            raise RuntimeError("One query row exceeds the configured score-element "
                               f"limit: {elements_per_query} > {max_score_elements}")
        chunk = min(max_query_chunk, max_score_elements // elements_per_query, q_length)
        score_elements = elements_per_query * chunk
        COUNTS["largest_score_elements"] = max(
            COUNTS["largest_score_elements"], score_elements)
        report_call = COUNTS["eligible"] == 1 or COUNTS["eligible"] % 27 == 0
        q_heads, k_heads, v_heads = (t.transpose(1, 2) for t in (q, k, v))
        outputs = []
        try:
            with sdpa_kernel(sdp_backend.MATH):
                if report_call:
                    backend = torch_module.backends.cuda
                    _emit("eligible_enter", counts=dict(COUNTS), query_shape=list(q.shape),
                          key_shape=list(k.shape), value_shape=list(v.shape),
                          dtype=str(q.dtype), query_chunk=chunk,
                          budget_batch=batch, budget_heads=score_heads,
                          score_elements_per_chunk=score_elements,
                          sdpa_flags={"math": backend.math_sdp_enabled(),
                                      "flash": backend.flash_sdp_enabled(),
                                      "efficient": backend.mem_efficient_sdp_enabled(),
                                      "cudnn": backend.cudnn_sdp_enabled()})
                for start in range(0, q_length, chunk):
                    outputs.append(torch_module.nn.functional.scaled_dot_product_attention(
                        q_heads[:, :, start:start + chunk, :], k_heads, v_heads,
                        dropout_p=0.0, scale=scale, enable_gqa=enable_gqa))
                    COUNTS["chunks"] += 1
            output = torch_module.cat(outputs, dim=2).transpose(1, 2)
            COUNTS["completed"] += 1
            if report_call:
                _emit("eligible_complete", counts=dict(COUNTS))
            return output
        except BaseException as exc:
            _emit("eligible_error", counts=dict(COUNTS), error_type=type(exc).__name__,
                  error=str(exc)[:800])
            raise

    apply_sdpa._robobrain_vision_math_compat = True
    return apply_sdpa


class VisionMathCompat:
    """Empty worker mixin; import installs the guarded compatibility candidate."""


def _install() -> None:
    global _ORIGINAL_APPLY_SDPA
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from vllm.model_executor.custom_op import get_oot_class_by_name
    from vllm.model_executor.layers.attention import mm_encoder_attention as mm
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import vit_attn_wrappers as vit

    verified = {module.__name__: _source_guard(module, EXPECTED_SOURCE_HASHES[module.__name__])
                for module in (mm, vit)}
    if not current_platform.is_rocm() or current_platform.is_out_of_tree():
        raise RuntimeError("Compatibility candidate requires captured ROCm-dispatch HCU")
    if get_oot_class_by_name("MMEncoderAttention") is not None:
        raise RuntimeError("Compatibility candidate refuses OOT MMEncoderAttention")
    if getattr(mm.MMEncoderAttention._forward_sdpa,
               "_robobrain_vision_math_diagnostic", False):
        raise RuntimeError("Remove the fixture diagnostic before enabling compatibility")
    if getattr(vit.apply_sdpa, "_robobrain_vision_math_compat", False):
        raise RuntimeError("Compatibility candidate already installed")
    _ORIGINAL_APPLY_SDPA = vit.apply_sdpa
    vit.apply_sdpa = _make_wrapper(vit.apply_sdpa, torch, SDPBackend, sdpa_kernel)
    _emit("installed", source_sha256=verified, head_size=72,
          max_query_chunk=MAX_QUERY_CHUNK, max_score_elements=MAX_SCORE_ELEMENTS,
          scope="all head72 vision apply_sdpa; original per-image cu boundaries retained",
          status="candidate; numerical GPU and heldout image validation required")
    atexit.register(lambda: _emit("process_exit", counts=dict(COUNTS)))


_install()
