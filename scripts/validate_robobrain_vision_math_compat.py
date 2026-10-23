"""GPU numerical checks for the guarded candidate; no model load or installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2049)
    args = parser.parse_args()
    if args.output.exists() or not 257 <= args.max_length <= 8192:
        parser.error("output must be new and max-length must be in [257, 8192]")
    report = {"schema_version": 1, "kind": "head72_chunked_math_gpu_numerical",
              "started_at_unix": time.time(), "gpu_numerical_verified": False,
              "heldout_semantics_verified": False, "official_accuracy_verified": False,
              "official_performance_verified": False, "cases": []}
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if not torch.cuda.is_available():
            raise RuntimeError("This validation requires the actual HCU GPU image")
        import robobrain_vision_math_compat as compat
        from vllm.v1.attention.ops import vit_attn_wrappers as vit
        report.update(torch_version=torch.__version__, hip=torch.version.hip,
                      device=torch.cuda.get_device_name(0),
                      candidate_sha256=hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest())
        original = compat._ORIGINAL_APPLY_SDPA
        torch.manual_seed(20261019)

        def compare(name, expected, actual, dtype):
            difference = (expected.float() - actual.float()).abs()
            tolerance = 0.01 if dtype == torch.bfloat16 else 2e-5
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            item = {"name": name, "dtype": str(dtype), "shape": list(actual.shape),
                    "all_finite": finite, "max_abs_error": difference.max().item(),
                    "mean_abs_error": difference.mean().item(),
                    "rtol": tolerance, "atol": tolerance, "passed": False}
            report["cases"].append(item)
            if not finite:
                raise AssertionError(f"{name}: non-finite output")
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            item["passed"] = True
            return item

        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                for length in (1, 129, 257, args.max_length):
                    packed = torch.randn(1, length, 3, 16, 72, device="cuda", dtype=dtype)
                    q, k, v = packed.unbind(2)  # Exercise the native QKV view strides.
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = original(q, k, v, scale=None, enable_gqa=False)
                    compare(f"self_{length}", expected, vit.apply_sdpa(q, k, v), dtype)

                q = torch.randn(2, 141, 4, 72, device="cuda", dtype=dtype)
                k = torch.randn(2, 199, 2, 72, device="cuda", dtype=dtype)
                v = torch.randn(2, 199, 2, 40, device="cuda", dtype=dtype)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = original(q, k, v, scale=0.125, enable_gqa=True)
                compare("gqa_batch2_cross_length_value40", expected,
                        vit.apply_sdpa(q, k, v, scale=0.125, enable_gqa=True), dtype)

                # The updated candidate budgets broadcast output heads, not Hq
                # alone. Record each actual SDPA chunk to catch undercounting.
                for qh, kh, vh in ((1, 4, 4), (1, 1, 4), (4, 1, 1)):
                    q = torch.randn(1, 37, qh, 72, device="cuda", dtype=dtype)
                    k = torch.randn(2, 67, kh, 72, device="cuda", dtype=dtype)
                    v = torch.randn(1, 67, vh, 40, device="cuda", dtype=dtype)
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = original(q, k, v, scale=0.125)
                    score_limit = 2 * 4 * 67 * 7
                    tight_broadcast = compat._make_wrapper(
                        original, torch, SDPBackend, sdpa_kernel,
                        max_score_elements=score_limit)
                    original_sdpa = torch.nn.functional.scaled_dot_product_attention
                    observed_chunks = []

                    def budgeted_sdpa(query, key, value, **kwargs):
                        batch = max(t.size(0) for t in (query, key, value))
                        heads = max(t.size(1) for t in (query, key, value))
                        elements = batch * heads * query.size(2) * key.size(2)
                        if elements > score_limit:
                            raise AssertionError(f"Broadcast score budget exceeded: "
                                                 f"{elements} > {score_limit}")
                        observed_chunks.append({"batch": batch, "heads": heads,
                                                "query_length": query.size(2),
                                                "key_length": key.size(2),
                                                "score_elements": elements})
                        return original_sdpa(query, key, value, **kwargs)

                    with patch.object(torch.nn.functional,
                                      "scaled_dot_product_attention", budgeted_sdpa):
                        actual = tight_broadcast(q, k, v, scale=0.125)
                    if len(observed_chunks) != 6:
                        raise AssertionError("Expected five chunks of7 and final chunk of2")
                    item = compare(f"broadcast_heads{qh}_{kh}_{vh}_batch1_2_1_chunk7",
                                   expected, actual, dtype)
                    item.update(score_element_limit=score_limit,
                                observed_chunks=observed_chunks,
                                budget_verified=True)

                q, k, v = (torch.randn(1, 257, 2, 72, device="cuda", dtype=dtype)
                           for _ in range(3))
                tight = compat._make_wrapper(original, torch, SDPBackend, sdpa_kernel,
                                             max_score_elements=2 * 257 * 13)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = original(q, k, v, scale=0.2)
                compare("score_budget_chunk13", expected, tight(q, k, v, scale=0.2), dtype)

                # Test the installed custom-op path, preserving unequal image boundaries.
                q, k, v = (torch.randn(1, 386, 16, 72, device="cuda", dtype=dtype)
                           for _ in range(3))
                cu = torch.tensor([0, 129, 386], device="cuda", dtype=torch.int32)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = torch.cat([original(q[:, :129], k[:, :129], v[:, :129]),
                                          original(q[:, 129:], k[:, 129:], v[:, 129:])], dim=1)
                actual = vit.vit_torch_sdpa_wrapper(q, k, v, cu_seqlens=cu)
                compare("registered_op_two_unequal_images", expected, actual, dtype)
                changed_v = v.clone()
                changed_v[:, 129:] += 7
                changed = vit.vit_torch_sdpa_wrapper(q, k, changed_v, cu_seqlens=cu)
                compare("second_image_isolation", actual[:, :129], changed[:, :129], dtype)

                q, k, v = (torch.randn(1, 129, 4, 64, device="cuda", dtype=dtype)
                           for _ in range(3))
                with sdpa_kernel(SDPBackend.MATH):
                    compare("non_target_head64", original(q, k, v), vit.apply_sdpa(q, k, v), dtype)
            torch.cuda.synchronize()
        report.update(gpu_numerical_verified=True, candidate_counts=dict(compat.COUNTS),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        result = 0
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        result = 2
    report["finished_at_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "gpu_numerical_verified":
                      report["gpu_numerical_verified"], "case_count": len(report["cases"]),
                      "error": report.get("error")}), flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
