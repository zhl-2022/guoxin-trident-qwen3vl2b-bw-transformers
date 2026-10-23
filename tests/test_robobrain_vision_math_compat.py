"""Real CPU numerical tests for the candidate, independent of vLLM installation."""

import ast
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    path = ROOT / "scripts" / "robobrain_vision_math_compat.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    assert isinstance(tree.body[-1], ast.Expr) and tree.body[-1].value.func.id == "_install"
    tree.body.pop()
    namespace = {"__name__": "compat_cpu_fixture"}
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["_emit"] = lambda *args, **kwargs: None
    return namespace


def full_math(q, k, v, scale=None, enable_gqa=False):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                             v.transpose(1, 2), scale=scale,
                                             enable_gqa=enable_gqa).transpose(1, 2)


class VisionMathCompatTests(unittest.TestCase):
    def test_head_and_batch_broadcast_math_parity_obeys_score_budget(self):
        torch.manual_seed(20261020)
        for dtype in (torch.float32, torch.bfloat16):
            for head_counts in ((1, 4, 4), (1, 1, 4), (4, 1, 1)):
                with self.subTest(dtype=dtype, heads=head_counts):
                    ns = load_helpers()
                    score_limit = 2 * 4 * 67 * 7
                    wrapped = ns["_make_wrapper"](full_math, torch, SDPBackend, sdpa_kernel,
                                                   max_score_elements=score_limit)
                    q = torch.randn(1, 37, head_counts[0], 72, dtype=dtype)
                    k = torch.randn(2, 67, head_counts[1], 72, dtype=dtype)
                    v = torch.randn(1, 67, head_counts[2], 40, dtype=dtype)
                    expected = full_math(q, k, v, scale=0.125)
                    original_sdpa = F.scaled_dot_product_attention
                    calls = []

                    def counted_sdpa(query, key, value, **kwargs):
                        batch = max(t.size(0) for t in (query, key, value))
                        heads = max(t.size(1) for t in (query, key, value))
                        calls.append(batch * heads * query.size(2) * key.size(2))
                        return original_sdpa(query, key, value, **kwargs)

                    with patch.object(F, "scaled_dot_product_attention", counted_sdpa):
                        actual = wrapped(q, k, v, scale=0.125)
                    tolerance = 0.01 if dtype == torch.bfloat16 else 2e-5
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    self.assertEqual(list(actual.shape), [2, 37, 4, 40])
                    self.assertEqual(len(calls), 6)
                    self.assertTrue(all(elements <= score_limit for elements in calls))
                    self.assertEqual(ns["COUNTS"]["largest_score_elements"], score_limit)

    def test_broadcast_single_row_limit_and_invalid_geometry_fail_before_sdpa(self):
        ns = load_helpers()
        wrapped = ns["_make_wrapper"](full_math, torch, SDPBackend, sdpa_kernel,
                                       max_score_elements=4 * 67 - 1)
        q = torch.randn(1, 3, 1, 72)
        k, v = (torch.randn(1, 67, 4, 72) for _ in range(2))
        with patch.object(F, "scaled_dot_product_attention",
                          side_effect=AssertionError("invalid geometry reached SDPA")):
            with self.assertRaisesRegex(RuntimeError, "One query row"):
                wrapped(q, k, v)
            with self.assertRaisesRegex(RuntimeError, "GQA"):
                wrapped(q, k, v, enable_gqa=True)
            with self.assertRaisesRegex(RuntimeError, "broadcastable attention heads"):
                wrapped(torch.randn(1, 3, 2, 72),
                        torch.randn(1, 67, 3, 72), torch.randn(1, 67, 3, 72))
            with self.assertRaisesRegex(RuntimeError, "broadcastable batch"):
                wrapped(torch.randn(2, 3, 1, 72),
                        torch.randn(3, 67, 1, 72), torch.randn(1, 67, 1, 72))
            with self.assertRaisesRegex(RuntimeError, "equal Q/K"):
                wrapped(q, torch.randn(1, 67, 1, 64), torch.randn(1, 67, 1, 72))
            with self.assertRaisesRegex(RuntimeError, "K/V lengths"):
                wrapped(q, torch.randn(1, 67, 1, 72), torch.randn(1, 66, 1, 72))

    def test_numerical_parity_for_nondivisible_cross_batch_gqa_and_budget(self):
        torch.manual_seed(20261019)
        ns = load_helpers()
        for dtype in (torch.float32, torch.bfloat16):
            wrapped = ns["_make_wrapper"](full_math, torch, SDPBackend, sdpa_kernel,
                                           max_score_elements=2 * 4 * 199 * 13)
            q = torch.randn(2, 141, 4, 72, dtype=dtype)
            k = torch.randn(2, 199, 2, 72, dtype=dtype)
            v = torch.randn(2, 199, 2, 40, dtype=dtype)
            tolerance = 0.01 if dtype == torch.bfloat16 else 2e-5
            for scale in (None, 0.125):
                torch.testing.assert_close(wrapped(q, k, v, scale, True),
                                           full_math(q, k, v, scale, True),
                                           rtol=tolerance, atol=tolerance)
            self.assertLessEqual(ns["COUNTS"]["largest_score_elements"], 2 * 4 * 199 * 13)
        self.assertEqual(ns["COUNTS"]["completed"], 4)
        self.assertGreater(ns["COUNTS"]["chunks"], 4)

    def test_captured_per_image_wrapper_retains_boundaries(self):
        ns = load_helpers()
        wrapped = ns["_make_wrapper"](full_math, torch, SDPBackend, sdpa_kernel)
        capture = json.loads((ROOT / "evidence" / "robobrain-runtime-source-002.json").read_text(encoding="utf-8"))
        source = capture["sources"]["vllm/v1/attention/ops/vit_attn_wrappers.py"]["content"]
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "torch_sdpa_wrapper")
        scope = {"torch": torch, "apply_sdpa": wrapped,
                 "current_platform": types.SimpleNamespace(is_rocm=lambda: True)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "captured_vit", "exec"), scope)
        q, k, v = (torch.randn(1, 386, 2, 72) for _ in range(3))
        cu = torch.tensor([0, 129, 386], dtype=torch.int32)
        actual = scope["torch_sdpa_wrapper"](q, k, v, cu_seqlens=cu)
        expected = torch.cat([full_math(q[:, :129], k[:, :129], v[:, :129]),
                              full_math(q[:, 129:], k[:, 129:], v[:, 129:])], dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        v[:, 129:] += 7
        changed = scope["torch_sdpa_wrapper"](q, k, v, cu_seqlens=cu)
        torch.testing.assert_close(changed[:, :129], actual[:, :129], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed[:, 129:], actual[:, 129:]))

    def test_non_target_and_empty_delegate_and_single_row_limit_fails(self):
        ns = load_helpers()
        calls, sentinel = [], object()

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        wrapped = ns["_make_wrapper"](original, torch, SDPBackend, sdpa_kernel,
                                       max_score_elements=100)
        tensor = torch.randn(1, 129, 4, 64)
        self.assertIs(wrapped(tensor, tensor, tensor, scale=0.2, enable_gqa=True), sentinel)
        self.assertIs(calls[0][0][0], tensor)
        self.assertEqual(calls[0][1], {"scale": 0.2, "enable_gqa": True})
        empty = torch.empty(1, 0, 4, 72)
        self.assertIs(wrapped(empty, empty, empty), sentinel)
        target = torch.randn(1, 129, 4, 72)
        with self.assertRaisesRegex(RuntimeError, "One query row"):
            wrapped(target, target, target)
        self.assertEqual(len(calls), 2)

    def test_source_guard_rejects_altered_installed_source(self):
        ns = load_helpers()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.py"
            path.write_bytes(b"# original\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            module = types.SimpleNamespace(__name__="captured")
            with patch.object(ns["inspect"], "getsourcefile", return_value=str(path)):
                self.assertEqual(ns["_source_guard"](module, digest), digest)
                path.write_bytes(b"# changed\n")
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    ns["_source_guard"](module, digest)


if __name__ == "__main__":
    unittest.main()
