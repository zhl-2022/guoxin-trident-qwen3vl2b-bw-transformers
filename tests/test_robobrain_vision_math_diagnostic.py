"""CPU-only behavior checks; no vLLM/PyTorch installation or model load."""

import ast
import hashlib
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robobrain_vision_math_diagnostic.py"


def load_helpers():
    # Execute the real helper definitions while intentionally omitting only the
    # final import-time installation, which needs the captured GPU image.
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), str(MODULE_PATH))
    assert isinstance(tree.body[-1], ast.Expr)
    assert isinstance(tree.body[-1].value, ast.Call)
    assert tree.body[-1].value.func.id == "_install"
    tree.body.pop()
    namespace = {"__name__": "diagnostic_cpu_fixture"}
    exec(compile(tree, str(MODULE_PATH), "exec"), namespace)
    return namespace


class FakeTensor:
    def __init__(self, length, dimensions=4):
        self.shape = (1, length, 16, 72) if dimensions == 4 else (1, length, 1152)
        self.dtype, self.device = "bfloat16", "cuda:0"

    def dim(self):
        return len(self.shape)

    def size(self, index):
        return self.shape[index]

    def stride(self):
        return (self.shape[1] * 1152, 1152, 72, 1) if self.dim() == 4 else (self.shape[1] * 1152, 1152, 1)

    def is_contiguous(self):
        return True


class VisionMathDiagnosticTests(unittest.TestCase):
    def test_math_only_for_fixture_shape_and_restores_context(self):
        ns = load_helpers()
        state = {"math_only": False, "enters": 0}
        calls, events = [], []
        ns["_emit"] = lambda event, **fields: events.append((event, fields))

        @contextmanager
        def sdpa_kernel(backend):
            self.assertEqual(backend, "MATH")
            self.assertFalse(state["math_only"])
            state.update(math_only=True, enters=state["enters"] + 1)
            try:
                yield
            finally:
                state["math_only"] = False

        backend = types.SimpleNamespace(
            math_sdp_enabled=lambda: state["math_only"],
            flash_sdp_enabled=lambda: not state["math_only"],
            mem_efficient_sdp_enabled=lambda: not state["math_only"],
            cudnn_sdp_enabled=lambda: not state["math_only"],
        )
        sentinel, cu_seqlens = object(), object()

        def original(layer, query, key, value, cu):
            calls.append((layer.head_size, query.size(1), state["math_only"], cu))
            self.assertIs(query, key)
            self.assertIs(key, value)
            return sentinel

        wrapped = ns["_make_wrapper"](
            original, types.SimpleNamespace(backends=types.SimpleNamespace(cuda=backend)),
            types.SimpleNamespace(MATH="MATH"), sdpa_kernel,
        )
        for head_size, length, dimensions, expected_math in (
            (72, 384, 4, True), (72, 768, 3, True),
            (72, 32768, 4, False), (64, 384, 4, False),
        ):
            tensor = FakeTensor(length, dimensions)
            self.assertIs(wrapped(types.SimpleNamespace(head_size=head_size), tensor,
                                  tensor, tensor, cu_seqlens), sentinel)
            self.assertEqual(calls[-1][2], expected_math)
            self.assertIs(calls[-1][3], cu_seqlens)
            self.assertFalse(state["math_only"])
        self.assertEqual(state["enters"], 2)
        self.assertEqual(ns["COUNTS"], {"seen": 4, "eligible": 2, "completed": 2, "unchanged": 2})
        self.assertEqual([event for event, _ in events], ["eligible_enter", "eligible_complete"])

    def test_source_guard_accepts_exact_source_and_rejects_changed_file(self):
        ns = load_helpers()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "captured.py"
            source.write_bytes(b"# captured source\n")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            module = types.SimpleNamespace(__name__="captured")
            with patch.object(ns["inspect"], "getsourcefile", return_value=str(source)):
                self.assertEqual(ns["_source_guard"](module, expected), expected)
                source.write_bytes(b"# changed source\n")
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    ns["_source_guard"](module, expected)


if __name__ == "__main__":
    unittest.main()
