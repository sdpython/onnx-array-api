import unittest
import numpy as np
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.validation.docs import make_euclidean, make_euclidean_skl2onnx


class TestDocs(ExtTestCase):
    def test_make_euclidean(self):
        model = make_euclidean()

        ref = ReferenceEvaluator(model)
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        self.assertEqualArray(expected, got)

    def test_make_euclidean_skl2onnx(self):
        model = make_euclidean_skl2onnx()

        ref = ReferenceEvaluator(model)
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        self.assertEqualArray(expected, got)

    def test_make_euclidean_np(self):
        from onnx_array_api.npx import jit_onnx

        def l2_loss(x, y):
            return ((x - y) ** 2).sum(keepdims=1)

        jitted_myloss = jit_onnx(l2_loss)
        dummy = np.array([0], dtype=np.float32)
        jitted_myloss(dummy, dummy)
        model = jitted_myloss.get_onnx()

        ref = ReferenceEvaluator(model)
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"x0": X, "x1": Y})[0]
        self.assertEqualArray(expected, got)

    def test_make_euclidean_light(self):
        from onnx_array_api.light_api import start

        model = (
            start()
            .vin("X")
            .vin("Y")
            .bring("X", "Y")
            .Sub()
            .rename("dxy")
            .cst(np.array([2], dtype=np.int64), "two")
            .bring("dxy", "two")
            .Pow()
            .ReduceSum()
            .rename("Z")
            .vout()
            .to_onnx()
        )

        ref = ReferenceEvaluator(model)
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        self.assertEqualArray(expected, got)

    def test_ort_make_euclidean(self):
        from onnxruntime import InferenceSession

        model = make_euclidean(opset=18)

        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
