import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from onnx import TensorProto
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession
from onnx_array_api.ext_test_case import ExtTestCase, skipif_ci_windows
from onnx_array_api.npx import eager_onnx, jit_onnx
from onnx_array_api.npx.npx_functions import absolute as absolute_inline
from onnx_array_api.npx.npx_functions import cdist as cdist_inline
from onnx_array_api.npx.npx_functions_test import absolute
from onnx_array_api.npx.npx_functions import copy as copy_inline
from onnx_array_api.npx.npx_types import Float32, Float64, DType
from onnx_array_api.npx.npx_var import Input
from onnx_array_api.ort.ort_tensors import EagerOrtTensor, JitOrtTensor, OrtTensor

DEFAULT_OPSET = onnx_opset_version()


class TestOrtTensor(ExtTestCase):
    @skipif_ci_windows("Unstable on Windows")
    def test_eager_numpy_type_ort(self):
        def impl(A):
            self.assertIsInstance(A, EagerOrtTensor)
            b = absolute(A)
            self.assertIsInstance(b, EagerOrtTensor)
            c = absolute_inline(A)
            self.assertIsInstance(c, EagerOrtTensor)
            return c

        e = eager_onnx(impl, EagerOrtTensor, target_opsets={"": 17}, ir_version=8)
        self.assertEqual(len(e.versions), 0)

        # Float64
        x = np.array([0, 1, -2], dtype=np.float64)
        z = np.abs(x)
        xort = OrtTensor.from_array(x)
        res = e(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)

        # again
        res = e(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)

    @skipif_ci_windows("Unstable on Windows")
    def test_eager_numpy_type_ort_op(self):
        def impl(A):
            self.assertIsInstance(A, EagerOrtTensor)
            b = absolute(A) + A
            self.assertIsInstance(b, EagerOrtTensor)
            return b

        e = eager_onnx(impl, EagerOrtTensor, target_opsets={"": 17}, ir_version=8)
        self.assertEqual(len(e.versions), 0)

        # Float64
        x = np.array([0, 1, -2], dtype=np.float64)
        z = np.abs(x) + x
        xort = OrtTensor.from_array(x)
        res = e(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)

        # again
        res = e(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)

    @skipif_ci_windows("Unstable on Windows")
    def test_eager_ort(self):
        def impl(A):
            print("A")
            b = absolute(A)
            print("B")
            c = b - A
            print("C")
            return c

        with redirect_stdout(StringIO()):
            f = impl(Input("A"))
            onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x) - x
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl, EagerOrtTensor, target_opsets={"": 17}, ir_version=8)

        # Float64
        xort = OrtTensor.from_array(x)
        with redirect_stdout(StringIO()):
            res = f(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)

        # Int64
        ix = x.astype(np.int64)
        xiort = OrtTensor.from_array(ix)
        with redirect_stdout(StringIO()):
            res = f(xiort)
        self.assertEqualArray(z.astype(np.int64), res.numpy())
        self.assertEqual(res.numpy().dtype, np.int64)

        # eager

        e = eager_onnx(impl, EagerOrtTensor, target_opsets={"": 17}, ir_version=8)

        # Float64
        s = StringIO()
        with redirect_stdout(s):
            res = e(xort)
        text = s.getvalue()
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)
        self.assertEqual(tuple(res.shape()), z.shape)
        self.assertStartsWith("A\nB\nC\n", text)

        # Int64
        s = StringIO()
        with redirect_stdout(s):
            res = e(xiort)
        text = s.getvalue()
        self.assertEqual(res.numpy().dtype, np.int64)
        self.assertEqual("A\nB\nC\n", text)
        self.assertEqualArray(z.astype(np.int64), res.numpy())
        self.assertEqual(ix.shape, tuple(res.shape()))

        # eager 2D

        x = np.array([[-5, 6], [-1, 2]], dtype=np.float64)
        xort = OrtTensor.from_array(x)
        z = np.abs(x) - x
        s = StringIO()
        with redirect_stdout(s):
            res = e(xort)
        text = s.getvalue()
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)
        self.assertEqual(tuple(res.shape()), z.shape)
        self.assertStartsWith("A\nB\nC\n", text)

    @skipif_ci_windows("Unstable on Windows")
    def test_cdist_com_microsoft(self):
        from scipy.spatial.distance import cdist as scipy_cdist

        metric = "euclidean"

        def impl(xa, xb):
            return cdist_inline(xa, xb, metric=metric)

        target_opsets = {"": 18, "com.microsoft": 1}
        onx = impl(Input("A"), Input("B")).to_onnx(
            constraints={
                "A": Float32[None],
                "B": Float32[None],
                (0, False): Float32[None],
            },
            target_opsets=target_opsets,
        )
        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        y = (np.arange(14).reshape((7, 2)) * 10).astype(dtype=np.float32)
        z = scipy_cdist(x, y, metric=metric).astype(np.float32)
        ref = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0], atol=1e-4)

        f = jit_onnx(impl, JitOrtTensor, target_opsets=target_opsets)

        # float32
        xort = OrtTensor.from_array(x)
        yort = OrtTensor.from_array(y)
        self.assertEqualArray(x, xort.numpy())
        self.assertEqualArray(y, yort.numpy())
        res = f(xort, yort)
        self.assertEqual(res.numpy().dtype, np.float32)
        self.assertEqualArray(z, res.numpy(), atol=1e-4)

        # float64
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        xort = OrtTensor.from_array(x)
        yort = OrtTensor.from_array(y)
        self.assertEqualArray(x.astype(np.float64), xort.numpy())
        self.assertEqualArray(y.astype(np.float64), yort.numpy())
        res = f(xort, yort)
        self.assertEqual(res.numpy().dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res.numpy(), atol=1e-5)

        pieces = str(onx).split('s: "euclidean"')
        if len(pieces) > 2:
            raise AssertionError(f"Function is not using argument:\n{onx}")

    def test_astype_w2(self):
        f = absolute_inline(copy_inline(Input("A")).astype(DType(TensorProto.FLOAT)))
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.astype(np.float32))
        ref = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_astype0_w2(self):
        f = absolute_inline(copy_inline(Input("A")).astype(DType(TensorProto.FLOAT)))
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array(-5, dtype=np.float64)
        z = np.abs(x.astype(np.float32))
        ref = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    @skipif_ci_windows("Unstable on Windows")
    def test_eager_ort_cast(self):
        def impl(A):
            return A.astype(DType("FLOAT"))

        e = eager_onnx(impl)
        self.assertEqual(len(e.versions), 0)

        # Float64
        x = np.array([0, 1, -2], dtype=np.float64)
        z = x.astype(np.float32)
        res = e(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float32)

        # again
        x = np.array(1, dtype=np.float64)
        z = x.astype(np.float32)
        res = e(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
