import sys
import unittest
import numpy as np
from onnx import TensorProto
from onnx_array_api.ext_test_case import ExtTestCase, ignore_warnings
from onnx_array_api.array_api import onnx_numpy as xp
from onnx_array_api.npx.npx_types import DType
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor as EagerTensor
from onnx_array_api.npx.npx_functions import linspace as linspace_inline
from onnx_array_api.npx.npx_types import Float64, Int64
from onnx_array_api.npx.npx_var import Input
from onnx_array_api.reference import ExtendedReferenceEvaluator


class TestOnnxNumpy(ExtTestCase):
    def test_empty(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        self.assertRaise(lambda: xp.empty(c, dtype=xp.int64), RuntimeError)

    def test_zeros(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.zeros(c, dtype=xp.int64)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (4, 5))
        self.assertNotEmpty(matnp[0, 0])
        a = xp.absolute(mat)
        self.assertEqualArray(np.absolute(mat.numpy()), a.numpy())

    @ignore_warnings(DeprecationWarning)
    def test_arange_default(self):
        a = EagerTensor(np.array([0], dtype=np.int64))
        b = EagerTensor(np.array([2], dtype=np.int64))
        mat = xp.arange(a, b)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (2,))
        self.assertEqualArray(matnp, np.arange(0, 2).astype(np.int64))

    @ignore_warnings(DeprecationWarning)
    def test_arange_step(self):
        a = EagerTensor(np.array([4], dtype=np.int64))
        s = EagerTensor(np.array([2], dtype=np.int64))
        mat = xp.arange(a, step=s)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (2,))
        self.assertEqualArray(matnp, np.arange(4, step=2).astype(np.int64))

    def test_zeros_none(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.zeros(c)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (4, 5))
        self.assertNotEmpty(matnp[0, 0])
        self.assertEqualArray(matnp, np.zeros((4, 5)))

    def test_ones_none(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.ones(c)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (4, 5))
        self.assertNotEmpty(matnp[0, 0])
        self.assertEqualArray(matnp, np.ones((4, 5)))

    def test_ones_like(self):
        x = np.array([5, 6], dtype=np.int8)
        y = np.ones_like(x)
        a = EagerTensor(x)
        b = xp.ones_like(a)
        self.assertEqualArray(y, b.numpy())

    def test_full(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.full(c, fill_value=5, dtype=xp.int64)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (4, 5))
        self.assertNotEmpty(matnp[0, 0])
        a = xp.absolute(mat)
        self.assertEqualArray(np.absolute(mat.numpy()), a.numpy())

    def test_full_bool(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.full(c, fill_value=False)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (4, 5))
        self.assertNotEmpty(matnp[0, 0])
        self.assertEqualArray(matnp, np.full((4, 5), False))

    @ignore_warnings(DeprecationWarning)
    def test_arange_int00a(self):
        a = EagerTensor(np.array([0], dtype=np.int64))
        b = EagerTensor(np.array([0], dtype=np.int64))
        mat = xp.arange(a, b)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (0,))
        expected = np.arange(0, 0)
        if sys.platform == "win32":
            expected = expected.astype(np.int64)
        self.assertEqualArray(matnp, expected)

    @ignore_warnings(DeprecationWarning)
    def test_arange_int00(self):
        mat = xp.arange(0, 0)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (0,))
        expected = np.arange(0, 0)
        if sys.platform == "win32":
            expected = expected.astype(np.int64)
        self.assertEqualArray(matnp, expected)

    def test_ones_like_uint16(self):
        x = EagerTensor(np.array(0, dtype=np.uint16))
        y = np.ones_like(x.numpy())
        z = xp.ones_like(x)
        self.assertEqual(y.dtype, x.numpy().dtype)
        self.assertEqual(x.dtype, z.dtype)
        self.assertEqual(x.dtype, DType(TensorProto.UINT16))
        self.assertEqual(z.dtype, DType(TensorProto.UINT16))
        self.assertEqual(x.numpy().dtype, np.uint16)
        self.assertEqual(z.numpy().dtype, np.uint16)
        self.assertNotIn("bfloat16", str(z.numpy().dtype))
        expected = np.array(1, dtype=np.uint16)
        self.assertEqualArray(expected, z.numpy())

    def test_full_like(self):
        c = EagerTensor(np.array(False))
        expected = np.full_like(c.numpy(), fill_value=False)
        mat = xp.full_like(c, fill_value=False)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, tuple())
        self.assertEqualArray(expected, matnp)

    def test_full_like_mx(self):
        c = EagerTensor(np.array([], dtype=np.uint8))
        expected = np.full_like(c.numpy(), fill_value=0)
        mat = xp.full_like(c, fill_value=0)
        matnp = mat.numpy()
        self.assertEqualArray(expected, matnp)

    def test_ones_like_mx(self):
        c = EagerTensor(np.array([], dtype=np.uint8))
        expected = np.ones_like(c.numpy())
        mat = xp.ones_like(c)
        matnp = mat.numpy()
        self.assertEqualArray(expected, matnp)

    def test_as_array(self):
        r = xp.asarray(9223372036854775809)
        self.assertEqual(r.dtype, DType(TensorProto.UINT64))
        self.assertEqual(r.numpy(), 9223372036854775809)
        r = EagerTensor(np.array(9223372036854775809, dtype=np.uint64))
        self.assertEqual(r.dtype, DType(TensorProto.UINT64))
        self.assertEqual(r.numpy(), 9223372036854775809)

    def test_eye(self):
        nr, nc = xp.asarray(4), xp.asarray(4)
        expected = np.eye(nr.numpy(), nc.numpy())
        got = xp.eye(nr, nc)
        self.assertEqualArray(expected, got.numpy())

    def test_eye_nosquare(self):
        nr, nc = xp.asarray(4), xp.asarray(5)
        expected = np.eye(nr.numpy(), nc.numpy())
        got = xp.eye(nr, nc)
        self.assertEqualArray(expected, got.numpy())

    def test_eye_k(self):
        nr = xp.asarray(4)
        expected = np.eye(nr.numpy(), k=1)
        got = xp.eye(nr, k=1)
        self.assertEqualArray(expected, got.numpy())

    def test_linspace_int(self):
        a = EagerTensor(np.array([0], dtype=np.int64))
        b = EagerTensor(np.array([6], dtype=np.int64))
        c = EagerTensor(np.array(3, dtype=np.int64))
        mat = xp.linspace(a, b, c)
        matnp = mat.numpy()
        expected = np.linspace(a.numpy(), b.numpy(), c.numpy()).astype(np.int64)
        self.assertEqualArray(expected, matnp)

    def test_linspace_int5(self):
        a = EagerTensor(np.array([0], dtype=np.int64))
        b = EagerTensor(np.array([5], dtype=np.int64))
        c = EagerTensor(np.array(3, dtype=np.int64))
        mat = xp.linspace(a, b, c)
        matnp = mat.numpy()
        expected = np.linspace(a.numpy(), b.numpy(), c.numpy()).astype(np.int64)
        self.assertEqualArray(expected, matnp)

    def test_linspace_float(self):
        a = EagerTensor(np.array([0.5], dtype=np.float64))
        b = EagerTensor(np.array([5.5], dtype=np.float64))
        c = EagerTensor(np.array(2, dtype=np.int64))
        mat = xp.linspace(a, b, c)
        matnp = mat.numpy()
        expected = np.linspace(a.numpy(), b.numpy(), c.numpy())
        self.assertEqualArray(expected, matnp)

    def test_linspace_float_noendpoint(self):
        a = EagerTensor(np.array([0.5], dtype=np.float64))
        b = EagerTensor(np.array([5.5], dtype=np.float64))
        c = EagerTensor(np.array(2, dtype=np.int64))
        mat = xp.linspace(a, b, c, endpoint=0)
        matnp = mat.numpy()
        expected = np.linspace(a.numpy(), b.numpy(), c.numpy(), endpoint=0)
        self.assertEqualArray(expected, matnp)

    @ignore_warnings((RuntimeWarning, DeprecationWarning))  # division by zero
    def test_linspace_zero(self):
        expected = np.linspace(0.0, 0.0, 0, endpoint=False)
        mat = xp.linspace(0.0, 0.0, 0, endpoint=False)
        matnp = mat.numpy()
        self.assertEqualArray(expected, matnp)

    @ignore_warnings((RuntimeWarning, DeprecationWarning))  # division by zero
    def test_linspace_zero_one(self):
        expected = np.linspace(0.0, 0.0, 1, endpoint=True)

        f = linspace_inline(Input("start"), Input("stop"), Input("num"))
        onx = f.to_onnx(
            constraints={
                "start": Float64[None],
                "stop": Float64[None],
                "num": Int64[None],
                (0, False): Float64[None],
            }
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(
            None,
            {
                "start": np.array(0, dtype=np.float64),
                "stop": np.array(0, dtype=np.float64),
                "num": np.array(1, dtype=np.int64),
            },
        )
        self.assertEqualArray(expected, got[0])

        mat = xp.linspace(0.0, 0.0, 1, endpoint=True)
        matnp = mat.numpy()

        self.assertEqualArray(expected, matnp)

    def test_slice_minus_one(self):
        g = EagerTensor(np.array([0.0]))
        expected = g.numpy()[:-1]
        got = g[:-1]
        self.assertEqualArray(expected, got.numpy())


if __name__ == "__main__":
    # import logging

    # logging.basicConfig(level=logging.DEBUG)
    TestOnnxNumpy().test_slice_minus_one()
    unittest.main(verbosity=2)
