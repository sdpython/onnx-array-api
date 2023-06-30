import sys
import unittest
import numpy as np
from onnx import TensorProto
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_numpy as xp
from onnx_array_api.npx.npx_types import DType
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor as EagerTensor


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

    def test_arange_default(self):
        a = EagerTensor(np.array([0], dtype=np.int64))
        b = EagerTensor(np.array([2], dtype=np.int64))
        mat = xp.arange(a, b)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (2,))
        self.assertEqualArray(matnp, np.arange(0, 2).astype(np.int64))

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
        mat = xp.full(c, fill_value=False)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, tuple())
        self.assertEqulaArray(mat, matnp.numpy())


if __name__ == "__main__":
    TestOnnxNumpy().test_full_like()
    unittest.main(verbosity=2)
