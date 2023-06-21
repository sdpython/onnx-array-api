import unittest
import numpy as np
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_numpy as xp
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


if __name__ == "__main__":
    TestOnnxNumpy().test_arange_step()
    unittest.main(verbosity=2)
