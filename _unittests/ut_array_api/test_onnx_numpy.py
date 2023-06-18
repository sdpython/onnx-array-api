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
    TestOnnxNumpy().test_full_bool()
    unittest.main(verbosity=2)
