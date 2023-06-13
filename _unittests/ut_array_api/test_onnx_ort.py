import unittest
import numpy as np
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_ort as xp
from onnx_array_api.ort.ort_tensors import EagerOrtTensor as EagerTensor


class TestOnnxOrt(ExtTestCase):
    def test_abs(self):
        c = EagerTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.zeros(c, dtype=xp.int64)
        matnp = mat.numpy()
        self.assertEqual(matnp.shape, (4, 5))
        self.assertNotEmpty(matnp[0, 0])
        a = xp.absolute(mat)
        self.assertEqualArray(np.absolute(mat.numpy()), a.numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)