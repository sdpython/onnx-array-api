import unittest
import numpy as np
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_numpy as xp
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor


class TestOnnxNumpy(ExtTestCase):
    def test_abs(self):
        c = EagerNumpyTensor(np.array([4, 5], dtype=np.int64))
        mat = xp.zeros(c, dtype=xp.int64)
        a = xp.absolute(mat)
        self.assertEqualArray(np.absolute(mat.numpy()), a.numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)