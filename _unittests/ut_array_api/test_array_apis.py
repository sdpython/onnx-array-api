import unittest
import numpy as np
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_numpy as xpn
from onnx_array_api.array_api import onnx_ort as xpo
# from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor
# from onnx_array_api.ort.ort_tensors import EagerOrtTensor


class TestArraysApis(ExtTestCase):
    def test_zeros_numpy_1(self):
        c = xpn.zeros(1)
        d = c.numpy()
        self.assertEqualArray(np.array([0], dtype=np.float32), d)

    def test_zeros_ort_1(self):
        c = xpo.zeros(1)
        d = c.numpy()
        self.assertEqualArray(np.array([0], dtype=np.float32), d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
