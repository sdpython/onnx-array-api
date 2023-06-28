import unittest
import numpy as np
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_ort as xp
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor
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

    def test_matmul(self):
        for cls in [EagerTensor, EagerNumpyTensor]:
            for dtype in (np.float32, np.float64):
                X = cls(
                    np.array(
                        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                        dtype=dtype,
                    )
                )
                coef = cls(np.array([[1e-13, 8]], dtype=dtype).T)
                self.assertEqualArray(
                    np.array([[1e-13, 8]], dtype=dtype), coef.numpy().T
                )
                expected = X.numpy() @ coef.numpy()
                got = X @ coef
                try:
                    self.assertEqualArray(expected, got.numpy())
                except AssertionError as e:
                    raise AssertionError(
                        f"Discrepancies (1) with cls={cls.__name__}, dtype={dtype}"
                    ) from e

                coef = np.array([[1e-13, 8]], dtype=dtype).T
                expected = X.numpy() @ coef
                got = X @ coef
                try:
                    self.assertEqualArray(expected, got.numpy())
                except AssertionError as e:
                    raise AssertionError(
                        f"Discrepancies (2) with cls={cls.__name__}, dtype={dtype}"
                    ) from e


if __name__ == "__main__":
    # import logging

    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxOrt().test_matmul()
    unittest.main(verbosity=2)
