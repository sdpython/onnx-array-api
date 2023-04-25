import unittest
import numpy as np
from pandas import DataFrame
from onnx_array_api.npx import absolute, jit_onnx
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.ort.ort_optimizers import ort_optimized_model
from onnx_array_api.ort.ort_profile import ort_profile


class TestOrtProfile(ExtTestCase):
    def test_ort_profile(self):
        def l1_loss(x, y):
            return absolute(x - y).sum()

        def l2_loss(x, y):
            return ((x - y) ** 2).sum()

        def myloss(x, y):
            return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])

        jitted_myloss = jit_onnx(myloss)
        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)
        jitted_myloss(x, y)
        onx = jitted_myloss.get_onnx()
        feeds = {"x0": x, "x1": y}
        self.assertRaise(lambda: ort_optimized_model(onx, "NO"), ValueError)
        optimized = ort_optimized_model(onx)
        prof = ort_profile(optimized, feeds)
        self.assertIsInstance(prof, DataFrame)
        prof = ort_profile(optimized, feeds, as_df=False)
        self.assertIsInstance(prof, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
