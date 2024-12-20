import unittest
import numpy as np
from packaging.version import Version
from onnx.defs import onnx_opset_version
from sklearn import config_context, __version__ as sklearn_version
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from onnx_array_api.ext_test_case import ExtTestCase, skipif_ci_windows
from onnx_array_api.ort.ort_tensors import EagerOrtTensor, OrtTensor


DEFAULT_OPSET = onnx_opset_version()


class TestSklearnArrayAPIOrt(ExtTestCase):
    @unittest.skipIf(
        Version(sklearn_version) <= Version("1.2.2"),
        reason="reshape ArrayAPI not followed",
    )
    @skipif_ci_windows("Unstable on Windows.")
    @unittest.skip("discontinued")
    def test_sklearn_array_api_linear_discriminant_ort(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float64
        )
        y = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
        ana = LinearDiscriminantAnalysis()
        ana.fit(X, y)
        expected = ana.predict(X)

        new_x = EagerOrtTensor(OrtTensor.from_array(X))
        self.assertEqual(new_x.device_name, "Cpu")
        self.assertStartsWith(
            "EagerOrtTensor(OrtTensor.from_array(array([[", repr(new_x)
        )
        with config_context(array_api_dispatch=True):
            got = ana.predict(new_x)
        self.assertEqualArray(expected, got.numpy())

    @unittest.skipIf(
        Version(sklearn_version) <= Version("1.2.2"),
        reason="reshape ArrayAPI not followed",
    )
    @skipif_ci_windows("Unstable on Windows.")
    @unittest.skip("discontinued")
    def test_sklearn_array_api_linear_discriminant_ort_float32(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        y = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
        ana = LinearDiscriminantAnalysis()
        ana.fit(X, y)
        expected = ana.predict(X)

        new_x = EagerOrtTensor(OrtTensor.from_array(X))
        self.assertEqual(new_x.device_name, "Cpu")
        self.assertStartsWith(
            "EagerOrtTensor(OrtTensor.from_array(array([[", repr(new_x)
        )
        with config_context(array_api_dispatch=True):
            got = ana.predict(new_x)
        self.assertEqualArray(expected, got.numpy())


if __name__ == "__main__":
    # import logging

    # logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
