import unittest
import numpy as np
from packaging.version import Version
from onnx.defs import onnx_opset_version
from sklearn import config_context, __version__ as sklearn_version
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from onnx_array_api.ext_test_case import ExtTestCase, ignore_warnings
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor


DEFAULT_OPSET = onnx_opset_version()


class TestSklearnArrayAPI(ExtTestCase):
    @unittest.skipIf(
        Version(sklearn_version) <= Version("1.2.2"),
        reason="reshape ArrayAPI not followed",
    )
    @ignore_warnings(DeprecationWarning)
    def test_sklearn_array_api_linear_discriminant(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float64
        )
        y = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
        ana = LinearDiscriminantAnalysis()
        ana.fit(X, y)
        expected = ana.predict(X)

        new_x = EagerNumpyTensor(X)
        self.assertStartsWith("EagerNumpyTensor(array([[", repr(new_x))
        with config_context(array_api_dispatch=True):
            # It fails if scikit-learn <= 1.2.2 because the ArrayAPI
            # is not strictly applied.
            got = ana.predict(new_x)
        self.assertEqualArray(expected, got.numpy())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
