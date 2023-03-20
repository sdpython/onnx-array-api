import unittest
import numpy as np
from onnx.defs import onnx_opset_version
from sklearn import config_context
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor


DEFAULT_OPSET = onnx_opset_version()


def take(self, X, indices, *, axis):
    # Overwritting method take as it is using iterators.
    # When array_api supports `take` we can use this directly
    # https://github.com/data-apis/array-api/issues/177
    X_np = self._namespace.take(X, indices, axis=axis)
    return self._namespace.asarray(X_np)


class TestSklearnArrayAPI(ExtTestCase):
    def test_sklearn_array_api_linear_discriminant(self):
        from sklearn.utils._array_api import _ArrayAPIWrapper

        _ArrayAPIWrapper.take = take
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        ana = LinearDiscriminantAnalysis()
        ana = LinearDiscriminantAnalysis()
        ana.fit(X, y)
        expected = ana.predict(X)

        new_x = EagerNumpyTensor(X)
        self.assertStartsWith("EagerNumpyTensor(array([[", repr(new_x))
        with config_context(array_api_dispatch=True):
            got = ana.predict(new_x)
        self.assertEqualArray(expected, got.numpy())


if __name__ == "__main__":
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
