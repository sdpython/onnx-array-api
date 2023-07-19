from ..npx.npx_numpy_tensors import EagerNumpyTensor
from . import _finalize_array_api


def _finalize():
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
    from . import onnx_numpy

    _finalize_array_api(onnx_numpy, None, EagerNumpyTensor)


_finalize()
