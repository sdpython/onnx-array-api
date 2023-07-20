from ..ort.ort_tensors import EagerOrtTensor
from . import _finalize_array_api


def _finalize():
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
    from . import onnx_ort

    _finalize_array_api(onnx_ort, None, EagerOrtTensor)


_finalize()
