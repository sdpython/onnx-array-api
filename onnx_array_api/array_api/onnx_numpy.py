"""
Array API valid for an :class:`EagerNumpyTensor`.
"""
from ..npx.npx_numpy_tensors import EagerNumpyTensor
from . import _finalize_array_api


def _finalize():
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
    from . import onnx_numpy

    _finalize_array_api(
        onnx_numpy,
        [
            "abs",
            "absolute",
            "all",
            "arange",
            "asarray",
            "astype",
            "empty",
            "equal",
            "full",
            "isdtype",
            "isfinite",
            "isnan",
            "ones",
            "ones_like",
            "reshape",
            "take",
            "zeros",
        ],
        EagerNumpyTensor,
    )


_finalize()
