"""
Array API valid for an :class:`EagerOrtTensor`.
"""
from ..npx.npx_functions import (
    all,
    abs,
    absolute,
    astype,
    equal,
    isdtype,
    reshape,
    take,
)
from . import _finalize_array_api

__all__ = [
    "all",
    "abs",
    "absolute",
    "astype",
    "equal",
    "isdtype",
    "reshape",
    "take",
]


def _finalize():
    from . import onnx_ort

    _finalize_array_api(onnx_ort)


_finalize()
