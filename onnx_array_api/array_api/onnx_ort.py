"""
Array API valid for an :class:`EagerOrtTensor`.
"""
from typing import Optional, Any
from ..ort.ort_tensors import EagerOrtTensor
from ..npx.npx_types import DType
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
from ._onnx_common import template_asarray
from . import _finalize_array_api

__all__ = [
    "all",
    "abs",
    "absolute",
    "asarray",
    "astype",
    "equal",
    "isdtype",
    "reshape",
    "take",
]


def asarray(
    a: Any,
    dtype: Optional[DType] = None,
    order: Optional[str] = None,
    like: Any = None,
    copy: bool = False,
) -> EagerOrtTensor:
    """
    Converts anything into an array.
    """
    return template_asarray(
        EagerOrtTensor, a, dtype=dtype, order=order, like=like, copy=copy
    )


def _finalize():
    from . import onnx_ort

    _finalize_array_api(onnx_ort)


_finalize()
