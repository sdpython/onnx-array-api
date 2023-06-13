"""
Array API valid for an :class:`EagerOrtTensor`.
"""
from typing import Optional, Any
import numpy as np
from onnx import TensorProto
from ..ort.ort_tensors import EagerOrtTensor
from ..npx.npx_functions import (
    all,
    abs,
    absolute,
    astype,
    equal,
    isdtype,
    isnan,
    isfinite,
    reshape,
    take,
)
from ..npx.npx_types import DType, ElemType, TensorType, OptParType
from ..npx.npx_functions import zeros as generic_zeros
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
    "isfinite",
    "isnan",
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


def zeros(
    shape: TensorType[ElemType.int64, "I", (None,)],
    dtype: OptParType[DType] = DType(TensorProto.FLOAT),
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    if isinstance(shape, tuple):
        return generic_zeros(
            EagerOrtTensor(np.array(shape, dtype=np.int64)), dtype=dtype, order=order
        )
    if isinstance(shape, int):
        return generic_zeros(
            EagerOrtTensor(np.array([shape], dtype=np.int64)), dtype=dtype, order=order
        )
    return generic_zeros(shape, dtype=dtype, order=order)


def _finalize():
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
    from . import onnx_ort

    _finalize_array_api(onnx_ort)


_finalize()
