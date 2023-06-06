"""
Array API valid for an :class:`EagerNumpyTensor`.
"""
from typing import Any, Optional
import numpy as np
from onnx import TensorProto
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
from ..npx.npx_functions import zeros as generic_zeros
from ..npx.npx_numpy_tensors import EagerNumpyTensor
from ..npx.npx_types import DType, ElemType, TensorType, OptParType
from ._onnx_common import template_asarray
from . import _finalize_array_api

__all__ = [
    "abs",
    "absolute",
    "all",
    "asarray",
    "astype",
    "equal",
    "isdtype",
    "reshape",
    "take",
    "zeros",
]


def asarray(
    a: Any,
    dtype: Optional[DType] = None,
    order: Optional[str] = None,
    like: Any = None,
    copy: bool = False,
) -> EagerNumpyTensor:
    """
    Converts anything into an array.
    """
    return template_asarray(
        EagerNumpyTensor, a, dtype=dtype, order=order, like=like, copy=copy
    )


def zeros(
    shape: TensorType[ElemType.int64, "I", (None,)],
    dtype: OptParType[DType] = DType(TensorProto.FLOAT),
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    if isinstance(shape, tuple):
        return generic_zeros(
            EagerNumpyTensor(np.array(shape, dtype=np.int64)), dtype=dtype, order=order
        )
    return generic_zeros(shape, dtype=dtype, order=order)


def _finalize():
    from . import onnx_numpy

    _finalize_array_api(onnx_numpy)


_finalize()
