"""
Array API valid for an :class:`EagerNumpyTensor`.
"""
from typing import Any, Optional
import numpy as np
from onnx import TensorProto
from ..npx.npx_array_api import BaseArrayApi
from ..npx.npx_functions import (
    all,
    abs,
    absolute,
    astype,
    copy as copy_inline,
    equal,
    isdtype,
    reshape,
    take,
)
from ..npx.npx_functions import zeros as generic_zeros
from ..npx.npx_numpy_tensors import EagerNumpyTensor
from ..npx.npx_types import DType, ElemType, TensorType, OptParType
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
    if order not in ("C", None):
        raise NotImplementedError(f"asarray is not implemented for order={order!r}.")
    if like is not None:
        raise NotImplementedError(
            f"asarray is not implemented for like != None (type={type(like)})."
        )
    if isinstance(a, BaseArrayApi):
        if copy:
            if dtype is None:
                return copy_inline(a)
            return copy_inline(a).astype(dtype=dtype)
        if dtype is None:
            return a
        return a.astype(dtype=dtype)

    if isinstance(a, int):
        v = EagerNumpyTensor(np.array(a, dtype=np.int64))
    elif isinstance(a, float):
        v = EagerNumpyTensor(np.array(a, dtype=np.float32))
    elif isinstance(a, bool):
        v = EagerNumpyTensor(np.array(a, dtype=np.bool_))
    elif isinstance(a, str):
        v = EagerNumpyTensor(np.array(a, dtype=np.str_))
    else:
        raise RuntimeError(f"Unexpected type {type(a)} for the first input.")
    if dtype is not None:
        vt = v.astype(dtype=dtype)
    else:
        vt = v
    return vt


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
