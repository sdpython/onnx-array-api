"""
Array API valid for an :class:`EagerNumpyTensor`.
"""
from typing import Any, Optional
import numpy as np
from ..npx.npx_functions import (
    all,
    abs,
    absolute,
    astype,
    equal,
    isdtype,
    isfinite,
    isnan,
    reshape,
    take,
)
from ..npx.npx_functions import arange as generic_arange
from ..npx.npx_functions import full as generic_full
from ..npx.npx_functions import ones as generic_ones
from ..npx.npx_functions import zeros as generic_zeros
from ..npx.npx_numpy_tensors import EagerNumpyTensor
from ..npx.npx_types import (
    DType,
    ElemType,
    TensorType,
    OptParType,
    OptTensorType,
    ParType,
    Scalar,
)
from ._onnx_common import template_asarray
from . import _finalize_array_api

__all__ = [
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


def arange(
    start_or_stop: TensorType[ElemType.int64, "I", (1,)],
    stop_or_step: OptTensorType[ElemType.int64, "I", (1,)] = None,
    step: OptTensorType[ElemType.int64, "I", (1,)] = None,
    dtype: OptParType[DType] = None,
) -> TensorType[ElemType.numerics, "T"]:
    print("####", start_or_stop, stop_or_step, step, dtype)
    if isinstance(start_or_stop, int):
        start_or_stop = EagerNumpyTensor(np.array([start_or_stop], dtype=np.int64))
    if isinstance(stop_or_step, int):
        stop_or_step = EagerNumpyTensor(np.array([stop_or_step], dtype=np.int64))
    if isinstance(step, int):
        step = EagerNumpyTensor(np.array([step], dtype=np.int64))
    return generic_arange(start_or_stop, stop_or_step, step, dtype=dtype)


def ones(
    shape: TensorType[ElemType.int64, "I", (None,)],
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    if isinstance(shape, tuple):
        return generic_ones(
            EagerNumpyTensor(np.array(shape, dtype=np.int64)), dtype=dtype, order=order
        )
    if isinstance(shape, int):
        return generic_ones(
            EagerNumpyTensor(np.array([shape], dtype=np.int64)),
            dtype=dtype,
            order=order,
        )
    return generic_ones(shape, dtype=dtype, order=order)


def empty(
    shape: TensorType[ElemType.int64, "I", (None,)],
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    raise RuntimeError(
        "ONNX assumes there is no inplace implementation. "
        "empty function is only used in that case."
    )


def zeros(
    shape: TensorType[ElemType.int64, "I", (None,)],
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    if isinstance(shape, tuple):
        return generic_zeros(
            EagerNumpyTensor(np.array(shape, dtype=np.int64)), dtype=dtype, order=order
        )
    if isinstance(shape, int):
        return generic_zeros(
            EagerNumpyTensor(np.array([shape], dtype=np.int64)),
            dtype=dtype,
            order=order,
        )
    return generic_zeros(shape, dtype=dtype, order=order)


def full(
    shape: TensorType[ElemType.int64, "I", (None,)],
    fill_value: ParType[Scalar] = None,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    if fill_value is None:
        raise TypeError("fill_value cannot be None")
    value = fill_value
    if isinstance(shape, tuple):
        return generic_full(
            EagerNumpyTensor(np.array(shape, dtype=np.int64)),
            fill_value=value,
            dtype=dtype,
            order=order,
        )
    if isinstance(shape, int):
        return generic_full(
            EagerNumpyTensor(np.array([shape], dtype=np.int64)),
            fill_value=value,
            dtype=dtype,
            order=order,
        )
    return generic_full(shape, fill_value=value, dtype=dtype, order=order)


def _finalize():
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
    from . import onnx_numpy

    _finalize_array_api(onnx_numpy)


_finalize()
