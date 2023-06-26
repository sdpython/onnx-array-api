from typing import Any, Optional
import numpy as np
from ..npx.npx_types import (
    DType,
    ElemType,
    OptParType,
    OptTensorType,
    ParType,
    Scalar,
    TensorType,
)
from ..npx.npx_tensors import EagerTensor
from ..npx.npx_array_api import BaseArrayApi
from ..npx.npx_functions import (
    abs as generic_abs,
    absolute as generic_absolute,
    all as generic_all,
    arange as generic_arange,
    astype as generic_astype,
    copy as generic_copy,
    equal as generic_equal,
    full as generic_full,
    isdtype as generic_isdtype,
    isfinite as generic_isfinite,
    isnan as generic_isnan,
    ones as generic_ones,
    ones_like as generic_ones_like,
    reshape as generic_reshape,
    take as generic_take,
    zeros as generic_zeros,
)


def abs(TEagerTensor: type, *args, **kwargs):
    return generic_abs(*args, **kwargs)


def absolute(TEagerTensor: type, *args, **kwargs):
    return generic_absolute(*args, **kwargs)


def all(TEagerTensor: type, *args, **kwargs):
    return generic_all(*args, **kwargs)


def asarray(
    TEagerTensor: type,
    a: Any,
    /,
    *,
    dtype: Optional[DType] = None,
    order: Optional[str] = None,
    like: Any = None,
    copy: bool = False,
) -> EagerTensor:
    """
    Converts anything into an array.
    """
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
        if a is False:
            v = TEagerTensor(np.array(False, dtype=np.bool_))
        elif a is True:
            v = TEagerTensor(np.array(True, dtype=np.bool_))
        else:
            try:
                v = TEagerTensor(np.asarray(a, dtype=np.int64))
            except OverflowError:
                v = TEagerTensor(np.asarray(a, dtype=np.uint64))
    elif isinstance(a, float):
        v = TEagerTensor(np.array(a, dtype=np.float64))
    elif isinstance(a, bool):
        v = TEagerTensor(np.array(a, dtype=np.bool_))
    elif isinstance(a, str):
        v = TEagerTensor(np.array(a, dtype=np.str_))
    elif isinstance(a, list):
        v = TEagerTensor(np.array(a))
    else:
        raise RuntimeError(f"Unexpected type {type(a)} for the first input.")
    if dtype is not None:
        if not isinstance(dtype, DType):
            raise TypeError(f"dtype must be a DType not {type(dtype)}.")
        vt = v.astype(dtype)
    else:
        vt = v
    return vt


def arange(
    TEagerTensor: type,
    start_or_stop: EagerTensor[TensorType[ElemType.int64, "I", (1,)]],
    stop_or_step: EagerTensor[OptTensorType[ElemType.int64, "I", (1,)]] = None,
    step: EagerTensor[OptTensorType[ElemType.int64, "I", (1,)]] = None,
    dtype: OptParType[DType] = None,
) -> EagerTensor[TensorType[ElemType.numerics, "T"]]:
    use_float = any(
        map(lambda x: isinstance(x, float), [start_or_stop, stop_or_step, step])
    )
    if isinstance(start_or_stop, int):
        start_or_stop = TEagerTensor(
            np.array([start_or_stop], dtype=np.float64 if use_float else np.int64)
        )
    elif isinstance(start_or_stop, float):
        start_or_stop = TEagerTensor(np.array([start_or_stop], dtype=np.float64))
        assert use_float

    if isinstance(stop_or_step, int):
        stop_or_step = TEagerTensor(
            np.array([stop_or_step], dtype=np.float64 if use_float else np.int64)
        )
    elif isinstance(stop_or_step, float):
        stop_or_step = TEagerTensor(np.array([stop_or_step], dtype=np.float64))
        assert use_float

    if isinstance(step, int):
        step = TEagerTensor(
            np.array([step], dtype=np.float64 if use_float else np.int64)
        )
    elif isinstance(step, float):
        step = TEagerTensor(np.array([step], dtype=np.float64))
        assert use_float

    if dtype is None and use_float:
        dtype = DType(TensorProto.DOUBLE)
    return generic_arange(start_or_stop, stop_or_step, step, dtype=dtype)


def astype(TEagerTensor: type, *args, **kwargs):
    return generic_astype(*args, **kwargs)


def copy(TEagerTensor: type, *args, **kwargs):
    return generic_copy(*args, **kwargs)


def empty(
    TEagerTensor: type,
    shape: EagerTensor[TensorType[ElemType.int64, "I", (None,)]],
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> EagerTensor[TensorType[ElemType.numerics, "T"]]:
    raise RuntimeError(
        "ONNX assumes there is no inplace implementation. "
        "empty function is only used in that case."
    )


def equal(TEagerTensor: type, *args, **kwargs):
    return generic_equal(*args, **kwargs)


def full(
    TEagerTensor: type,
    shape: EagerTensor[TensorType[ElemType.int64, "I", (None,)]],
    fill_value: ParType[Scalar] = None,
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> EagerTensor[TensorType[ElemType.numerics, "T"]]:
    if fill_value is None:
        raise TypeError("fill_value cannot be None")
    value = fill_value
    if isinstance(shape, tuple):
        return generic_full(
            TEagerTensor(np.array(shape, dtype=np.int64)),
            fill_value=value,
            dtype=dtype,
            order=order,
        )
    if isinstance(shape, int):
        return generic_full(
            TEagerTensor(np.array([shape], dtype=np.int64)),
            fill_value=value,
            dtype=dtype,
            order=order,
        )
    return generic_full(shape, fill_value=value, dtype=dtype, order=order)


def isdtype(TEagerTensor: type, *args, **kwargs):
    return generic_isdtype(*args, **kwargs)


def isfinite(TEagerTensor: type, *args, **kwargs):
    return generic_isfinite(*args, **kwargs)


def isnan(TEagerTensor: type, *args, **kwargs):
    return generic_isnan(*args, **kwargs)


def ones(
    TEagerTensor: type,
    shape: EagerTensor[TensorType[ElemType.int64, "I", (None,)]],
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> EagerTensor[TensorType[ElemType.numerics, "T"]]:
    if isinstance(shape, tuple):
        return generic_ones(
            TEagerTensor(np.array(shape, dtype=np.int64)), dtype=dtype, order=order
        )
    if isinstance(shape, int):
        return generic_ones(
            TEagerTensor(np.array([shape], dtype=np.int64)),
            dtype=dtype,
            order=order,
        )
    return generic_ones(shape, dtype=dtype, order=order)


def ones_like(TEagerTensor: type, *args, **kwargs):
    return generic_ones_like(*args, **kwargs)


def reshape(TEagerTensor: type, *args, **kwargs):
    return generic_reshape(*args, **kwargs)


def take(TEagerTensor: type, *args, **kwargs):
    return generic_take(*args, **kwargs)


def zeros(
    TEagerTensor: type,
    shape: EagerTensor[TensorType[ElemType.int64, "I", (None,)]],
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> EagerTensor[TensorType[ElemType.numerics, "T"]]:
    if isinstance(shape, tuple):
        return generic_zeros(
            TEagerTensor(np.array(shape, dtype=np.int64)), dtype=dtype, order=order
        )
    if isinstance(shape, int):
        return generic_zeros(
            TEagerTensor(np.array([shape], dtype=np.int64)),
            dtype=dtype,
            order=order,
        )
    return generic_zeros(shape, dtype=dtype, order=order)
