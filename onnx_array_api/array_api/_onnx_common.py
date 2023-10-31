from typing import Any, Optional
import warnings
import numpy as np
from onnx import TensorProto

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from numpy.array_api._array_object import Array
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
    arange as generic_arange,
    copy as copy_inline,
    eye as generic_eye,
    full as generic_full,
    full_like as generic_full_like,
    linspace as generic_linspace,
    ones as generic_ones,
    zeros as generic_zeros,
)


# These functions with no specific code do not have to be
# implemented. They are automatically added in
# :mod:`onnx_array_api.array_api`. It needs
# to be added to `onnx_array_api.array_api.supported_functions`.
def abs(TEagerTensor: type, *args, **kwargs):
    return generic_abs(*args, **kwargs)


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
            va = np.asarray(a)
            v = None
            try:
                vai = np.asarray(a, dtype=np.int64)
            except OverflowError:
                v = TEagerTensor(va)
            if v is None:
                if int(va) == int(vai):
                    v = TEagerTensor(vai)
                else:
                    v = TEagerTensor(va)
    elif isinstance(a, float):
        v = TEagerTensor(np.array(a, dtype=np.float64))
    elif isinstance(a, bool):
        v = TEagerTensor(np.array(a, dtype=np.bool_))
    elif isinstance(a, str):
        v = TEagerTensor(np.array(a, dtype=np.str_))
    elif isinstance(a, list):
        if all(map(lambda x: isinstance(x, bool), a)):
            v = TEagerTensor(np.array(a, dtype=np.bool_))
        elif all(map(lambda x: isinstance(x, int), a)):
            try:
                cvt = np.array(a, dtype=np.int64)
            except OverflowError as e:
                if all(map(lambda x: x >= 0, a)):
                    cvt = np.array(a, dtype=np.uint64)
                else:
                    raise e
            v = TEagerTensor(cvt)
        else:
            v = TEagerTensor(np.array(a))
    elif isinstance(a, np.ndarray):
        v = TEagerTensor(a)
    elif isinstance(a, Array):
        v = TEagerTensor(np.asarray(a))
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


def eye(
    TEagerTensor: type,
    n_rows: TensorType[ElemType.int64, "I"],
    n_cols: OptTensorType[ElemType.int64, "I"] = None,
    /,
    *,
    k: ParType[int] = 0,
    dtype: ParType[DType] = DType(TensorProto.DOUBLE),
):
    if isinstance(n_rows, int):
        n_rows = TEagerTensor(np.array(n_rows, dtype=np.int64))
    if n_cols is None:
        n_cols = n_rows
    elif isinstance(n_cols, int):
        n_cols = TEagerTensor(np.array(n_cols, dtype=np.int64))
    return generic_eye(n_rows, n_cols, k=k, dtype=dtype)


def full_like(
    TEagerTensor: type,
    x: TensorType[ElemType.allowed, "T"],
    /,
    fill_value: ParType[Scalar] = None,
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> EagerTensor[TensorType[ElemType.allowed, "TR"]]:
    if dtype is None:
        if isinstance(fill_value, TEagerTensor):
            dtype = fill_value.dtype
        elif isinstance(x, TEagerTensor):
            dtype = x.dtype
    return generic_full_like(x, fill_value=fill_value, dtype=dtype, order=order)


def linspace(
    TEagerTensor: type,
    start: EagerTensor[TensorType[{ElemType.int64, ElemType.float64}, "I", (1,)]],
    stop: EagerTensor[
        OptTensorType[{ElemType.int64, ElemType.float64}, "I", (1,)]
    ] = None,
    num: EagerTensor[OptTensorType[ElemType.int64, "I", (1,)]] = None,
    dtype: OptParType[DType] = None,
    endpoint: ParType[int] = 1,
) -> EagerTensor[TensorType[ElemType.numerics, "T"]]:
    use_float = any(map(lambda x: isinstance(x, float), [start, stop]))
    if isinstance(start, int):
        start = TEagerTensor(
            np.array(start, dtype=np.float64 if use_float else np.int64)
        )
    elif isinstance(start, float):
        start = TEagerTensor(np.array(start, dtype=np.float64))
        assert use_float

    if isinstance(stop, int):
        stop = TEagerTensor(np.array(stop, dtype=np.float64 if use_float else np.int64))
    elif isinstance(stop, float):
        stop = TEagerTensor(np.array(stop, dtype=np.float64))
        assert use_float

    if isinstance(num, int):
        num = TEagerTensor(np.array(num, dtype=np.int64))
    elif isinstance(num, float):
        raise TypeError(f"num must be an integer not {type(num)}.")

    if dtype is None and use_float:
        dtype = DType(TensorProto.DOUBLE)
    return generic_linspace(start, stop, num, dtype=dtype, endpoint=endpoint)


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
