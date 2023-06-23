from typing import Any, Optional
import numpy as np
from ..npx.npx_types import DType
from ..npx.npx_array_api import BaseArrayApi
from ..npx.npx_functions import (
    copy as copy_inline,
)


def template_asarray(
    TEagerTensor: type,
    a: Any,
    dtype: Optional[DType] = None,
    order: Optional[str] = None,
    like: Any = None,
    copy: bool = False,
) -> Any:
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
