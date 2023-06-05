"""
Array API valid for an :class:`EagerOrtTensor`.
"""
from typing import Any, Optional
import numpy as np
from ..npx.npx_array_api import BaseArrayApi
from ..npx.npx_functions import (
    abs,
    absolute,
    astype,
    isdtype,
    reshape,
    take,
)
from ..npx.npx_functions import asarray as generic_asarray
from ..ort.ort_tensors import EagerOrtTensor
from . import _finalize_array_api

__all__ = [
    "abs",
    "absolute",
    "asarray",
    "astype",
    "isdtype",
    "reshape",
    "take",
]


def asarray(
    a: Any,
    dtype: Any = None,
    order: Optional[str] = None,
    like: Any = None,
    copy: bool = False,
):
    """
    Converts anything into an array.
    """
    if isinstance(a, BaseArrayApi):
        return generic_asarray(a, dtype=dtype, order=order, like=like, copy=copy)
    if isinstance(a, int):
        return EagerOrtTensor(np.array(a, dtype=np.int64))
    if isinstance(a, float):
        return EagerOrtTensor(np.array(a, dtype=np.float32))
    raise NotImplementedError(f"asarray not implemented for type {type(a)}.")


def _finalize():
    from . import onnx_ort

    _finalize_array_api(onnx_ort)


_finalize()
