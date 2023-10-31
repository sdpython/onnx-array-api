from typing import Any, Callable, List, Dict
import warnings
import numpy as np
from onnx import TensorProto
from .._helpers import np_dtype_to_tensor_dtype
from ..npx.npx_types import DType
from ..npx import npx_functions


supported_functions = [
    "abs",
    "absolute",
    "all",
    "any",
    "arange",
    "asarray",
    "astype",
    "empty",
    "equal",
    "eye",
    "full",
    "full_like",
    "isdtype",
    "isfinite",
    "isinf",
    "isnan",
    "linspace",
    "ones",
    "ones_like",
    "reshape",
    "sum",
    "take",
    "zeros",
    "zeros_like",
]


def _finfo(dtype):
    """
    Similar to :class:`numpy.finfo`.
    """
    dt = dtype.np_dtype if isinstance(dtype, DType) else dtype
    res = np.finfo(dt)
    d = {}
    for k, v in res.__dict__.items():
        if k.startswith("__"):
            continue
        if isinstance(v, (np.float32, np.float64, np.float16)):
            d[k] = float(v)
        else:
            d[k] = v
    d["dtype"] = DType(np_dtype_to_tensor_dtype(dt))
    nres = type("finfo", (res.__class__,), d)
    setattr(nres, "smallest_normal", float(res.smallest_normal))
    setattr(nres, "tiny", float(res.tiny))
    return nres


def _iinfo(dtype):
    """
    Similar to :class:`numpy.finfo`.
    """
    dt = dtype.np_dtype if isinstance(dtype, DType) else dtype
    res = np.iinfo(dt)
    d = {}
    for k, v in res.__dict__.items():
        if k.startswith("__"):
            continue
        if isinstance(
            v,
            (
                np.int16,
                np.int32,
                np.int64,
                np.uint16,
                np.uint32,
                np.uint64,
                np.int8,
                np.uint8,
            ),
        ):
            d[k] = int(v)
        else:
            d[k] = v
    d["dtype"] = DType(np_dtype_to_tensor_dtype(dt))
    nres = type("iinfo", (res.__class__,), d)
    setattr(nres, "min", int(res.min))
    setattr(nres, "max", int(res.max))
    return nres


def array_api_wrap_function(f: Callable, TEagerTensor: type) -> Callable:
    """
    Converts an eager function takeing EagerTensor into a function
    available through an Array API.

    :param callable: function
    :param TEagerTensor: EagerTensor class
    :return: new function
    """

    def wrap(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        new_args = []
        for a in args:
            if isinstance(a, np.ndarray):
                b = TEagerTensor(a)
            else:
                b = a
            new_args.append(b)
        res = f(TEagerTensor, *new_args, **kwargs)
        return res

    wrap.__doc__ = f.__doc__
    return wrap


def _finalize_array_api(module, function_names, TEagerTensor):
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
    from . import _onnx_common

    module.float16 = DType(TensorProto.FLOAT16)
    module.float32 = DType(TensorProto.FLOAT)
    module.float64 = DType(TensorProto.DOUBLE)
    module.int8 = DType(TensorProto.INT8)
    module.int16 = DType(TensorProto.INT16)
    module.int32 = DType(TensorProto.INT32)
    module.int64 = DType(TensorProto.INT64)
    module.uint8 = DType(TensorProto.UINT8)
    module.uint16 = DType(TensorProto.UINT16)
    module.uint32 = DType(TensorProto.UINT32)
    module.uint64 = DType(TensorProto.UINT64)
    module.bfloat16 = DType(TensorProto.BFLOAT16)
    setattr(module, "bool", DType(TensorProto.BOOL))
    setattr(module, "str", DType(TensorProto.STRING))
    setattr(module, "finfo", _finfo)
    setattr(module, "iinfo", _iinfo)

    if function_names is None:
        function_names = supported_functions

    for name in function_names:
        f = getattr(_onnx_common, name, None)
        if f is None:
            f2 = getattr(npx_functions, name, None)
            if f2 is None:
                warnings.warn(f"Function {name!r} is not available in {module!r}.")
                continue
            f = lambda TEagerTensor, *args, _f=f2, **kwargs: _f(  # noqa: E731
                *args, **kwargs
            )
        setattr(module, name, array_api_wrap_function(f, TEagerTensor))
