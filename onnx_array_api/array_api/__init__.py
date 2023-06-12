import numpy as np
from onnx import TensorProto
from .._helpers import np_dtype_to_tensor_dtype
from ..npx.npx_types import DType


def _finfo(dtype):
    """
    Similar to :class:`numpy.finfo`.
    """
    dt = dtype.np_dtype if isinstance(dtype, DType) else dtype
    res = np.finfo(dt)
    d = res.__dict__.copy()
    d["dtype"] = DType(np_dtype_to_tensor_dtype(dt))
    nres = type("finfo", (res.__class__,), d)
    setattr(nres, "smallest_normal", res.smallest_normal)
    setattr(nres, "tiny", res.tiny)
    return nres


def _iinfo(dtype):
    """
    Similar to :class:`numpy.finfo`.
    """
    dt = dtype.np_dtype if isinstance(dtype, DType) else dtype
    res = np.iinfo(dt)
    d = res.__dict__.copy()
    d["dtype"] = DType(np_dtype_to_tensor_dtype(dt))
    nres = type("finfo", (res.__class__,), d)
    setattr(nres, "min", res.min)
    setattr(nres, "max", res.max)
    return nres


def _finalize_array_api(module):
    """
    Adds common attributes to Array API defined in this modules
    such as types.
    """
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
