import numpy as np
from typing import Any
from onnx import helper, TensorProto


def np_dtype_to_tensor_dtype(dtype: Any):
    """
    Improves :func:`onnx.helper.np_dtype_to_tensor_dtype`.
    """
    try:
        dt = helper.np_dtype_to_tensor_dtype(dtype)
    except KeyError:
        if dtype == np.float32:
            dt = TensorProto.FLOAT
        elif dtype == np.float64:
            dt = TensorProto.DOUBLE
        elif dtype == np.int64:
            dt = TensorProto.INT64
        elif dtype == np.int32:
            dt = TensorProto.INT32
        elif dtype == np.int16:
            dt = TensorProto.INT16
        elif dtype == np.int8:
            dt = TensorProto.INT8
        elif dtype == np.uint64:
            dt = TensorProto.UINT64
        elif dtype == np.uint32:
            dt = TensorProto.UINT32
        elif dtype == np.uint16:
            dt = TensorProto.UINT16
        elif dtype == np.uint8:
            dt = TensorProto.UINT8
        elif dtype == np.float16:
            dt = TensorProto.FLOAT16
        elif dtype in (bool, np.bool_):
            dt = TensorProto.BOOL
        elif dtype in (str, np.str_):
            dt = TensorProto.STRING
        elif dtype is int:
            dt = TensorProto.INT64
        elif dtype is float:
            dt = TensorProto.DOUBLE
        else:
            raise KeyError(f"Unable to guess type for dtype={dtype}.")
    return dt
