from typing import Optional
import numpy as np
from onnx import TensorProto
from onnx.numpy_helper import from_array as onnx_from_array
from onnx.reference.ops.op_cast import (
    bfloat16,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import to_array_extended
from .evaluator import ExtendedReferenceEvaluator


def from_array_extended(tensor: np.array, name: Optional[str] = None) -> TensorProto:
    """
    Converts an array into a TensorProto.

    :param tensor: numpy array
    :param name: name
    :return: TensorProto
    """
    dt = tensor.dtype
    if dt == float8e4m3fn and dt.descr[0][0] == "e4m3fn":
        to = TensorProto.FLOAT8E4M3FN
        dt_to = np.uint8
    elif dt == float8e4m3fnuz and dt.descr[0][0] == "e4m3fnuz":
        to = TensorProto.FLOAT8E4M3FNUZ
        dt_to = np.uint8
    elif dt == float8e5m2 and dt.descr[0][0] == "e5m2":
        to = TensorProto.FLOAT8E5M2
        dt_to = np.uint8
    elif dt == float8e5m2fnuz and dt.descr[0][0] == "e5m2fnuz":
        to = TensorProto.FLOAT8E5M2FNUZ
        dt_to = np.uint8
    elif dt == bfloat16 and dt.descr[0][0] == "bfloat16":
        to = TensorProto.BFLOAT16
        dt_to = np.uint16
    else:
        return onnx_from_array(tensor, name)

    t = onnx_from_array(tensor.astype(dt_to), name)
    t.data_type = to
    return t
