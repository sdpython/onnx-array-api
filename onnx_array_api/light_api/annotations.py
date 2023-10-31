from typing import Tuple, Union
import numpy as np
from onnx import FunctionProto, GraphProto, ModelProto, TensorProto, TensorShapeProto
from onnx.helper import np_dtype_to_tensor_dtype

NP_DTYPE = np.dtype
ELEMENT_TYPE = Union[int, NP_DTYPE]
SHAPE_TYPE = Tuple[int, ...]
VAR_CONSTANT_TYPE = Union["Var", TensorProto, np.ndarray]
GRAPH_PROTO = Union[FunctionProto, GraphProto, ModelProto]

ELEMENT_TYPE_NAME = {
    getattr(TensorProto, k): k
    for k in dir(TensorProto)
    if isinstance(getattr(TensorProto, k), int)
}


def elem_type_int(elem_type: ELEMENT_TYPE) -> int:
    """
    Converts an element type into an onnx element type (int).

    :param elem_type: integer or numpy type
    :return: int
    """
    if isinstance(elem_type, int):
        return elem_type
    return np_dtype_to_tensor_dtype(elem_type)


def make_shape(shape: TensorShapeProto) -> SHAPE_TYPE:
    "Extracts a shape from a tensor type."
    if hasattr(shape, "dims"):
        res = [(d.dim_value if d.dim_value else d.dim_param) for d in shape.dims]
        return tuple(res)
    return None
