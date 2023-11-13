from typing import Any, Dict, Optional, Tuple

import numpy
from onnx import (
    AttributeProto,
    GraphProto,
    NodeProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
)
from onnx.helper import tensor_dtype_to_np_dtype
from ..reference import to_array_extended as to_array
from ..npx.npx_types import DType


class Graph:
    __slots__ = ["g"]

    def __init__(self, g: GraphProto) -> None:
        self.g = g


class OnnxType:
    def __init__(self, type_proto: TypeProto):
        if not isinstance(type_proto, TypeProto):
            raise TypeError(f"type_proto {type(type_proto)} must be of type TypeProto.")
        self.type_proto = type_proto

    def __repr__(self) -> str:
        return f"OnnxType({self.type_proto!r})"


class SparseTensor:
    """
    Simple representation of a sparse tensor.
    It is based on numpy but does not require scipy.
    """

    def __init__(
        self, values: numpy.ndarray, indices: numpy.ndarray, shape: Tuple[int]
    ) -> None:
        self.values = values
        self.indices = indices
        self.shape = shape

    @property
    def dtype(self) -> DType:
        return self.values.dtype


def to_sparse_tensor(att: AttributeProto) -> SparseTensor:
    """
    Hosts a sparse tensor.
    """
    shape = tuple(d for d in att.dims)
    return SparseTensor(to_array(att.values), to_array(att.indices), shape)


_attribute_conversion_functions = {
    AttributeProto.FLOAT: lambda att: numpy.float32(att.f),
    AttributeProto.FLOATS: lambda att: [numpy.float32(f) for f in att.floats],
    AttributeProto.GRAPH: lambda att: Graph(att.g),
    AttributeProto.GRAPHS: lambda att: [Graph(g) for g in att.graphs],
    AttributeProto.INT: lambda att: int(att.i),
    AttributeProto.INTS: lambda att: [int(i) for i in att.ints],
    AttributeProto.SPARSE_TENSOR: lambda att: to_sparse_tensor(att.sparse_tensor),
    AttributeProto.SPARSE_TENSORS: lambda att: [
        to_sparse_tensor(t) for t in att.sparse_tensors
    ],
    AttributeProto.STRING: lambda att: att.s.decode("utf-8"),
    AttributeProto.STRINGS: lambda att: [s.decode("utf-8") for s in att.strings],
    AttributeProto.TENSOR: lambda att: to_array(att.t),
    AttributeProto.TENSORS: lambda att: [to_array(t) for t in att.tensors],
    AttributeProto.TYPE_PROTO: lambda att: OnnxType(att.tp),
    AttributeProto.TYPE_PROTOS: lambda att: [OnnxType(t) for t in att.type_protos],
}


def _extract_attribute_value(
    att: AttributeProto, ref_att: Optional[AttributeProto] = None
) -> Any:
    """
    Converts an attribute value into a python value.
    """
    if att.type == AttributeProto.GRAPH:
        return att.g
    if att.type in _attribute_conversion_functions:
        fct = _attribute_conversion_functions[att.type]
        value = fct(att)
        return value
    if ref_att is None:
        raise AttributeError(  # pragma: no cover
            f"Unable to convert attribute {att.name!r} type {att.type!r}."
        )
    raise AttributeError(  # pragma: no cover
        f"Unable to convert default value for {ref_att.name!r} " f"type {att.type!r}."
    )


def attributes_as_dict(node: NodeProto) -> Dict[str, Dict[str, Any]]:
    """
    Returns all attributes in a dictionary.
    """
    res = {}
    for att in node.attribute:
        res[att.name] = _extract_attribute_value(att)
    return res


def get_tensor_shape(obj):
    """
    Returns the shape if that makes sense for this object.
    """
    if isinstance(obj, ValueInfoProto):
        return get_tensor_shape(obj.type)
    elif not isinstance(obj, TypeProto):
        raise TypeError(f"Unexpected type {type(obj)!r}.")  # pragma: no cover
    shape = []
    for d in obj.tensor_type.shape.dim:
        v = d.dim_value if d.dim_value > 0 else d.dim_param
        shape.append(v)
    shape = None if not shape else list(None if s == 0 else s for s in shape)
    return shape


def _get_type(obj0):
    obj = obj0
    if hasattr(obj, "data_type"):
        if obj.data_type == TensorProto.FLOAT and hasattr(obj, "float_data"):
            return tensor_dtype_to_np_dtype(TensorProto.FLOAT)
        if obj.data_type == TensorProto.DOUBLE and hasattr(obj, "double_data"):
            return tensor_dtype_to_np_dtype(TensorProto.DOUBLE)
        if obj.data_type == TensorProto.INT64 and hasattr(obj, "int64_data"):
            return tensor_dtype_to_np_dtype(TensorProto.INT64)
        if obj.data_type in (
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        ) and hasattr(obj, "int32_data"):
            return tensor_dtype_to_np_dtype(TensorProto.INT32)
        if hasattr(obj, "raw_data") and len(obj.raw_data) > 0:
            arr = to_array(obj)
            return arr.dtype
        raise RuntimeError(
            f"Unable to guess type from obj.data_type={obj.data_type} "
            f"and obj={obj0!r} - {TensorProto.__dict__}."
        )
    if hasattr(obj, "type"):
        obj = obj.type
    if hasattr(obj, "tensor_type"):
        obj = obj.tensor_type
    if hasattr(obj, "elem_type"):
        if obj.elem_type == 0:
            return "NOTENSOR"
        return tensor_dtype_to_np_dtype(obj.elem_type)
    raise RuntimeError(f"Unable to guess type from {obj0!r}.")  # pragma: no cover


def _get_shape(obj):
    try:
        arr = to_array(obj)
        return arr.shape
    except Exception:
        pass
    obj0 = obj
    if hasattr(obj, "data_type"):
        if obj.data_type == TensorProto.FLOAT and hasattr(obj, "float_data"):
            return (len(obj.float_data),)
        if obj.data_type == TensorProto.DOUBLE and hasattr(obj, "double_data"):
            return (len(obj.double_data),)
        if obj.data_type == TensorProto.INT64 and hasattr(obj, "int64_data"):
            return (len(obj.int64_data),)
        if obj.data_type == TensorProto.INT32 and hasattr(obj, "int32_data"):
            return (len(obj.int32_data),)
        if hasattr(obj, "raw_data") and len(obj.raw_data) > 0:
            arr = to_array(obj)
            return arr.shape
        raise RuntimeError(  # pragma: no cover
            f"Unable to guess type from {obj0!r}, " f"data_type is {obj.data_type!r}."
        )
    if hasattr(obj, "type"):
        obj = obj.type
    if hasattr(obj, "tensor_type"):
        return get_tensor_shape(obj)
    raise RuntimeError(f"Unable to guess type from {obj0!r}.")  # pragma: no cover
