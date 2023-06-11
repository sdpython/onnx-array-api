from onnx import TensorProto
from ..npx.npx_types import DType


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
