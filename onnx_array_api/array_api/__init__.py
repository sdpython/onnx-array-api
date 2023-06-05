from onnx import TensorProto


def _finalize_array_api(module):
    module.float16 = TensorProto.FLOAT16
    module.float32 = TensorProto.FLOAT
    module.float64 = TensorProto.DOUBLE
    module.int8 = TensorProto.INT8
    module.int16 = TensorProto.INT16
    module.int32 = TensorProto.INT32
    module.int64 = TensorProto.INT64
    module.uint8 = TensorProto.UINT8
    module.uint16 = TensorProto.UINT16
    module.uint32 = TensorProto.UINT32
    module.uint64 = TensorProto.UINT64
    module.bfloat16 = TensorProto.BFLOAT16
    setattr(module, "bool", TensorProto.BOOL)
    # setattr(module, "str", TensorProto.STRING)
