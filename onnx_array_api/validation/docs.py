from typing import Optional, Tuple
import numpy as np
import onnx
import onnx.helper as oh


def make_euclidean(
    input_names: Tuple[str] = ("X", "Y"),
    output_name: str = "Z",
    elem_type: int = onnx.TensorProto.FLOAT,
    opset: Optional[int] = None,
) -> onnx.ModelProto:
    """
    Creates the onnx graph corresponding to the euclidean distance.

    :param input_names: names of the inputs
    :param output_name: name of the output
    :param elem_type: onnx is strongly types, which type is it?
    :param opset: opset version
    :return: onnx.ModelProto
    """
    if opset is None:
        opset = onnx.defs.onnx_opset_version()

    X = oh.make_tensor_value_info(input_names[0], elem_type, None)
    Y = oh.make_tensor_value_info(input_names[1], elem_type, None)
    Z = oh.make_tensor_value_info(output_name, elem_type, None)
    two = oh.make_tensor("two", onnx.TensorProto.INT64, [1], [2])
    n1 = oh.make_node("Sub", ["X", "Y"], ["dxy"])
    n2 = oh.make_node("Pow", ["dxy", "two"], ["dxy2"])
    n3 = oh.make_node("ReduceSum", ["dxy2"], [output_name])
    graph = oh.make_graph([n1, n2, n3], "euclidian", [X, Y], [Z], [two])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    return model


def make_euclidean_skl2onnx(
    input_names: Tuple[str] = ("X", "Y"),
    output_name: str = "Z",
    elem_type: int = onnx.TensorProto.FLOAT,
    opset: Optional[int] = None,
) -> onnx.ModelProto:
    """
    Creates the onnx graph corresponding to the euclidean distance
    with :epkg:`sklearn-onnx`.

    :param input_names: names of the inputs
    :param output_name: name of the output
    :param elem_type: onnx is strongly types, which type is it?
    :param opset: opset version
    :return: onnx.ModelProto
    """
    if opset is None:
        opset = onnx.defs.onnx_opset_version()

    from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxPow, OnnxReduceSum

    dxy = OnnxSub(input_names[0], input_names[1], op_version=opset)
    dxy2 = OnnxPow(dxy, np.array([2], dtype=np.int64), op_version=opset)
    final = OnnxReduceSum(dxy2, op_version=opset, output_names=[output_name])

    np_type = oh.tensor_dtype_to_np_dtype(elem_type)
    dummy = np.empty([1], np_type)
    return final.to_onnx({"X": dummy, "Y": dummy})
