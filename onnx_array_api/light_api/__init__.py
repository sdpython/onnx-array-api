from typing import Dict, Optional
from onnx import ModelProto
from ..annotations import domain
from .model import OnnxGraph, ProtoType
from .var import Var, Vars


def start(
    opset: Optional[int] = None,
    opsets: Optional[Dict[str, int]] = None,
    ir_version: Optional[int] = None,
) -> OnnxGraph:
    """
    Starts an onnx model.

    :param opset: main opset version
    :param opsets: others opsets as a dictionary
    :param ir_version: specify the ir_version as well
    :return: an instance of :class:`onnx_array_api.light_api.OnnxGraph`

    A very simple model:

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start

        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
        print(onx)

    Another with operator Add:

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start

        onx = (
            start()
            .vin("X")
            .vin("Y")
            .bring("X", "Y")
            .Add()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        print(onx)
    """
    return OnnxGraph(opset=opset, opsets=opsets, ir_version=ir_version)


def g() -> OnnxGraph:
    """
    Starts a subgraph.
    :return: an instance of :class:`onnx_array_api.light_api.OnnxGraph`
    """
    return OnnxGraph(proto_type=ProtoType.GRAPH)
