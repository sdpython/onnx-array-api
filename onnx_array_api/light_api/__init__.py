from typing import Dict, Optional
from onnx import ModelProto
from .annotations import domain
from .model import OnnxGraph, ProtoType
from .translate import Translater
from .var import Var, Vars
from .inner_emitter import InnerEmitter


def start(
    opset: Optional[int] = None,
    opsets: Optional[Dict[str, int]] = None,
) -> OnnxGraph:
    """
    Starts an onnx model.

    :param opset: main opset version
    :param opsets: others opsets as a dictionary
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
    return OnnxGraph(opset=opset, opsets=opsets)


def g() -> OnnxGraph:
    """
    Starts a subgraph.
    :return: an instance of :class:`onnx_array_api.light_api.OnnxGraph`
    """
    return OnnxGraph(proto_type=ProtoType.GRAPH)


def translate(proto: ModelProto, single_line: bool = False, api: str = "light") -> str:
    """
    Translates an ONNX proto into a code using :ref:`l-light-api`
    to describe the ONNX graph.

    :param proto: model to translate
    :param single_line: as a single line or not
    :param api: API to export into,
        default is `"light"` and this is handle by class
        :class:`onnx_array_api.light_api.emitter.Emitter`,
        another value is `"onnx"` which is the inner API implemented
        in onnx package.
    :return: code

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start, translate

        onx = (
            start()
            .vin("X")
            .reshape((-1, 1))
            .Transpose(perm=[1, 0])
            .rename("Y")
            .vout()
            .to_onnx()
        )
        code = translate(onx)
        print(code)

    The inner API from onnx packahe is also available.

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start, translate

        onx = (
            start()
            .vin("X")
            .reshape((-1, 1))
            .Transpose(perm=[1, 0])
            .rename("Y")
            .vout()
            .to_onnx()
        )
        code = translate(onx, api="onnx")
        print(code)
    """
    if api == "light":
        tr = Translater(proto)
        return tr.export(single_line=single_line, as_str=True)
    if api == "onnx":
        tr = Translater(proto, emitter=InnerEmitter())
        return tr.export(as_str=True)
    raise ValueError(f"Unexpected value {api!r} for api.")
