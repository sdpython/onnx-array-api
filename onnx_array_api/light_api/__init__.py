from typing import Dict, Optional
from onnx import ModelProto
from .model import OnnxGraph
from .translate import Translater
from .var import Var, Vars
from .inner_emitter import InnerEmitter


def start(
    opset: Optional[int] = None,
    opsets: Optional[Dict[str, int]] = None,
    is_function: bool = False,
) -> OnnxGraph:
    """
    Starts an onnx model.

    :param opset: main opset version
    :param is_function: a :class:`onnx.ModelProto` or a :class:`onnx.FunctionProto`
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
    return OnnxGraph(opset=opset, opsets=opsets, is_function=is_function)


def translate(proto: ModelProto, single_line: bool = False, api: str = "light") -> str:
    """
    Translates an ONNX proto into a code using :ref:`l-light-api`
    to describe the ONNX graph.

    :param proto: model to translate
    :param single_line: as a single line or not
    :param api: API to export into,
        default is `"light"` and this is handle by class
        :class:`onnx_array_api.light_api.translate.Emitter`,
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
    elif api == "onnx":
        tr = Translater(proto, emitter=InnerEmitter())
    else:
        raise ValueError(f"Unexpected value {api!r} for api.")
    return tr.export(single_line=single_line, as_str=True)
