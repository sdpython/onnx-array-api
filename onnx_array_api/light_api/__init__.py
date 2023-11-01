from typing import Dict, Optional
from .model import OnnxGraph
from .var import Var, Vars


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
            start().vin("X").vin("Y").bring("X", "Y").Add().rename("Z").vout().to_onnx()
        )
        print(onx)
    """
    return OnnxGraph(opset=opset, opsets=opsets, is_function=is_function)
