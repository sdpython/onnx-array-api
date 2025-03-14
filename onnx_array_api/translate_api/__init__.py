from onnx import ModelProto
from .translate import Translater
from .inner_emitter import InnerEmitter, InnerEmitterShortInitializer
from .builder_emitter import BuilderEmitter


def translate(proto: ModelProto, single_line: bool = False, api: str = "light") -> str:
    """
    Translates an ONNX proto into a code using :ref:`l-light-api`
    to describe the ONNX graph.

    :param proto: model to translate
    :param single_line: as a single line or not
    :param api: API to export into,
        default is `"light"` and this is handle by class
        :class:`onnx_array_api.translate_api.light_emitter.LightEmitter`,
        another value is `"onnx"` which is the inner API implemented
        in onnx package, `"builder"` follows the syntax for the
        class :class:`onnx_array_api.graph_api.GraphBuilder`,
        `"onnx-short"` replaces long initializer with random values
    :return: code

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start
        from onnx_array_api.translate_api import translate

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

    The inner API from onnx package is also available.

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start
        from onnx_array_api.translate_api import translate

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

    The :class:`GraphBuilder
    <onnx_array_api.graph_api.GraphBuilder>` API returns this:

    .. runpython::
        :showcode:

        from onnx_array_api.light_api import start
        from onnx_array_api.translate_api import translate

        onx = (
            start()
            .vin("X")
            .reshape((-1, 1))
            .Transpose(perm=[1, 0])
            .rename("Y")
            .vout()
            .to_onnx()
        )
        code = translate(onx, api="builder")
        print(code)
    """
    if api == "light":
        tr = Translater(proto)
        return tr.export(single_line=single_line, as_str=True)
    if api == "onnx":
        tr = Translater(proto, emitter=InnerEmitter())
        return tr.export(as_str=True)
    if api == "onnx-short":
        tr = Translater(proto, emitter=InnerEmitterShortInitializer())
        return tr.export(as_str=True)
    if api == "builder":
        tr = Translater(proto, emitter=BuilderEmitter())
        return tr.export(as_str=True)
    raise ValueError(f"Unexpected value {api!r} for api.")
