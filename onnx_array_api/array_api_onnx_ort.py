import inspect
from .npx import npx_functions


class onnx_ort_array_api:
    """
    Defines the ArrayApi for tensors based on :epkg:`onnxruntime`.
    It is an extension of module :mod:`onnx_array_api.npx.npx_functions`.
    """

    pass


def _setup():
    for k, v in npx_functions.__dict__.items():
        if inspect.isfunction(v):
            setattr(onnx_ort_array_api, k, v)


_setup()
