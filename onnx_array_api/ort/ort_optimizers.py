from typing import Union
from onnx import ModelProto, load
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi._pybind_state import GraphOptimizationLevel
from ..cache import get_cache_file


def ort_optimized_model(
    onx: Union[str, ModelProto], level: str = "ORT_ENABLE_ALL"
) -> Union[str, ModelProto]:
    """
    Returns the optimized model used by onnxruntime before
    running computing the inference.

    :param onx: ModelProto
    :param level: optimization level, `'ORT_ENABLE_BASIC'`,
        `'ORT_ENABLE_EXTENDED'`, `'ORT_ENABLE_ALL'`
    :return: optimized model
    """
    glevel = getattr(GraphOptimizationLevel, level, None)
    if glevel is None:
        raise ValueError(
            f"Unrecognized level {level!r} among {dir(GraphOptimizationLevel)}."
        )

    cache = get_cache_file("ort_optimized_model.onnx", remove=True)
    so = SessionOptions()
    so.graph_optimization_level = glevel
    so.optimized_model_filepath = str(cache)
    InferenceSession(onx if isinstance(onx, str) else onx.SerializeToString(), so)
    if not cache.exists():
        raise RuntimeError(f"The optimized model {str(cache)!r} not found.")
    if isinstance(onx, str):
        return str(cache)
    opt_onx = load(str(cache))
    cache.unlink()
    return opt_onx
