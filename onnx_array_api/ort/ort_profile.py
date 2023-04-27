import json
import os
from typing import Any, Dict, List, Optional, Union
import numpy
from onnx import ModelProto
from pandas import DataFrame


def ort_profile(
    filename_or_bytes: Union[str, bytes, ModelProto],
    feeds: Dict[str, numpy.ndarray],
    sess_options: Optional[Any] = None,
    disable_optimization: bool = False,
    repeat: int = 10,
    as_df: bool = True,
    providers: Optional[List[str]] = None,
    **kwargs,
) -> Union[List, DataFrame]:
    """
    Profiles the execution of an onnx graph with onnxruntime.

    :param filename_or_bytes: filename or bytes
    :param feeds: inputs, dictionary of numpy arrays
    :param sess_options: instance of :class:`onnxruntime.SessionOptions`
    :param disable_optimization: disable onnxruntime optimization
    :param repeat: number of times to run the inference
    :param as_df: returns the
    :param providers: list of providers to use when initializing the inference session
    :param kwargs: additional parameters when initializing the inference session
    :return: DataFrame or dictionary
    """
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

    if sess_options is None:
        sess_options = SessionOptions()
    if disable_optimization:
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.enable_profiling = True
    obj = (
        filename_or_bytes.SerializeToString()
        if isinstance(filename_or_bytes, ModelProto)
        else filename_or_bytes
    )
    sess = InferenceSession(obj, sess_options, providers=providers, **kwargs)
    for i in range(repeat):
        sess.run(None, feeds)
    prof = sess.end_profiling()
    with open(prof, "r") as f:
        content = f.read()
    js = json.loads(content)
    os.remove(prof)

    suffixes = ["_kernel_time", "_fence_before", "_fence_after"]
    rows = []
    for row in js:
        if "args" in row and isinstance(row["args"], dict):
            for k, v in row["args"].items():
                row[f"args_{k}"] = v
            del row["args"]
        name = row["name"]
        for suf in suffixes:
            if name.endswith(suf):
                changed = name[: -len(suf)]
                row["op_name"] = changed
                break
        rows.append(row)
    if as_df:
        return DataFrame(rows)
    return rows
