import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
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
    :param providers: list of providers to use when initializing the inference session,
        if None, the default value is `["CPUExecutionProvider"]`
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
    if providers is None:
        providers = ["CPUExecutionProvider"]
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


def _merge_ort_profile_preprocess(df):
    groupkey = [
        "args_op_name",
        "args_output_type_shape",
        "args_input_type_shape",
        "args_provider",
    ]

    def _idx(row):
        """
        There may be multiple node with the same
        input/output types and shapes.
        This function gives every instance a distinct id.
        First unique op with same I/O receives the index 0.
        The counter restart when the session goes to the
        next image.
        """
        if row["cat"] == "Session":
            occurences[0] = {}
            return -1
        assert "idx" not in groupkey
        vals = [row[k] for k in groupkey]
        key = tuple(map(str, vals))
        if key not in occurences[0]:
            occurences[0][key] = 0
        else:
            occurences[0][key] += 1
        return occurences[0][key]

    df = df.copy()
    occurences = [{}]
    df["idx"] = df.apply(_idx, axis=1)
    df = df[(df["cat"] == "Node") & df["name"].str.contains("kernel_time")]
    groupkey.append("idx")
    for c in groupkey:
        if c != "idx":
            df[c] = df[c].apply(str)
    df = df.copy()
    df["count"] = 1
    gr = df[groupkey + ["dur", "count"]].groupby(groupkey)
    return gr.sum()


def _process_shape(s: Tuple[int, ...], keys: Dict[str, str]) -> str:
    value = eval(s)
    ns = []
    for v in value:
        if len(v) != 1:
            raise NotImplementedError(f"Unexpected value {v} in {s!r}.")
        k, v = list(v.items())[0]
        n = "-".join([keys[k], "x".join(map(str, v))])
        ns.append(n)
    return ",".join(ns)


def _label(row: Dict[str, Any], column: Optional[str], keys: Dict[str, str]) -> str:
    name = row["args_op_name"]
    inshape = _process_shape(row["args_input_type_shape"], keys)
    outshape = _process_shape(row["args_output_type_shape"], keys)
    side = row["side"][0]
    prov = row["args_provider"][:3]
    add = "" if column is None else f"[{row[column]}]"
    return f"[{side}{prov}]{name}({inshape})->{outshape}{add}"


def merge_ort_profile(
    prof1: DataFrame,
    prof2: DataFrame,
    suffixes: Tuple[str, str] = ("base", "opti"),
    by_column: Optional[str] = None,
) -> Tuple[DataFrame, DataFrame]:
    """
    Merges two profiles produced by function  :func:`ort_profile
    <onnx_array_api.ort.ort_profile.ort_profile>`.

    :param prof1: first profile
    :param prof2: second profile
    :param suffixes: used by pandas merge
    :param by_column: the second profile merged by input, output shapes and types
        plus an additional column, usually `None`, `'idx'` or `'op_name'`
    :return: merged profiles
    """
    # First merge
    base = _merge_ort_profile_preprocess(prof1)
    opti = _merge_ort_profile_preprocess(prof2)
    merge = base.merge(
        opti, how="outer", suffixes=suffixes, left_index=True, right_index=True
    )
    merge = merge.reset_index(drop=False)

    # Second merge

    def classify(row):
        if numpy.isnan(row[f"dur{suffixes[1]}"]):
            return "-"
        if numpy.isnan(row[f"dur{suffixes[0]}"]):
            return "+"
        return "="

    keys = {"float": "f"}

    df = merge.copy()
    df["side"] = df.apply(classify, axis=1)
    df["label"] = df.apply(lambda row: _label(row, by_column, keys), axis=1)
    gr = (
        df[
            [
                "label",
                f"dur{suffixes[0]}",
                f"dur{suffixes[1]}",
                f"count{suffixes[0]}",
                f"count{suffixes[1]}",
            ]
        ]
        .groupby("label")
        .agg(
            {
                f"dur{suffixes[0]}": numpy.sum,
                f"dur{suffixes[1]}": numpy.sum,
                f"count{suffixes[0]}": numpy.sum,
                f"count{suffixes[1]}": numpy.sum,
            }
        )
    )
    return merge, gr
