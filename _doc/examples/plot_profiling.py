"""

.. _l-onnx-array-onnxruntime-profiling:

Profiling with onnxruntime
==========================

*onnxruntime* optimizes the onnx graph by default before running
the inference. It modifies, fuses or add new operators.
Some of them are standard onnx operators, some of them
are implemented in onnxruntime (see `Supported Operators
<https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md>`_).
This example profiles the two models.

Optimize a model with onnxruntime
+++++++++++++++++++++++++++++++++
"""
import os
import numpy
import matplotlib.pyplot as plt
from onnx_array_api.ext_test_case import example_path
from onnx_array_api.ort.ort_optimizers import ort_optimized_model
from onnx_array_api.ort.ort_profile import ort_profile


filename = example_path("data/small.onnx")
optimized = filename + ".optimized.onnx"

if not os.path.exists(optimized):
    ort_optimized_model(filename, output=optimized)
print(optimized)

#############################
# Profiling
# +++++++++

feeds = {"input": numpy.random.random((1, 3, 112, 112)).astype(numpy.float32)}
prof_base = ort_profile(filename, feeds, repeat=6, disable_optimization=True)
prof_base.to_excel("prof_base.xlsx", index=False)
prof_base

#######################################
# And the optimized model.

prof_opt = ort_profile(optimized, feeds, repeat=6, disable_optimization=True)
prof_opt

#######################################
# And the graph is:


def plot_profile(df, ax0, ax1=None, title=None):
    gr_dur = (
        df[["dur", "args_op_name"]].groupby("args_op_name").sum().sort_values("dur")
    )
    gr_dur.plot.barh(ax=ax0)
    if title is not None:
        ax0.set_title(title)
    if ax1 is not None:
        gr_n = (
            df[["dur", "args_op_name"]]
            .groupby("args_op_name")
            .count()
            .sort_values("dur")
        )
        gr_n = gr_n.loc[gr_dur.index, :]
        gr_n.plot.barh(ax=ax1)
        ax1.set_title("n occurences")


unique_op = set(prof_base["args_op_name"])
fig, ax = plt.subplots(2, 2, figsize=(10, len(unique_op)), sharex="col")
plot_profile(prof_base, ax[0, 0], ax[0, 1], title="baseline")
plot_profile(prof_opt, ax[1, 0], ax[1, 1], title="optimized")

fig.savefig("plot_profiling.png")

##################################################
# Merging profiles
# ++++++++++++++++
#
# Let's try to compare both profiles assuming every iteration
# process the same image and the input and output size are the
# same at every iteration.


def preprocess(df):
    df = df[(df["cat"] == "Node") & df["name"].str.contains("kernel_time")]
    groupkey = [
        "args_op_name",
        "args_output_type_shape",
        "args_input_type_shape",
        "args_provider",
    ]
    df = df.copy()
    for c in groupkey:
        df[c] = df[c].apply(str)
    gr = df[groupkey + ["dur"]].groupby(groupkey)
    return gr.sum()


base = preprocess(prof_base)
opti = preprocess(prof_opt)
merge = base.merge(
    opti, how="outer", suffixes=("base", "opti"), left_index=True, right_index=True
)
merge = merge.reset_index(drop=False)
merge

#####################################################
# Aggregation


def classify(row):
    if numpy.isnan(row["duropti"]):
        return "-"
    if numpy.isnan(row["durbase"]):
        return "+"
    return "="


keys = {"float": "f"}


def process_shape(s):
    value = eval(s)
    ns = []
    for v in value:
        if len(v) != 1:
            raise NotImplementedError(f"Unexpected value {v} in {s!r}.")
        k, v = list(v.items())[0]
        n = "-".join([keys[k], "x".join(map(str, v))])
        ns.append(n)
    return ",".join(ns)


def label(row):
    name = row["args_op_name"]
    inshape = process_shape(row["args_input_type_shape"])
    outshape = process_shape(row["args_output_type_shape"])
    side = row["side"][0]
    prov = row["args_provider"][:3]
    return f"[{side}{prov}]{name}({inshape})->{outshape}"


df = merge.copy()
df["side"] = df.apply(classify, axis=1)
df["label"] = df.apply(label, axis=1)
gr = (
    df[["label", "durbase", "duropti"]]
    .groupby("label")
    .agg({"durbase": [numpy.sum, len], "duropti": [sum, len]})
)
gr


################################
# Final plot
# ++++++++++


fig, ax = plt.subplots(1, 2, figsize=(14, gr.shape[0]), sharey=True)
gr[[("durbase", "sum"), ("duropti", "sum")]].plot.barh(ax=ax[0])
ax[0].set_title("Side by side duration")
gr[[("durbase", "len"), ("duropti", "len")]].plot.barh(ax=ax[1])
ax[1].set_title("Side by side count")
fig.tight_layout()
fig.savefig("plot_profiling_side_by_side.png")
