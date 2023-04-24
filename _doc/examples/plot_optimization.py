"""

.. _l-onnx-array-onnxruntime-optimization:

Optimization with onnxruntime
=============================


Optimize a model with onnxruntime
+++++++++++++++++++++++++++++++++
"""
import os
from pprint import pprint
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from onnx import load
from onnx_array_api.ext_test_case import example_path
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_array_api.validation.diff import text_diff, html_diff
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnx_array_api.ext_test_case import measure_time
from onnx_array_api.ort.ort_optimizers import ort_optimized_model


filename = example_path("data/small.onnx")
optimized = filename + ".optimized.onnx"

if not os.path.exists(optimized):
    ort_optimized_model(filename, output=optimized)
print(optimized)

#############################
# Output comparison
# +++++++++++++++++

so = SessionOptions()
so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
img = numpy.random.random((1, 3, 112, 112)).astype(numpy.float32)

sess = InferenceSession(filename, so)
sess_opt = InferenceSession(optimized, so)
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: img})[0]
out_opt = sess_opt.run(None, {input_name: img})[0]
if out.shape != out_opt.shape:
    print("ERROR shape are different {out.shape} != {out_opt.shape}")
diff = numpy.abs(out - out_opt).max()
print(f"Differences: {diff}")

####################################
# Difference
# ++++++++++
#
# Unoptimized model.

with open(filename, "rb") as f:
    model = load(f)
print("first model to text...")
text1 = onnx_simple_text_plot(model, indent=False)
print(text1)

#####################################
# Optimized model.


with open(optimized, "rb") as f:
    model = load(f)
print("second model to text...")
text2 = onnx_simple_text_plot(model, indent=False)
print(text2)

########################################
# Differences

print("differences...")
print(text_diff(text1, text2))

#####################################
# HTML version.

print("html differences...")
output = html_diff(text1, text2)
with open("diff_html.html", "w", encoding="utf-8") as f:
    f.write(output)
print("done.")

#####################################
# Benchmark
# +++++++++

img = numpy.random.random((1, 3, 112, 112)).astype(numpy.float32)

t1 = measure_time(lambda: sess.run(None, {input_name: img}), repeat=25, number=25)
t1["name"] = "original"
print("Original model")
pprint(t1)

t2 = measure_time(lambda: sess_opt.run(None, {input_name: img}), repeat=25, number=25)
t2["name"] = "optimized"
print("Optimized")
pprint(t2)


############################
# Plots
# +++++


fig, ax = plt.subplots(1, 1, figsize=(12, 4))

df = DataFrame([t1, t2]).set_index("name")
print(df)

print(df["average"].values)
print((df["average"] - df["deviation"]).values)

ax.bar(df.index, df["average"].values, yerr=df["deviation"].values, capsize=6)
ax.set_title("Measure performance of optimized model\nlower is better")
plt.grid()
fig.savefig("plot_optimization.png")
