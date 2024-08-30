"""
First examples with onnxruntime
===============================

Example :ref:`l-onnx-array-first-api-example` defines a custom
loss and then executes it with class
:class:`onnx.reference.ReferenceEvaluator`.
Next example replaces it with :epkg:`onnxruntime`.

Example
+++++++
"""

import numpy as np

from onnx_array_api.npx import absolute, jit_onnx
from onnx_array_api.ort.ort_tensors import JitOrtTensor, OrtTensor


def l1_loss(x, y):
    return absolute(x - y).sum()


def l2_loss(x, y):
    return ((x - y) ** 2).sum()


def myloss(x, y):
    l1 = l1_loss(x[:, 0], y[:, 0])
    l2 = l2_loss(x[:, 1], y[:, 1])
    return l1 + l2


ort_myloss = jit_onnx(myloss, JitOrtTensor, target_opsets={"": 17}, ir_version=8)

x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

xort = OrtTensor.from_array(x)
yort = OrtTensor.from_array(y)

res = ort_myloss(xort, yort)
print(res.numpy())

###############################
# Profiling
# +++++++++
from onnx_array_api.profiling import profile, profile2graph

x = np.random.randn(10000, 2).astype(np.float32)
y = np.random.randn(10000, 2).astype(np.float32)
xort = OrtTensor.from_array(x)
yort = OrtTensor.from_array(y)


def loop_ort(n):
    for _ in range(n):
        ort_myloss(xort, yort)


def loop_numpy(n):
    for _ in range(n):
        myloss(x, y)


def loop(n=1000):
    loop_numpy(n)
    loop_ort(n)


ps = profile(loop)[0]
root, nodes = profile2graph(ps, clean_text=lambda x: x.split("/")[-1])
text = root.to_text()
print(text)

##############################
# Benchmark
# +++++++++

from pandas import DataFrame
from tqdm import tqdm

from onnx_array_api.ext_test_case import measure_time

data = []
for n in tqdm([1, 10, 100, 1000, 10000, 100000]):
    x = np.random.randn(n, 2).astype(np.float32)
    y = np.random.randn(n, 2).astype(np.float32)

    obs = measure_time(lambda x=x, y=y: myloss(x, y))
    obs["name"] = "numpy"
    obs["n"] = n
    data.append(obs)

    xort = OrtTensor.from_array(x)
    yort = OrtTensor.from_array(y)
    obs = measure_time(lambda xort=xort, yort=yort: ort_myloss(xort, yort))
    obs["name"] = "ort"
    obs["n"] = n
    data.append(obs)

df = DataFrame(data)
piv = df.pivot(index="n", columns="name", values="average")
piv

############################
# Plots
# +++++

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
piv.plot(
    title="Comparison between numpy and onnxruntime", logx=True, logy=True, ax=ax[0]
)
piv["ort/numpy"] = piv["ort"] / piv["numpy"]
piv["ort/numpy"].plot(title="Ratio ort/numpy", logx=True, ax=ax[1])
fig.savefig("plot_onnxruntime.png")
