"""

.. _l-onnx-diff-example:

Compares the conversions of the same model with different options
=================================================================

The script compares two onnx models obtained with the same trained
scikit-learn models but converted with different options.

A model
+++++++
"""

from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx
from onnx_array_api.reference import compare_onnx_execution
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


data = load_iris()
X_train, X_test = train_test_split(data.data)
model = GaussianMixture()
model.fit(X_train)

#################################
# Conversion to onnx
# ++++++++++++++++++

onx = to_onnx(
    model, X_train[:1], options={id(model): {"score_samples": True}}, target_opset=12
)

print(onnx_simple_text_plot(onx))

##################################
# Conversion to onnx without ReduceLogSumExp
# ++++++++++++++++++++++++++++++++++++++++++

onx2 = to_onnx(
    model,
    X_train[:1],
    options={id(model): {"score_samples": True}},
    black_op={"ReduceLogSumExp"},
    target_opset=12,
)

print(onnx_simple_text_plot(onx2))


#############################################
# Differences
# +++++++++++
#
# Function :func:`onnx_array_api.reference.compare_onnx_execution`
# compares the intermediate results of two onnx models. Then it finds
# the best alignmet between the two models using an edit distance.

res1, res2, align, dc = compare_onnx_execution(onx, onx2, verbose=1)
print("------------")
text = dc.to_str(res1, res2, align)
print(text)

###############################
# See :ref:`l-long-output-compare_onnx_execution` for a better view.
# The display shows that ReduceSumSquare was replaced by Mul + ReduceSum,
# and ReduceLogSumExp by ReduceMax + Sub + Exp + Log + Add.
