import numpy as np
from onnx.reference.op_run import OpRun


def sigmoid(x):  # type: ignore
    if x > 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


class QuickGelu(OpRun):
    op_domain = "com.microsoft"

    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.vf = np.vectorize(sigmoid)

    def _run(self, X, alpha=1.0):
        if len(X.shape) == 0:
            return ((X * sigmoid(X * alpha)).astype(X.dtype),)
        if X.size == 0:
            return (X,)
        return ((X * self.vf(X * alpha)).astype(X.dtype),)
