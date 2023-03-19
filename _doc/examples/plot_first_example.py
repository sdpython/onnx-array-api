"""

.. _l-onnx-array-first-api-example:

First examples with onnx-array-api
==================================

This demonstrates an easy case with :epkg:`onnx-array-api`.
It shows how a function can be easily converted into
ONNX.

A loss function from numpy to ONNX
++++++++++++++++++++++++++++++++++

The first example takes a loss function and converts it into ONNX.
"""

import numpy as np

from onnx_array_api.npx import absolute, jit_onnx
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


################################
# The function looks like a numpy function.
def l1_loss(x, y):
    return absolute(x - y).sum()


################################
# The function needs to be converted into ONNX with function jit_onnx.
# jitted_l1_loss is a wrapper. It intercepts all calls to l1_loss.
# When it happens, it checks the input types and creates the
# corresponding ONNX graph.
jitted_l1_loss = jit_onnx(l1_loss)

################################
# First execution and conversion to ONNX.
# The wrapper caches the created onnx graph.
# It reuses it if the input types and the number of dimension are the same.
# It creates a new one otherwise and keep the old one.

x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

res = jitted_l1_loss(x, y)
print(res)

####################################
# The ONNX graph can be accessed the following way.
print(onnx_simple_text_plot(jitted_l1_loss.get_onnx()))

################################
# We can also define a more complex loss by computing L1 loss on
# the first column and L2 loss on the seconde one.


def l1_loss(x, y):
    return absolute(x - y).sum()


def l2_loss(x, y):
    return ((x - y) ** 2).sum()


def myloss(x, y):
    return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])


jitted_myloss = jit_onnx(myloss)

x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

res = jitted_myloss(x, y)
print(res)

print(onnx_simple_text_plot(jitted_myloss.get_onnx()))

############################
# Eager mode
# ++++++++++

import numpy as np

from onnx_array_api.npx import absolute, eager_onnx


def l1_loss(x, y):
    err = absolute(x - y).sum()
    # err is a type inheriting from :class:`EagerTensor`.
    # It needs to be converted to numpy first before any display.
    print(f"l1_loss={err.numpy()}")
    return err


def l2_loss(x, y):
    err = ((x - y) ** 2).sum()
    print(f"l2_loss={err.numpy()}")
    return err


def myloss(x, y):
    return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])


#################################
# Eager mode is enabled by function :func:`eager_onnx`.
# It intercepts all calls to `my_loss`. On the first call,
# it replaces a numpy array by a tensor corresponding to the
# selected runtime, here numpy as well through :class:`EagerNumpyTensor`.
eager_myloss = eager_onnx(myloss)

x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

#################################
# First execution and conversion to ONNX.
# The wrapper caches many Onnx graphs corresponding to
# simple opeator, (`+`, `-`, `/`, `*`, ...), reduce functions,
# any other function from the API.
# It reuses it if the input types and the number of dimension are the same.
# It creates a new one otherwise and keep the old ones.
res = eager_myloss(x, y)
print(res)

################################
# There is no ONNX graph to show. Every operation
# is converted into small ONNX graphs.
