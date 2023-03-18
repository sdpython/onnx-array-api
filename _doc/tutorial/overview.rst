.. _l-numpy-api-onnx:

==================
Numpy API for ONNX
==================

Many users have difficulties to write onnx graphs.
Many packages tries to symplify it either by implementing
their own api very close to onnx operators
(`sklearn-onnx <http://onnx.ai/sklearn-onnx/>`_,
`tf2onnx <https://github.com/onnx/tensorflow-onnx>`_,
`spox <https://spox.readthedocs.io/en/latest/>`_,
`onnx-script <https://github.com/microsoft/onnx-script>`_).
This contribution tries a different approach by implementing
a numpy API for ONNX. It does not cover everything numpy
or ONNX can do but it can easily be used to define
loss functions for example without knowing too much about ONNX.

.. note:: control flow

    The first version (onnx==1.15) does not support control flow yet (test and loops).
    There is no easy syntax for that yet and the main challenge is to deal with local context.

Overview
========

.. toctree::

    auto_examples/plot_first_example

Example
+++++++

Let's define the L1 loss computed from two vectors:

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.npx import jit_onnx
    from onnx_array_api.npx import absolute

    # The function looks like a numpy function.
    def l1_loss(x, y):
        return absolute(x - y).sum()

    # The function needs to be converted into ONNX with function jit_onnx.
    # jitted_l1_loss is a wrapper. It intercepts all calls to l1_loss.
    # When it happens, it checks the input types and creates the
    # corresponding ONNX graph.
    jitted_l1_loss = jit_onnx(l1_loss)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    # First execution and conversion to ONNX.
    # The wrapper caches the created onnx graph.
    # It reuses it if the input types and the number of dimension are the same.
    # It creates a new one otherwise and keep the old one.
    res = jitted_l1_loss(x, y)
    print(res)

    # The ONNX graph can be accessed the following way.
    print(jitted_l1_loss.get_onnx())


We can also define a more complex loss by computing L1 loss on
the first column and L2 loss on the seconde one.

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.npx import jit_onnx
    from onnx_array_api.npx import absolute

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

onnxruntime
+++++++++++

The backend is the class :class:`ReferenceEvalutor` by default but it could
be replaced by onnxruntime. The backend is not implemented in onnx package
but is added to the following example. The current implementation
is available with class `OrtTensor
<https://github.com/onnx/onnx/tree/main/onnx/test/npx_test.py#L100>`_.

.. runpython::
    :showcode:

    from typing import Any, Callable, List, Optional, Tuple, Union

    import numpy as np
    from onnxruntime import InferenceSession, RunOptions, get_available_providers
    from onnxruntime.capi._pybind_state import OrtDevice as C_OrtDevice
    from onnxruntime.capi._pybind_state import OrtMemType
    from onnxruntime.capi._pybind_state import (
        OrtValue as C_OrtValue,
    )
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    from onnx import ModelProto, TensorProto
    from onnx.defs import onnx_opset_version
    from onnx_array_api.npx.npx_tensors import EagerTensor, JitTensor
    from onnx_array_api.npx.npx_types import TensorType

    import numpy as np
    from onnx_array_api.npx import jit_onnx
    from onnx_array_api.npx import absolute


    def l1_loss(x, y):
        return absolute(x - y).sum()

    def l2_loss(x, y):
        return ((x - y) ** 2).sum()

    def myloss(x, y):
        l1 = l1_loss(x[:, 0], y[:, 0])
        l2 = l2_loss(x[:, 1], y[:, 1])
        return l1 + l2 

    ort_myloss = jit_onnx(myloss, BackendOrtTensor, target_opsets={"": 17}, ir_version=8)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    xort = OrtTensor.from_array(x)
    yort = OrtTensor.from_array(y)

    res = ort_myloss(xort, yort)
    print(res.numpy())

This backend do not support numpy array but only the 
class OrtValue which represents a tensor in onnxruntime.
This value can be easily created from a numpy array and could
be placed on CPU or CUDA if it is available.

Eager mode
++++++++++

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.npx import eager_onnx
    from onnx_array_api.npx import absolute

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

    # Eager mode is enabled by function :func:`eager_onnx`.
    # It intercepts all calls to `my_loss`. On the first call,
    # it replaces a numpy array by a tensor corresponding to the
    # selected runtime, here numpy as well through :class:`EagerNumpyTensor`.
    eager_myloss = eager_onnx(myloss)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    # First execution and conversion to ONNX.
    # The wrapper caches many Onnx graphs corresponding to
    # simple opeator, (+, -, /, *, ..), reduce functions,
    # any other function from the API.
    # It reuses it if the input types and the number of dimension are the same.
    # It creates a new one otherwise and keep the old ones.
    res = eager_myloss(x, y)
    print(res)
