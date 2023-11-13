
.. image:: https://github.com/sdpython/onnx-array-api/raw/main/_doc/_static/logo.png
    :width: 120

onnx-array-api: APIs to create ONNX Graphs
==========================================

.. image:: https://dev.azure.com/xavierdupre3/onnx-array-api/_apis/build/status/sdpython.onnx-array-api
    :target: https://dev.azure.com/xavierdupre3/onnx-array-api/

.. image:: https://badge.fury.io/py/onnx-array-api.svg
    :target: http://badge.fury.io/py/onnx-array-api

.. image:: http://img.shields.io/github/issues/sdpython/onnx-array-api.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/onnx-array-api/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/license/MIT/

.. image:: https://img.shields.io/github/repo-size/sdpython/onnx-array-api
    :target: https://github.com/sdpython/onnx-array-api/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/sdpython/onnx-array-api/branch/main/graph/badge.svg?token=Wb9ZGDta8J 
    :target: https://codecov.io/gh/sdpython/onnx-array-api

**onnx-array-api** implements APIs to create custom ONNX graphs.
The objective is to speed up the implementation of converter libraries.
The first one matches **numpy API**.
It gives the user the ability to convert functions written
following the numpy API to convert that function into ONNX as
well as to execute it.

.. code-block:: python

    import numpy as np
    from onnx_array_api.npx import absolute, jit_onnx
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

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

::

    [0.042]
    opset: domain='' version=18
    input: name='x0' type=dtype('float32') shape=['', '']
    input: name='x1' type=dtype('float32') shape=['', '']
    Sub(x0, x1) -> r__0
      Abs(r__0) -> r__1
        ReduceSum(r__1, keepdims=0) -> r__2
    output: name='r__2' type=dtype('float32') shape=None

It supports eager mode as well:

.. code-block:: python

    import numpy as np
    from onnx_array_api.npx import absolute, eager_onnx


    def l1_loss(x, y):
        err = absolute(x - y).sum()
        print(f"l1_loss={err.numpy()}")
        return err


    def l2_loss(x, y):
        err = ((x - y) ** 2).sum()
        print(f"l2_loss={err.numpy()}")
        return err


    def myloss(x, y):
        return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])


    eager_myloss = eager_onnx(myloss)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    res = eager_myloss(x, y)
    print(res)

::

    l1_loss=[0.04]
    l2_loss=[0.002]
    [0.042]

The second API or **Light API** tends to do every thing in one line.
The euclidean distance looks like the following:

::

    import numpy as np
    from onnx_array_api.light_api import start
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

    model = (
        start()
        .vin("X")
        .vin("Y")
        .bring("X", "Y")
        .Sub()
        .rename("dxy")
        .cst(np.array([2], dtype=np.int64), "two")
        .bring("dxy", "two")
        .Pow()
        .ReduceSum()
        .rename("Z")
        .vout()
        .to_onnx()
    )    

The library is released on
`pypi/onnx-array-api <https://pypi.org/project/onnx-array-api/>`_
and its documentation is published at
`APIs to create ONNX Graphs <https://sdpython.github.io/doc/onnx-array-api/dev/>`_.
