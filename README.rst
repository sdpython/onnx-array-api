
.. image:: https://github.com/sdpython/onnx-array-api/raw/main/_doc/_static/logo.png
    :width: 120

onnx-array-api: (Numpy) Array API for ONNX
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
    :target: http://opensource.org/licenses/MIT

.. image:: https://img.shields.io/github/repo-size/sdpython/onnx-array-api
    :target: https://github.com/sdpython/onnx-array-api/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

**onnx-array-api** implements a numpy API for ONNX.
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

The library is released on
`pypi/onnx-array-api <https://pypi.org/project/onnx-array-api/>`_
and its documentation is published at
`onnx-array-api: (Numpy) Array API for ONNX
<http://www.xavierdupre.fr/app/onnx-array-api/helpsphinx/index.html>`_.
