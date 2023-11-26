
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

.. toctree::
    :maxdepth: 1
    :caption: Contents

    tutorial/index
    api/index
    tech/index
    auto_examples/index

.. toctree::
    :maxdepth: 1
    :caption: More

    CHANGELOGS
    license

Numpy API
+++++++++

Sources available on
`github/onnx-array-api <https://github.com/sdpython/onnx-array-api>`_.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning, FutureWarning
    :process:

    import numpy as np  # A
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

.. gdot::
    :script: DOT-SECTION
    :process:

    # index
    import numpy as np
    from onnx_array_api.npx import absolute, jit_onnx
    from onnx_array_api.plotting.dot_plot import to_dot


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
    print(to_dot(jitted_myloss.get_onnx()))

Light API
+++++++++

.. runpython::
    :showcode:

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

    print(onnx_simple_text_plot(model))


Older versions
++++++++++++++

* `0.1.3 <../v0.1.3/index.html>`_
* `0.1.2 <../v0.1.2/index.html>`_
