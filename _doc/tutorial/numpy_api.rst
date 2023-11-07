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

You read :ref:`l-array-api-painpoint` as well.

Overview
========

.. toctree::

    ../auto_examples/plot_first_example
    ../auto_examples/plot_onnxruntime
