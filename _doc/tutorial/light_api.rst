.. _l-light-api:

==========================================
Light API for ONNX: everything in one line
==========================================

It is inspired from the :epkg:`reverse Polish notation`.
Following example implements the euclidean distance.
This API tries to keep it simple and intuitive to short functions.

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

There are two kinds of methods, the graph methods, playing with the graph structure,
and the methods for operators starting with an upper letter.

Graph methods
=============

Any graph must start with function :func:`start <onnx_array_api.light_api.start>`.
It is usually following by `vin` to add an input.

* bring (:meth:`cst <onnx_array_api.light_api.Var.bring>`, :meth:`cst <onnx_array_api.light_api.Vars.bring>`):
  assembles multiple results into a set before calling an operator taking mulitple inputs,
* cst (:meth:`cst <onnx_array_api.light_api.Var.cst>`, :meth:`cst <onnx_array_api.light_api.Vars.cst>`):
  adds a constant tensor to the graph,
* rename  (:meth:`cst <onnx_array_api.light_api.Var.rename>`, :meth:`cst <onnx_array_api.light_api.Vars.rename>`):
  renames or give a name to a variable in order to call it later.
* vin (:meth:`cst <onnx_array_api.light_api.Var.vin>`, :meth:`cst <onnx_array_api.light_api.Vars.vin>`):
  adds an input to the graph,
* vout (:meth:`cst <onnx_array_api.light_api.Var.vout>`, :meth:`cst <onnx_array_api.light_api.Vars.vout>`):
  declares an existing result as an output.

These methods are implemented in class :class:`onnx_array_api.light_api.var.BaseVar`

Operator methods
================

They are described in :epkg:`ONNX Operators` and redefined in a stable API
so that the definition should not change depending on this opset.
:class:`onnx_array_api.light_api.Var` defines all operators taking only one input.
:class:`onnx_array_api.light_api.Vars` defines all other operators.

Numpy methods
=============

Numpy users expect methods such as `reshape`, property `shape` or
operator `+` to be available as well and that the case. They are
defined in class :class:`Var <onnx_array_api.light_api.Var>` or
:class:`Vars <onnx_array_api.light_api.Vars>` depending on the number of
inputs they require. Their name starts with a lower letter.
