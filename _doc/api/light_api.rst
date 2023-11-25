========================
onnx_array_api.light_api
========================


Main API
========

start
+++++

.. autofunction:: onnx_array_api.light_api.start

translate
+++++++++

.. autofunction:: onnx_array_api.light_api.translate

Classes for the Light API
=========================

domain
++++++

..autofunction:: onnx_array_api.light_api.domain

BaseVar
+++++++

.. autoclass:: onnx_array_api.light_api.var.BaseVar
    :members:

OnnxGraph
+++++++++

.. autoclass:: onnx_array_api.light_api.OnnxGraph
    :members:

ProtoType
+++++++++

.. autoclass:: onnx_array_api.light_api.model.ProtoType
    :members:

SubDomain
+++++++++

.. autoclass:: onnx_array_api.light_api.var.SubDomain
    :members:

Var
+++

.. autoclass:: onnx_array_api.light_api.Var
    :members:
    :inherited-members:

Vars
++++

.. autoclass:: onnx_array_api.light_api.Vars
    :members:
    :inherited-members:

Classes for the Translater
==========================

BaseEmitter
+++++++++++

.. autoclass:: onnx_array_api.light_api.emitter.BaseEmitter
    :members:

Emitter
+++++++

.. autoclass:: onnx_array_api.light_api.emitter.Emitter
    :members:

EventType
+++++++++

.. autoclass:: onnx_array_api.light_api.translate.EventType
    :members:

InnerEmitter
++++++++++++

.. autoclass:: onnx_array_api.light_api.inner_emitter.InnerEmitter
    :members:

Translater
++++++++++

.. autoclass:: onnx_array_api.light_api.translate.Translater
    :members:

Available operators
===================

One input
+++++++++

.. autoclass:: onnx_array_api.light_api._op_var.OpsVar
    :members:

Two inputs or more
++++++++++++++++++

.. autoclass:: onnx_array_api.light_api._op_vars.OpsVars
    :members:



