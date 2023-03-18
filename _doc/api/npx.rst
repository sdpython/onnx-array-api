.. _l-api-npx:

npx
===

Functions
+++++++++

.. autofunction:: onnx_array_api.npx.npx_functions.abs

.. autofunction:: onnx_array_api.npx.npx_functions.absolute

.. autofunction:: onnx_array_api.npx.npx_functions.arccos

.. autofunction:: onnx_array_api.npx.npx_functions.arccosh

.. autofunction:: onnx_array_api.npx.npx_functions.amax

.. autofunction:: onnx_array_api.npx.npx_functions.amin

.. autofunction:: onnx_array_api.npx.npx_functions.arange

.. autofunction:: onnx_array_api.npx.npx_functions.argmax

.. autofunction:: onnx_array_api.npx.npx_functions.argmin

.. autofunction:: onnx_array_api.npx.npx_functions.arcsin

.. autofunction:: onnx_array_api.npx.npx_functions.arcsinh

.. autofunction:: onnx_array_api.npx.npx_functions.arctan

.. autofunction:: onnx_array_api.npx.npx_functions.arctanh

.. autofunction:: onnx_array_api.npx.npx_functions.cdist

.. autofunction:: onnx_array_api.npx.npx_functions.ceil

.. autofunction:: onnx_array_api.npx.npx_functions.clip

.. autofunction:: onnx_array_api.npx.npx_functions.compress

.. autofunction:: onnx_array_api.npx.npx_functions.compute

.. autofunction:: onnx_array_api.npx.npx_functions.concat

.. autofunction:: onnx_array_api.npx.npx_functions.cos

.. autofunction:: onnx_array_api.npx.npx_functions.cosh

.. autofunction:: onnx_array_api.npx.npx_functions.cumsum

.. autofunction:: onnx_array_api.npx.npx_functions.det

.. autofunction:: onnx_array_api.npx.npx_functions.dot

.. autofunction:: onnx_array_api.npx.npx_functions.einsum

.. autofunction:: onnx_array_api.npx.npx_functions.erf

.. autofunction:: onnx_array_api.npx.npx_functions.exp

.. autofunction:: onnx_array_api.npx.npx_functions.expand_dims

.. autofunction:: onnx_array_api.npx.npx_functions.expit

.. autofunction:: onnx_array_api.npx.npx_functions.floor

.. autofunction:: onnx_array_api.npx.npx_functions.hstack

.. autofunction:: onnx_array_api.npx.npx_functions.copy

.. autofunction:: onnx_array_api.npx.npx_functions.identity

.. autofunction:: onnx_array_api.npx.npx_functions.isnan

.. autofunction:: onnx_array_api.npx.npx_functions.log

.. autofunction:: onnx_array_api.npx.npx_functions.log1p

.. autofunction:: onnx_array_api.npx.npx_functions.matmul

.. autofunction:: onnx_array_api.npx.npx_functions.pad

.. autofunction:: onnx_array_api.npx.npx_functions.reciprocal

.. autofunction:: onnx_array_api.npx.npx_functions.relu

.. autofunction:: onnx_array_api.npx.npx_functions.round

.. autofunction:: onnx_array_api.npx.npx_functions.sigmoid

.. autofunction:: onnx_array_api.npx.npx_functions.sign

.. autofunction:: onnx_array_api.npx.npx_functions.sin

.. autofunction:: onnx_array_api.npx.npx_functions.sinh

.. autofunction:: onnx_array_api.npx.npx_functions.squeeze

.. autofunction:: onnx_array_api.npx.npx_functions.tan

.. autofunction:: onnx_array_api.npx.npx_functions.tanh

.. autofunction:: onnx_array_api.npx.npx_functions.topk

.. autofunction:: onnx_array_api.npx.npx_functions.transpose

.. autofunction:: onnx_array_api.npx.npx_functions.unsqueeze

.. autofunction:: onnx_array_api.npx.npx_functions.vstack

.. autofunction:: onnx_array_api.npx.npx_functions.where

Var
+++

.. autoclass:: onnx_array_api.npx.npx_var.Var
    :class:

Cst, Input
++++++++++

.. autoclass:: onnx_array_api.npx.npx_var.Cst
    :class:

.. autoclass:: onnx_array_api.npx.npx_var.Input
    :class:

API
+++

.. autofunction:: onnx_array_api.npx.npx_core_api.var

.. autofunction:: onnx_array_api.npx.npx_core_api.cst

.. autofunction:: onnx_array_api.npx.npx_jit_eager.jit_eager

.. autofunction:: onnx_array_api.npx.npx_jit_eager.jit_onnx

.. autofunction:: onnx_array_api.npx.npx_core_api.make_tuple

.. autofunction:: onnx_array_api.npx.npx_core_api.tuple_var

.. autofunction:: onnx_array_api.npx.npx_core_api.npxapi_inline

.. autofunction:: onnx_array_api.npx.npx_core_api.npxapi_function

JIT, Eager
++++++++++

.. autoclass:: onnx_array_api.npx.npx_jit_eager.JitEager
    :class:

.. autoclass:: onnx_array_api.npx.npx_jit_eager.JitOnnx
    :class:

Tensors
+++++++

.. autoclass:: onnx_array_api.npx.npx_tensors.NumpyTensor
    :class:

Annotations
+++++++++++

.. autoclass:: onnx_array_api.npx.npx_types.ElemType
    :members:

.. autoclass:: onnx_array_api.npx.npx_types.ParType
    :members:

.. autoclass:: onnx_array_api.npx.npx_types.OptParType
    :members:

.. autoclass:: onnx_array_api.npx.npx_types.TensorType
    :members:

.. autoclass:: onnx_array_api.npx.npx_types.SequenceType
    :members:

.. autoclass:: onnx_array_api.npx.npx_types.TupleType
    :members:

Shortcuts
+++++++++

.. autoclass:: onnx_array_api.npx.npx_types.Bool

.. autoclass:: onnx_array_api.npx.npx_types.BFloat16

.. autoclass:: onnx_array_api.npx.npx_types.Float16

.. autoclass:: onnx_array_api.npx.npx_types.Float32

.. autoclass:: onnx_array_api.npx.npx_types.Float64

.. autoclass:: onnx_array_api.npx.npx_types.Int8

.. autoclass:: onnx_array_api.npx.npx_types.Int16

.. autoclass:: onnx_array_api.npx.npx_types.Int32

.. autoclass:: onnx_array_api.npx.npx_types.Int64

.. autoclass:: onnx_array_api.npx.npx_types.UInt8

.. autoclass:: onnx_array_api.npx.npx_types.UInt16

.. autoclass:: onnx_array_api.npx.npx_types.UInt32

.. autoclass:: onnx_array_api.npx.npx_types.UInt64
