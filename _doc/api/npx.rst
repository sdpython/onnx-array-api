.. _l-npx:

Numpy API for ONNX
==================

See `Python array API standard <https://data-apis.org/array-api/latest/index.html>`_.

.. contents::
    :local:

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

.. autofunction:: onnx_array_api.npx.npx_var.Var

Cst, Input
++++++++++

.. autofunction:: onnx_array_api.npx.npx_var.Cst

.. autofunction:: onnx_array_api.npx.npx_var.Input

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

.. autofunction:: onnx_array_api.npx.npx_jit_eager.JitEager

.. autofunction:: onnx_array_api.npx.npx_jit_eager.JitOnnx

Tensors
+++++++

.. autofunction:: onnx_array_api.npx.npx_tensors.NumpyTensor

Annotations
+++++++++++

.. autofunction:: onnx_array_api.npx.npx_types.ElemType

.. autofunction:: onnx_array_api.npx.npx_types.ParType

.. autofunction:: onnx_array_api.npx.npx_types.OptParType

.. autofunction:: onnx_array_api.npx.npx_types.TensorType

.. autofunction:: onnx_array_api.npx.npx_types.SequenceType

.. autofunction:: onnx_array_api.npx.npx_types.TupleType

.. autofunction:: onnx_array_api.npx.npx_types.Bool

.. autofunction:: onnx_array_api.npx.npx_types.BFloat16

.. autofunction:: onnx_array_api.npx.npx_types.Float16

.. autofunction:: onnx_array_api.npx.npx_types.Float32

.. autofunction:: onnx_array_api.npx.npx_types.Float64

.. autofunction:: onnx_array_api.npx.npx_types.Int8

.. autofunction:: onnx_array_api.npx.npx_types.Int16

.. autofunction:: onnx_array_api.npx.npx_types.Int32

.. autofunction:: onnx_array_api.npx.npx_types.Int64

.. autofunction:: onnx_array_api.npx.npx_types.UInt8

.. autofunction:: onnx_array_api.npx.npx_types.UInt16

.. autofunction:: onnx_array_api.npx.npx_types.UInt32

.. autofunction:: onnx_array_api.npx.npx_types.UInt64
