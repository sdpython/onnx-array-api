
Technical details
=================

Implementing the full array API is not always easy with :epkg:`onnx`.
Python is not strongly typed and many different types can be used
to represent a value. Argument *axis* can be an integer or a tuple
(see `min from Array API
<https://data-apis.org/array-api/2022.12/API_specification/
generated/array_api.min.html>`
for example). On the other side, `ReduceMin from ONNX
<https://onnx.ai/onnx/operators/onnx__ReduceMin.html>`_
is considered as a tensor.

Performance
+++++++++++

The Array API must work in eager mode and for every operation,
it generates an ONNX graph and executes it with a specific
backend. It can be :epkg:`numpy`, :epkg:`onnxruntime` or any other
backend. The generation of every graph takes a significant amount of time.
It must be avoided. These graphs are cached. But a graph can be reused
only if the inputs - by ONNX semantic - change. If a parameter change,
a new graph must be cached. Method :meth:`JitEager.make_key`
generates a unique key based on the input it receives,
the signature of the function to call. If the key is the same,
a cached onnx can be reused on the second call.

However, eager mode - use a small single onnx graph for every operation -
is not the most efficient one. At the same time, the design must allow
to merge every needed operation into a bigger graph.
Bigger graphs can be more easily optimized by the backend.

Input vs parameter
++++++++++++++++++

An input is a tensor or array, a parameter is any other type.
Following onnx semantic, an input is variable, a parameter is frozen
cannot be changed. It is a constant. A good design would be 
to considered any named input (`**kwargs`) a parameter and
any input (`*args`) a tensor. But the Array API does not follow that
design. Function `astype
<https://data-apis.org/array-api/2022.12/API_specification/
generated/array_api.astype.html>_`
takes two inputs. Operator `Cast
<https://onnx.ai/onnx/operators/onnx__Cast.html>_`
takes one input and a frozen parameter `to`.
And python allows `astype(x, dtype)` as well as `astype(x, dtype=dtype)`
unless the signature enforces one call over another type.
There may be ambiguities from time to time.
Beside, from onnx point of view, argument dtype should be named.

Tensor type
+++++++++++

An :class:`EagerTensor` must be used to represent any tensor.
This class defines the backend to use as well.
`EagerNumpyTensor` for :epkg:`numpy`, `EagerOrtTensor`
for :epkg:`onnxruntime`. Since the Array API is new, 
existing packages do not fully support the API if they support it
(:epkg:`scikit-learn`). Some numpy array may still be used.

Inplace
+++++++

ONNX has no notion of inplace computation. Therefore something
like `coefs[:, 1] = 1` is not valid unless some code is written
to create another tensor. The current design supports some of these
by storing every call to `__setitem__`. The user sees `coefs`
but the framework sees that `coefs` holds a reference to another
tensor. That's the one the framework needs to use. However, since
`__setitem__` is used for efficiency, it becomes less than efficient
with this design and should be avoided. This assumption may be true
when the backend is relying on CPU but not on GPU.
A function such as `empty
<https://data-apis.org/array-api/2022.12/API_specification/
generated/array_api.astype.html>`_ should be avoided as it
has to be followed by calls to `__setitem__`.
