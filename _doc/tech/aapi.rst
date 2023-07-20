
Difficulty to implement an an Array API for ONNX
================================================

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

An :class:`EagerTensor <onnx_array_api.npx.npx_tensors.EagerTensor>`
must be used to represent any tensor.
This class defines the backend to use as well.
:class:`EagerNumpyTensor
<onnx_array_api.npx.npx_numpy_tensors.EagerNumpyTensor>`
for :epkg:`numpy`, :class:`EagerOrtTensor
<onnx_array_api.ort.ort_tensors.EagerOrtTensor>`
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

Eager or compilation
++++++++++++++++++++

Eager mode is what the Array API implies.
Every function is converted into an ONNX graph based
on its inputs without any knownledge of how these inputs
were obtained. This graph is then executed before going
to the next call of a function from the API.
The conversion of a machine learned model
into ONNX implies the gathering of all these operations
into a graph. It means using a mode that records all the function
calls to compile every tiny onnx graph into a unique graph.

Iterators and Reduction
+++++++++++++++++++++++

An efficient implementation of function
:func:`numpy.any` or :func:`numpy.all` returns
as soon as the result is known. :func:`numpy.all` is
false whenever the first false condition is met.
Same goes for :func:`numpy.any` which is true 
whenever the first true condition is met.
There is no such operator in ONNX (<= 20) because
it is unlikely to appear in a machine learned model.
However, it is highly used when two results are
compared in unit tests. The ONNX implementation is
not efficient due to that reason but it only impacts
the unit tests.

Types
+++++

:epkg:`onnx` supports more types than :epkg:`numpy` does.
It is not always easy to deal with bfloat16 or float8 types.
