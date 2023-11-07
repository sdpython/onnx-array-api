=============================================
Many ways to implement a custom graph in ONNX
=============================================

:epkg:`ONNX` defines a long list of operators used in machine learning models.
They are used to implement functions. This step is usually taken care of
by converting libraries: :epkg:`sklearn-onnx` for :epkg:`scikit-learn`,
:epkg:`torch.onnx` for :epkg:`pytorch`, :epkg:`tensorflow-onnx` for :epkg:`tensorflow`.
Both :epkg:`torch.onnx` and :epkg:`tensorflow-onnx` converts any function expressed
with the available function in those packages and that works because
there is usually no need to mix packages.
But in some occasions, there is a need to directly write functions with the
onnx syntax. :epkg:`scikit-learn` is implemented with :epkg:`numpy` and there
is no converter from numpy to onnx. Sometimes, it is needed to extend
an existing onnx models or to merge models coming from different packages.
Sometimes, they are just not available, only onnx is.
Let's see how it looks like a very simply example.

Euclidian distance
==================

For example, the well known Euclidian distance
:math:`f(X,Y)=\sum_{i=1}^n (X_i - Y_i)^2` can be expressed with numpy as follows:

.. code-block:: python

    import numpy as np

    def euclidan(X: np.array, Y: np.array) -> float:
        return ((X - Y) ** 2).sum()

The mathematical function must first be translated with :epkg:`ONNX Operators` or
primitives. It is usually easy because the primitives are very close to what
numpy defines. It can be expressed as (the syntax is just for illustration).

::

    import onnx

    onnx-def euclidian(X: onnx.TensorProto[FLOAT], X: onnx.TensorProto[FLOAT]) -> onnx.FLOAT:
        dxy = onnx.Sub(X, Y)
        sxy = onnx.Pow(dxy, 2)
        d = onnx.ReduceSum(sxy)
        return d

This example is short but does not work as it is.
The :epkg:`inner API` defined in :epkg:`onnx.helper` is quite verbose and
the true implementation would be the following.

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh


    def make_euclidean(
        input_names: tuple[str] = ("X", "Y"),
        output_name: str = "Z",
        elem_type: int = onnx.TensorProto.FLOAT,
        opset: int | None = None,
    ) -> onnx.ModelProto:
        if opset is None:
            opset = onnx.defs.onnx_opset_version()

            X = oh.make_tensor_value_info(input_names[0], elem_type, None)
            Y = oh.make_tensor_value_info(input_names[1], elem_type, None)
            Z = oh.make_tensor_value_info(output_name, elem_type, None)
            two = oh.make_tensor("two", onnx.TensorProto.INT64, [1], [2])
            n1 = oh.make_node("Sub", ["X", "Y"], ["dxy"])
            n2 = oh.make_node("Pow", ["dxy", "two"], ["dxy2"])
            n3 = oh.make_node("ReduceSum", ["dxy2"], [output_name])
            graph = oh.make_graph([n1, n2, n3], "euclidian", [X, Y], [Z], [two])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
            return model


    model = make_euclidean()
    print(model)

Since it is a second implementation of an existing function, it is necessary to
check the output is the same.

.. runpython::
    :showcode:

    import numpy as np
    from numpy.testing import assert_allclose
    from onnx.reference import ReferenceEvaluator
    from onnx_array_api.ext_test_case import ExtTestCase
    # This is the same function.
    from onnx_array_api.validation.docs import make_euclidean


    def test_make_euclidean():
        model = make_euclidean()

        ref = ReferenceEvaluator(model)
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        assert_allclose(expected, got, atol=1e-6)


    test_make_euclidean()

But the reference implementation in onnx is not the runtime used to deploy the model.
A second unit test must be added to check this one as well.

.. runpython::
    :showcode:

    import numpy as np
    from numpy.testing import assert_allclose
    from onnx_array_api.ext_test_case import ExtTestCase
    # This is the same function.
    from onnx_array_api.validation.docs import make_euclidean


    def test_make_euclidean_ort():
        from onnxruntime import InferenceSession
        model = make_euclidean()

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        assert_allclose(expected, got, atol=1e-6)


    try:
        test_make_euclidean_ort()
    except Exception as e:
        print(e)

The list of operators is constantly evolving: onnx is versioned.
The function may fail because the model says it is using a version
a runtime does not support. Let's change it.

.. runpython::
    :showcode:

    import numpy as np
    from numpy.testing import assert_allclose
    from onnx_array_api.ext_test_case import ExtTestCase
    # This is the same function.
    from onnx_array_api.validation.docs import make_euclidean


    def test_make_euclidean_ort():
        from onnxruntime import InferenceSession

        # opset=18: it uses the opset version 18, this number
        # is incremented at every minor release.
        model = make_euclidean(opset=18)

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)
        got = ref.run(None, {"X": X, "Y": Y})[0]
        assert_allclose(expected, got, atol=1e-6)


    test_make_euclidean_ort()

But the runtime must support many versions and the unit tests may look like
the following:

.. runpython::
    :showcode:

    import numpy as np
    from numpy.testing import assert_allclose
    import onnx.defs
    from onnx_array_api.ext_test_case import ExtTestCase
    # This is the same function.
    from onnx_array_api.validation.docs import make_euclidean


    def test_make_euclidean_ort():
        from onnxruntime import InferenceSession

        # opset=18: it uses the opset version 18, this number
        # is incremented at every minor release.
        X = np.random.rand(3, 4).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        expected = ((X - Y) ** 2).sum(keepdims=1)

        for opset in range(6, onnx.defs.onnx_opset_version()-1):
            model = make_euclidean(opset=opset)

            try:
                ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                got = ref.run(None, {"X": X, "Y": Y})[0]
            except Exception as e:
                print(f"fail opset={opset}", e)
                if opset < 18:
                    continue
                raise e
            assert_allclose(expected, got, atol=1e-6)


    test_make_euclidean_ort()

This work is quite long even for a simple function. For a longer one,
due to the verbosity of the inner API, it is quite difficult to write
the correct implementation on the first try. The unit test cannot be avoided.
The inner API is usually enough when the translation from python to onnx
does not happen often. When it is, almost every library implements
its own simplified way to create onnx graphs and because creating its own
API is not difficult, many times, the decision was made to create a new one
rather than using an existing one.

Existing API
============

Many existing options are available to write custom onnx graphs.
The development is usually driven by what they are used for. Each of them
may not fully support all your needs and it is not always easy to understand
the error messages they provide when something goes wrong.
It is better to understand its own need before choosing one.
Here are some of the questions which may need to be answered.

* ability to easily write loops and tests (control flow)
* ability to debug (eager mode)
* ability to use the same function to produce different implementations
  based on the same version
* ability to interact with other frameworks
* ability to merge existing onnx graph
* ability to describe an existing graph with this API
* ability to easily define constants
* ability to handle multiple domains
* ability to support local functions
* easy error messages
* is it actively maintained?

Use torch or tensorflow
+++++++++++++++++++++++

:epkg:`pytorch` offers the possibility to convert any function
implemented with pytorch function into onnx with :epkg:`torch.onnx`.
A couple of examples.

.. code-block:: python

    import torch
    import torch.nn


    class MyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x, bias=None):
            out = self.linear(x)
            out = out + bias
            return out

    model = MyModel()
    kwargs = {"bias": 3.}
    args = (torch.randn(2, 2, 2),)

    export_output = torch.onnx.dynamo_export(
        model,
        *args,
        **kwargs).save("my_simple_model.onnx")    

.. code-block:: python

    from typing import Dict, Tuple
    import torch
    import torch.onnx


    def func_with_nested_input_structure(
        x_dict: Dict[str, torch.Tensor],
        y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        if "a" in x_dict:
            x = x_dict["a"]
        elif "b" in x_dict:
            x = x_dict["b"]
        else:
            x = torch.randn(3)

        y1, (y2, y3) = y_tuple

        return x + y1 + y2 + y3

    x_dict = {"a": torch.tensor(1.)}
    y_tuple = (torch.tensor(2.), (torch.tensor(3.), torch.tensor(4.)))
    export_output = torch.onnx.dynamo_export(func_with_nested_input_structure, x_dict, y_tuple)

    print(export_output.adapt_torch_inputs_to_onnx(x_dict, y_tuple))

onnxscript
++++++++++

:epkg:`onnxscript` is used in `Torch Export to ONNX
<https://pytorch.org/tutorials//beginner/onnx/export_simple_model_to_onnx_tutorial.html>`_.
It converts python code to onnx code by analyzing the python code
(through :epkg:`ast`). The package makes it very easy to use loops and tests in onnx.
It is very close to onnx syntax. It is not easy to support multiple
implementation depending on the opset version required by the user.

Example taken from the documentation :

.. code-block:: python

    import onnx

    # We use ONNX opset 15 to define the function below.
    from onnxscript import FLOAT
    from onnxscript import opset15 as op
    from onnxscript import script


    # We use the script decorator to indicate that
    # this is meant to be translated to ONNX.
    @script()
    def onnx_hardmax(X, axis: int):
        """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""

        # The type annotation on X indicates that it is a float tensor of
        # unknown rank. The type annotation on axis indicates that it will
        # be treated as an int attribute in ONNX.
        #
        # Invoke ONNX opset 15 op ArgMax.
        # Use unnamed arguments for ONNX input parameters, and named
        # arguments for ONNX attribute parameters.
        argmax = op.ArgMax(X, axis=axis, keepdims=False)
        xshape = op.Shape(X, start=axis)
        # use the Constant operator to create constant tensors
        zero = op.Constant(value_ints=[0])
        depth = op.GatherElements(xshape, zero)
        empty_shape = op.Constant(value_ints=[0])
        depth = op.Reshape(depth, empty_shape)
        values = op.Constant(value_ints=[0, 1])
        cast_values = op.CastLike(values, X)
        return op.OneHot(argmax, depth, cast_values, axis=axis)


    # We use the script decorator to indicate that
    # this is meant to be translated to ONNX.
    @script()
    def sample_model(X: FLOAT[64, 128], Wt: FLOAT[128, 10], Bias: FLOAT[10]) -> FLOAT[64, 10]:
        matmul = op.MatMul(X, Wt) + Bias
        return onnx_hardmax(matmul, axis=1)


    # onnx_model is an in-memory ModelProto
    onnx_model = sample_model.to_model_proto()

    # Save the ONNX model at a given path
    onnx.save(onnx_model, "sample_model.onnx")

    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid!")

An Eager mode is available to debug what the code does.

.. code-block:: python

    import numpy as np

    v = np.array([[0, 1], [2, 3]], dtype=np.float32)
    result = Hardmax(v)

spox
++++

The syntax of :epkg:`spox` is similar but it does not use :epkg:`ast`.
Therefore, `loops and tests <https://spox.readthedocs.io/en/latest/guides/advanced.html#Control-flow>`_
are expressed in a very different way. The tricky part with it is to handle
the local context. A variable created in the main graph is known by any
of its subgraphs.

Example taken from the documentation :

.. code-block::

    import onnx

    from spox import argument, build, Tensor, Var
    # Import operators from the ai.onnx domain at version 17
    from spox.opset.ai.onnx import v17 as op

    def geometric_mean(x: Var, y: Var) -> Var:
        # use the standard Sqrt and Mul
        return op.sqrt(op.mul(x, y))

    # Create typed model inputs. Each tensor is of rank 1
    # and has the runtime-determined length 'N'.
    a = argument(Tensor(float, ('N',)))
    b = argument(Tensor(float, ('N',)))

    # Perform operations on `Var`s
    c = geometric_mean(a, b)

    # Build an `onnx.ModelProto` for the given inputs and outputs.
    model: onnx.ModelProto = build(inputs={'a': a, 'b': b}, outputs={'c': c})

The function can be tested with a mechanism called
`value propagation <https://spox.readthedocs.io/en/latest/guides/inference.html#Value-propagation>`_.

sklearn-onnx
++++++++++++

:epkg:`sklearn-onnx` also implements its own API to add custom graphs.
It was designed to shorten the time spent in reimplementing :epkg:`scikit-learn`
code into :epkg:`onnx` code. It can be used to implement a new converter
mapped a custom model as described in this example:
`Implement a new converter
<https://onnx.ai/sklearn-onnx/auto_tutorial/plot_icustom_converter.html>`_.
But it can also be used to build standalone models.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


    def make_euclidean_skl2onnx(
        input_names: tuple[str] = ("X", "Y"),
        output_name: str = "Z",
        elem_type: int = onnx.TensorProto.FLOAT,
        opset: int | None = None,
    ) -> onnx.ModelProto:
        if opset is None:
            opset = onnx.defs.onnx_opset_version()

        from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxPow, OnnxReduceSum

        dxy = OnnxSub(input_names[0], input_names[1], op_version=opset)
        dxy2 = OnnxPow(dxy, np.array([2], dtype=np.int64), op_version=opset)
        final = OnnxReduceSum(dxy2, op_version=opset, output_names=[output_name])

        np_type = oh.tensor_dtype_to_np_dtype(elem_type)
        dummy = np.empty([1], np_type)
        return final.to_onnx({"X": dummy, "Y": dummy})


    model = make_euclidean_skl2onnx()
    print(onnx_simple_text_plot(model))
    
onnxblocks
++++++++++

`onnxblocks <https://onnxruntime.ai/docs/api/python/on_device_training/training_artifacts.html#prepare-for-training>`_
was introduced in onnxruntime to define custom losses in order to train
a model with :epkg:`onnxruntime-training`. It is mostly used for this usage.

.. code-block:: python

    import onnxruntime.training.onnxblock as onnxblock
    from onnxruntime.training import artifacts

    # Define a custom loss block that takes in two inputs
    # and performs a weighted average of the losses from these
    # two inputs.
    class WeightedAverageLoss(onnxblock.Block):
        def __init__(self):
            self._loss1 = onnxblock.loss.MSELoss()
            self._loss2 = onnxblock.loss.MSELoss()
            self._w1 = onnxblock.blocks.Constant(0.4)
            self._w2 = onnxblock.blocks.Constant(0.6)
            self._add = onnxblock.blocks.Add()
            self._mul = onnxblock.blocks.Mul()

        def build(self, loss_input_name1, loss_input_name2):
            # The build method defines how the block should be stacked on top of
            # loss_input_name1 and loss_input_name2

            # Returns weighted average of the two losses
            return self._add(
                self._mul(self._w1(), self._loss1(loss_input_name1, target_name="target1")),
                self._mul(self._w2(), self._loss2(loss_input_name2, target_name="target2"))
            )

    my_custom_loss = WeightedAverageLoss()

    # Load the onnx model
    model_path = "model.onnx"
    base_model = onnx.load(model_path)

    # Define the parameters that need their gradient computed
    requires_grad = ["weight1", "bias1", "weight2", "bias2"]
    frozen_params = ["weight3", "bias3"]

    # Now, we can invoke generate_artifacts with this custom loss function
    artifacts.generate_artifacts(base_model, requires_grad = requires_grad, frozen_params = frozen_params,
                                loss = my_custom_loss, optimizer = artifacts.OptimType.AdamW)

    # Successful completion of the above call will generate 4 files in the current working directory,
    # one for each of the artifacts mentioned above (training_model.onnx, eval_model.onnx, checkpoint, op)

numpy API for onnx
++++++++++++++++++

See :ref:`l-numpy-api-onnx`. This API was introduced to create graphs
by using numpy API. If a function is defined only with numpy,
it should be possible to use the exact same code to create the
corresponding onnx graph. That's what this API tries to achieve.
It works with the exception of control flow. In that case, the function
produces different onnx graphs depending on the execution path.

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.npx import jit_onnx
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

    def l2_loss(x, y):
        return ((x - y) ** 2).sum(keepdims=1)

    jitted_myloss = jit_onnx(l2_loss)
    dummy = np.array([0], dtype=np.float32)

    # The function is executed. Only then a onnx graph is created.
    # One is created depending on the input type.
    jitted_myloss(dummy, dummy)

    # get_onnx only works if it was executed once or at least with
    # the same input type
    model = jitted_myloss.get_onnx()
    print(onnx_simple_text_plot(model))

Light API
+++++++++

See :ref:`l-light-api`. This API was created to be able to write an onnx graph
in one instruction. It is inspired from the :epkg:`reverse Polish notation`.
There is no eager mode.

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
