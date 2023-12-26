.. _l-graph-api:

=================================
GraphBuilder: common API for ONNX
=================================

This is a very common way to build ONNX graph. There are some
annoying steps while building an ONNX graph. The first one is to
give unique names to every intermediate result in the graph. The second
is the conversion from numpy arrays to onnx tensors. A *graph builder*,
here implemented by class
:class:`GraphBuilder <onnx_array_api.graph_api.GraphBuilder>`
usually makes these two frequent tasks easier.

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.graph_api  import GraphBuilder
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

    g = GraphBuilder()
    g.make_tensor_input("X", np.float32, (None, None))
    g.make_tensor_input("Y", np.float32, (None, None))
    r1 = g.make_node("Sub", ["X", "Y"])  # the name given to the output is given by the class,
                                         # it ensures the name is unique
    init = g.make_initializer(np.array([2], dtype=np.int64))  # the class automatically
                                                              # converts the array to a tensor
    r2 = g.make_node("Pow", [r1, init])
    g.make_node("ReduceSum", [r2], outputs=["Z"])  # the output name is given because
                                                   # the user wants to choose the name
    g.make_tensor_output("Z", np.float32, (None, None))

    onx = g.to_onnx()  # final conversion to onnx

    print(onnx_simple_text_plot(onx))

A more simple versions of the same code to produce the same graph.

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.graph_api  import GraphBuilder
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

    g = GraphBuilder()
    g.make_tensor_input("X", np.float32, (None, None))
    g.make_tensor_input("Y", np.float32, (None, None))
    r1 = g.op.Sub("X", "Y")  # the method name indicates which operator to use,
                             # this can be used when there is no ambiguity about the
                             # number of outputs
    r2 = g.op.Pow(r1, np.array([2], dtype=np.int64))
    g.op.ReduceSum(r2, outputs=["Z"])  # the still wants the user to specify the name
    g.make_tensor_output("Z", np.float32, (None, None))
    
    onx = g.to_onnx()

    print(onnx_simple_text_plot(onx))
