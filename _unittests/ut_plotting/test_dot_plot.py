import os
import unittest

import numpy
from onnx import TensorProto, load
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from skl2onnx import to_onnx
from skl2onnx.algebra.onnx_ops import (
    OnnxAdd,
    OnnxGreater,
    OnnxIf,
    OnnxLeakyRelu,
    OnnxReduceSum,
    OnnxSub,
)
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from onnx_array_api.ext_test_case import ExtTestCase, ignore_warnings
from onnx_array_api.plotting.dot_plot import to_dot

TARGET_OPSET = 18


class TestDotPlot(ExtTestCase):
    def test_onnx_text_plot_tree_reg(self):
        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeRegressor(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        dot = to_dot(onx)
        self.assertIn("X -> TreeEnsembleRegressor;", dot)

    def test_onnx_text_plot_tree_cls(self):
        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeClassifier(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        dot = to_dot(onx)
        self.assertIn("X -> TreeEnsembleClassifier;", dot)

    @ignore_warnings((UserWarning, FutureWarning))
    def test_to_dot_kmeans(self):
        x = numpy.random.randn(10, 3)
        model = KMeans(3)
        model.fit(x)
        onx = to_onnx(model, x.astype(numpy.float32), target_opset=15)
        dot = to_dot(onx)
        self.assertIn("Sq_Sqrt -> scores;", dot)

    def test_to_dot_knnr(self):
        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = RadiusNeighborsRegressor(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32), target_opset=15)
        dot = to_dot(onx)
        self.assertIn("Di_Div -> Di_C0;", dot)

    def test_to_dot_leaky(self):
        x = OnnxLeakyRelu("X", alpha=0.5, op_version=15, output_names=["Y"])
        onx = x.to_onnx(
            {"X": FloatTensorType()}, outputs={"Y": FloatTensorType()}, target_opset=15
        )
        dot = to_dot(onx)
        self.assertIn("Le_LeakyRelu -> Y;", dot)

    def test_to_dot_if(self):
        opv = TARGET_OPSET
        x1 = numpy.array([[0, 3], [7, 0]], dtype=numpy.float32)
        x2 = numpy.array([[1, 0], [2, 0]], dtype=numpy.float32)

        node = OnnxAdd("x1", "x2", output_names=["absxythen"], op_version=opv)
        then_body = node.to_onnx(
            {"x1": x1, "x2": x2},
            target_opset=opv,
            outputs=[("absxythen", FloatTensorType())],
        )
        node = OnnxSub("x1", "x2", output_names=["absxyelse"], op_version=opv)
        else_body = node.to_onnx(
            {"x1": x1, "x2": x2},
            target_opset=opv,
            outputs=[("absxyelse", FloatTensorType())],
        )
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum("x1", op_version=opv),
            OnnxReduceSum("x2", op_version=opv),
            op_version=opv,
        )
        ifnode = OnnxIf(
            cond,
            then_branch=then_body.graph,
            else_branch=else_body.graph,
            op_version=opv,
            output_names=["y"],
        )
        model_def = ifnode.to_onnx(
            {"x1": x1, "x2": x2}, target_opset=opv, outputs=[("y", FloatTensorType())]
        )
        dot = to_dot(model_def)
        self.assertIn("If_If -> y;", dot)

    def test_to_dot_if_recursive(self):
        opv = TARGET_OPSET
        x1 = numpy.array([[0, 3], [7, 0]], dtype=numpy.float32)
        x2 = numpy.array([[1, 0], [2, 0]], dtype=numpy.float32)

        node = OnnxAdd("x1", "x2", output_names=["absxythen"], op_version=opv)
        then_body = node.to_onnx(
            {"x1": x1, "x2": x2},
            target_opset=opv,
            outputs=[("absxythen", FloatTensorType())],
        )
        node = OnnxSub("x1", "x2", output_names=["absxyelse"], op_version=opv)
        else_body = node.to_onnx(
            {"x1": x1, "x2": x2},
            target_opset=opv,
            outputs=[("absxyelse", FloatTensorType())],
        )
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum("x1", op_version=opv),
            OnnxReduceSum("x2", op_version=opv),
            op_version=opv,
        )
        ifnode = OnnxIf(
            cond,
            then_branch=then_body.graph,
            else_branch=else_body.graph,
            op_version=opv,
            output_names=["y"],
        )
        model_def = ifnode.to_onnx(
            {"x1": x1, "x2": x2}, target_opset=opv, outputs=[("y", FloatTensorType())]
        )
        dot = to_dot(model_def, recursive=True)
        self.assertIn("If_If -> y;", dot)

    @ignore_warnings((UserWarning, FutureWarning))
    def test_to_dot_kmeans_links(self):
        x = numpy.random.randn(10, 3)
        model = KMeans(3)
        model.fit(x)
        onx = to_onnx(model, x.astype(numpy.float32), target_opset=15)
        dot = to_dot(onx, recursive=True)
        self.assertIn("Sq_Sqrt -> scores;", dot)

    def test_function_plot(self):
        new_domain = "custom"
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,  # domain name
            "LinearRegression",  # function name
            ["X", "A", "B"],  # input names
            ["Y"],  # output names
            [node1, node2],  # nodes
            opset_imports,  # opsets
            [],
        )  # attribute names

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, None)

        graph = make_graph(
            [
                make_node(
                    "LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain
                ),
                make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [X, A, B],
            [Y],
        )

        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=[linear_regression]
        )  # functions to add)
        dot = to_dot(onnx_model, add_functions=True, recursive=True)
        self.assertIn("LinearRegression -> Y1;", dot)

    def test_onnx_text_plot_tree_simple(self):
        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeRegressor(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        dot = to_dot(onx)
        self.assertIn("TreeEnsembleRegressor -> variable;", dot)

    def test_simple_text_plot_bug(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        onx_file = os.path.join(data, "tree_torch.onnx")
        onx = load(onx_file)
        dot = to_dot(onx)
        self.assertIn("onnx____ReduceSum_140 [shape=box", dot)

    def test_simple_text_plot_ref_attr_name(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        onx_file = os.path.join(data, "bug_Hardmax.onnx")
        onx = load(onx_file)
        dot = to_dot(onx)
        self.assertIn("Hardmax -> y;", dot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
