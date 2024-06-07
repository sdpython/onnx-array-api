import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.reference import (
    ExtendedReferenceEvaluator as ReferenceEvaluator,
)
from onnx_array_api.tools.replace_constants import (
    replace_initializer_by_constant_of_shape,
)


class TestReplaceConstants(ExtTestCase):

    def test_replace_initializer(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = onh.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = onh.from_array(value, name="C")

        X = oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = oh.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = oh.make_node("Sub", ["AX", "C"], ["Y"])
        graph = oh.make_graph([node1, node2], "lr", [X], [Y], [A, C])
        model_def = oh.make_model(graph)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = {n.op_type for n in repl.graph.node}
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        self.assertEqualArray(y1, y2)

    def test_replace_constant(self):
        dtype = np.float32
        value = np.random.randn(2, 10).astype(dtype)
        A = onh.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = onh.from_array(value, name="C")

        X = oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node0 = oh.make_node("Constant", [], ["A"], value=A)
        node1 = oh.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = oh.make_node("Sub", ["AX", "C"], ["Y"])
        graph = oh.make_graph([node0, node1, node2], "lr", [X], [Y], [C])
        model_def = oh.make_model(graph)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def, threshold=0)
        node_types = {n.op_type for n in repl.graph.node}
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 4
        y1[0, :] = 1
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        self.assertEqualArray(y1, y2)

    def test_replace_constant_function(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = onh.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = onh.from_array(value, name="C")

        X = oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        nodeC = oh.make_node("Constant", [], ["C"], value=C)
        node0 = oh.make_node("Constant", [], ["A"], value=A)
        node1 = oh.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = oh.make_node("Sub", ["AX", "C"], ["Y"])
        opset_imports = [
            oh.make_opsetid("", onnx.defs.onnx_opset_version()),
            oh.make_opsetid("custom", 1),
        ]
        fct = oh.make_function(
            "custom",
            "unittest",
            ["X"],
            ["Y"],
            [nodeC, node0, node1, node2],
            opset_imports,
        )

        node = oh.make_node("unittest", ["X"], ["Y"], domain="custom")
        graph = oh.make_graph([node], "lr", [X], [Y], [C])
        model_def = oh.make_model(graph, functions=[fct], opset_imports=opset_imports)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = {n.op_type for n in repl.functions[0].node}
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        self.assertEqualArray(y1, y2)

    def test_replace_constant_graph(self):
        value = np.array([0], dtype=np.float32)
        zero = onh.from_array(value, name="zero")

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])

        rsum = oh.make_node("ReduceSum", ["X"], ["rsum"])
        cond = oh.make_node("Greater", ["rsum", "zero"], ["cond"])

        then_out = oh.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, None)
        then_cst = onh.from_array(np.array([1] * 129).astype(np.float32))

        then_const_node = oh.make_node(
            "Constant", inputs=[], outputs=["then_out"], value=then_cst, name="cst1"
        )
        then_body = oh.make_graph([then_const_node], "then_body", [], [then_out])

        else_out = oh.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, None)
        else_cst = onh.from_array(np.array([-1] * 129).astype(np.float32))
        else_const_node = oh.make_node(
            "Constant", inputs=[], outputs=["else_out"], value=else_cst, name="cst2"
        )
        else_body = oh.make_graph([else_const_node], "else_body", [], [else_out])

        if_node = oh.make_node(
            "If", ["cond"], ["Y"], then_branch=then_body, else_branch=else_body
        )
        graph = oh.make_graph([rsum, cond, if_node], "if", [X], [Y], [zero])
        onnx_model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", onnx.defs.onnx_opset_version())]
        )
        self.assertNotIn("ConstantOfShape", str(onnx_model))

        x = np.ones((3, 2), dtype=np.float32)
        oinf1 = ReferenceEvaluator(onnx_model)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(onnx_model)
        self.assertIn("ConstantOfShape", str(repl))
        oinf2 = ReferenceEvaluator(repl)
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        y1 = y1.copy()
        y1[:] = 0.5
        self.assertEqualArray(y1, y2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
