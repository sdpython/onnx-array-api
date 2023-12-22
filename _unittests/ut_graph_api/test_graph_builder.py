import contextlib
import io
import unittest
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.graph_api.graph_builder import GraphBuilder


class TestGraphBuilder(ExtTestCase):
    def call_optimizer(self, onx):
        gr = GraphBuilder(onx)
        gr.remove_unused()
        return gr.to_onnx()

    def test_remove_unused_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }"""
        )
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Mul")

    def test_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z)
            <float two = {2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }"""
        )
        self.assertEqual(len(model.graph.initializer), 1)
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Mul")
        self.assertEqual(len(onx.graph.initializer), 0)

    def test_keep_unused_outputs(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[M] z) {
                w1, w2, w3 = Split (x)
                z = Mul(w3, w3)
            }"""
        )
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 2)
        self.assertEqual(onx.graph.node[0].op_type, "Split")

    def test_exc(self):
        self.assertRaise(lambda: GraphBuilder([]), NotImplementedError)

    def test_simple(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (10, 4)
            w = np.random.randn(*shape).astype(np.float32)

            x = g.make_tensor_input("X", np.float32, shape)
            weight = g.make_initializer(w)
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.MatMul(x, transposed)
            g.op.Reshape(res, one, outputs="y")
            g.make_tensor_output("y", np.float32, (10, 1))
            onx = g.to_onnx()
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])

    def test_simple_big(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (30, 40)
            w = np.random.randn(*shape).astype(np.float32)

            x = g.make_tensor_input("X", np.float32, shape)
            weight = g.make_initializer(w)
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.MatMul(x, transposed)
            g.op.Reshape(res, one, outputs="y")
            g.make_tensor_output("y", np.float32, (30, 1))
            onx = g.to_onnx()
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])

    def test_constant_folding(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (10, 4)
            w = np.random.randn(*shape).astype(np.float32)
            x = g.make_tensor_input("X", np.float32, shape)
            weight = g.make_initializer(w)
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.MatMul(x, transposed)
            g.op.Reshape(res, one, outputs="y")
            g.make_tensor_output("y", np.float32, (10, 1))

            g.constant_folding()

            onx = g.to_onnx()
            node_types = [n.op_type for n in onx.graph.node]
            self.assertNotIn("Transpose", node_types)
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])

    def test_remove_identity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (10, 4)
            w = np.random.randn(*shape).astype(np.float32)
            x = g.make_tensor_input("X", np.float32, shape)
            weight = g.make_initializer(w)
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.Identity(g.op.MatMul(x, transposed))
            g.op.Reshape(res, one, outputs="y")
            g.make_tensor_output("y", np.float32, (10, 1))

            g.remove_identity_nodes()

            onx = g.to_onnx()
            node_types = [n.op_type for n in onx.graph.node]
            self.assertNotIn("Identity", node_types)
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])

    def test_remove_identity_input(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (10, 4)
            w = np.random.randn(*shape).astype(np.float32)
            x = g.make_tensor_input("X", np.float32, shape)
            x = g.op.Identity(x)
            weight = g.make_initializer(w)
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.MatMul(x, transposed)
            g.op.Reshape(res, one, outputs="y")
            g.make_tensor_output("y", np.float32, (10, 1))

            g.remove_identity_nodes()

            onx = g.to_onnx()
            node_types = [n.op_type for n in onx.graph.node]
            self.assertNotIn("Identity", node_types)
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])

    def test_remove_identity_output(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (10, 4)
            w = np.random.randn(*shape).astype(np.float32)
            x = g.make_tensor_input("X", np.float32, shape)
            weight = g.make_initializer(w)
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.MatMul(x, transposed)
            r = g.op.Reshape(res, one)
            g.op.Identity(r, outputs=["y"])
            g.make_tensor_output("y", np.float32, (10, 1))

            g.remove_identity_nodes()

            onx = g.to_onnx()
            node_types = [n.op_type for n in onx.graph.node]
            self.assertNotIn("Identity", node_types)
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])

    def test_remove_unused_nodes_simple(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = GraphBuilder(verbose=10)

            shape = (10, 4)
            w = np.random.randn(*shape).astype(np.float32)
            x = g.make_tensor_input("X", np.float32, shape)
            weight = g.make_initializer(w)
            cst = g.make_initializer(np.array([2], dtype=np.float32))
            one = g.make_initializer(np.array([-1, 1], dtype=np.int64))
            transposed = g.make_node("Transpose", [weight], perm=[1, 0])
            res = g.op.MatMul(x, transposed)
            g.op.Add(res, cst)
            g.op.Reshape(res, one, outputs=["y"])
            g.make_tensor_output("y", np.float32, (10, 1))

            g.remove_identity_nodes()

            onx = g.to_onnx()
            node_types = [n.op_type for n in onx.graph.node]
            self.assertNotIn("Add", node_types)
            ref = ReferenceEvaluator(onx)
            x = np.random.randn(*shape).astype(np.float32)
            expected = (x @ w.T).reshape((-1, 1))
            feeds = {"X": x}
            got = ref.run(None, feeds)
            self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
