import os
import unittest
import onnx
from onnx.inliner import inline_local_functions
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.graph_api.graph_builder import GraphBuilder


class TestGraphSimplification(ExtTestCase):
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

    def test_check_afiles(self):
        import onnxruntime

        data = os.path.join(os.path.dirname(__file__), "data")
        filename = [f for f in os.listdir(data) if f.endswith(".onnx")]
        for f in filename:
            with self.subTest(f=f):
                onx = onnx.load(os.path.join(data, f))
                sess = onnxruntime.InferenceSession(
                    os.path.join(data, f), providers=["CPUExecutionProvider"]
                )
                assert sess
                onxi = inline_local_functions(onx)
                sess = onnxruntime.InferenceSession(
                    onxi.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                assert sess
                g = GraphBuilder(onxi)
                g.optimize(check_order=True)
                g.check_order()
                onx2 = g.to_onnx()
                sess2 = onnxruntime.InferenceSession(
                    onx2.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                assert sess2


if __name__ == "__main__":
    unittest.main(verbosity=2)
