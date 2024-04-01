import unittest
from textwrap import dedent
import numpy as np
from onnx import ModelProto, TensorProto
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.light_api import start
from onnx_array_api.graph_api import GraphBuilder
from onnx_array_api.translate_api import translate


OPSET_API = min(19, onnx_opset_version() - 1)


class TestTranslateBuilder(ExtTestCase):
    def setUp(self):
        self.maxDiff = None

    def test_exp(self):
        onx = start(opset=19).vin("X").Exp().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Exp", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

        code = translate(onx, api="builder")
        expected = dedent(
            """
        def light_api(
            op: "GraphBuilder",
            X: "FLOAT[]",
        ):
            Y = op.Exp(X)
            op.Identity(Y, outputs=["Y"])
            return Y

        g = GraphBuilder({'': 19})
        g.make_tensor_input("X", TensorProto.FLOAT, ())
        light_api(g.op, X)
        g.make_tensor_output("Y", TensorProto.FLOAT, ())
        model = g.to_onnx()
        """
        ).strip("\n")
        self.assertEqual(expected, code.strip("\n"))

        def light_api(
            op: "GraphBuilder",
            X: "FLOAT[]",  # noqa: F722
        ):
            Y = op.Exp(X)
            op.Identity(Y, outputs=["Y"])
            return Y

        g2 = GraphBuilder({"": 19})
        g2.make_tensor_input("X", TensorProto.FLOAT, ("A",))
        light_api(g2.op, "X")
        g2.make_tensor_output("Y", TensorProto.FLOAT, ("A",))
        onx2 = g2.to_onnx()

        ref = ReferenceEvaluator(onx2)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

    def test_zdoc(self):
        onx = (
            start()
            .vin("X")
            .reshape((-1, 1))
            .Transpose(perm=[1, 0])
            .rename("Y")
            .vout()
            .to_onnx()
        )
        code = translate(onx, api="builder")
        expected = dedent(
            """
            (
                start()
                .vin("X")
                .reshape((-1, 1))
                .Transpose(perm=[1, 0])
                .rename("Y")
                .vout()
                .to_onnx()
            )"""
        ).strip("\n")
        self.maxDiff = None
        self.assertEqual(expected, code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
