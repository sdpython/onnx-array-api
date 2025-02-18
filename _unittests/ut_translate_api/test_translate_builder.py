import unittest
from textwrap import dedent
import numpy as np
import onnx.helper as oh
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.light_api import start
from onnx_array_api.graph_api import GraphBuilder
from onnx_array_api.translate_api import translate, Translater
from onnx_array_api.translate_api.builder_emitter import BuilderEmitter


OPSET_API = min(19, onnx_opset_version() - 1)


class TestTranslateBuilder(ExtTestCase):
    def setUp(self):
        self.maxDiff = None

    def test_exp(self):
        onx = start(opset=19, ir_version=10).vin("X").Exp().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Exp", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

        code = translate(onx, api="builder")
        expected = (
            dedent(
                """
        def light_api(
            op: "GraphBuilder",
            X: "FLOAT[]",
        ):
            Y = op.Exp(X)
            op.Identity(Y, outputs=["Y"])
            return Y

        g = GraphBuilder({'': 19}, ir_version=10)
        g.make_tensor_input("X", TensorProto.FLOAT, ())
        light_api(g.op, "X")
        g.make_tensor_output("Y", TensorProto.FLOAT, ()__SUFFIX__)
        model = g.to_onnx()
        """
            )
            .strip("\n")
            .replace("__SUFFIX__", ", is_dimension=False, indexed=False")
        )
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
        g2.make_tensor_output(
            "Y", TensorProto.FLOAT, ("A",), is_dimension=False, indexed=False
        )
        onx2 = g2.to_onnx()

        ref = ReferenceEvaluator(onx2)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

    def test_zdoc(self):
        onx = (
            start(opset=19, ir_version=10)
            .vin("X")
            .reshape((-1, 1))
            .Transpose(perm=[1, 0])
            .rename("Y")
            .vout()
            .to_onnx()
        )
        code = translate(onx, api="builder")
        expected = (
            dedent(
                """
            def light_api(
                op: "GraphBuilder",
                X: "FLOAT[]",
            ):
                r = np.array([-1, 1], dtype=np.int64)
                r0_0 = op.Reshape(X, r)
                Y = op.Transpose(r0_0, perm=[1, 0])
                op.Identity(Y, outputs=["Y"])
                return Y

            g = GraphBuilder({'': 19}, ir_version=10)
            g.make_tensor_input("X", TensorProto.FLOAT, ())
            light_api(g.op, "X")
            g.make_tensor_output("Y", TensorProto.FLOAT, ()__SUFFIX__)
            model = g.to_onnx()
            """
            )
            .strip("\n")
            .replace("__SUFFIX__", ", is_dimension=False, indexed=False")
        )
        self.maxDiff = None
        self.assertEqual(expected, code.strip("\n"))

        def light_api(
            op: "GraphBuilder",
            X: "FLOAT[]",  # noqa: F722
        ):
            r = np.array([-1, 1], dtype=np.int64)
            r0_0 = op.Reshape(X, r)
            Y = op.Transpose(r0_0, perm=[1, 0])
            op.Identity(Y, outputs=["Y"])
            return Y

        g = GraphBuilder({"": 21})
        X = g.make_tensor_input("X", TensorProto.FLOAT, ())
        light_api(g.op, X)
        g.make_tensor_output("Y", TensorProto.FLOAT, ())
        model = g.to_onnx()
        self.assertNotEmpty(model)
        check_model(model)

    def test_exp_f(self):
        onx = start(opset=19, ir_version=10).vin("X").Exp().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Exp", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

        tr = Translater(onx, emitter=BuilderEmitter("mm"))
        code = tr.export(as_str=True)

        expected = (
            dedent(
                """
        def light_api(
            op: "GraphBuilder",
            X: "FLOAT[]",
        ):
            Y = op.Exp(X)
            op.Identity(Y, outputs=["Y"])
            return Y


        def mm() -> "ModelProto":
            g = GraphBuilder({'': 19}, ir_version=10)
            g.make_tensor_input("X", TensorProto.FLOAT, ())
            light_api(g.op, "X")
            g.make_tensor_output("Y", TensorProto.FLOAT, ()__SUFFIX__)
            model = g.to_onnx()
            return model


        model = mm()
        """
            )
            .strip("\n")
            .replace("__SUFFIX__", ", is_dimension=False, indexed=False")
        )
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
        g2.make_tensor_output(
            "Y", TensorProto.FLOAT, ("A",), is_dimension=False, indexed=False
        )
        onx2 = g2.to_onnx()

        ref = ReferenceEvaluator(onx2)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

    def test_local_function(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node(
                    "LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain
                ),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        tr = Translater(onnx_model, emitter=BuilderEmitter("mm"))
        code = tr.export(as_str=True)

        expected = (
            dedent(
                """
            def example(
                op: "GraphBuilder",
                X: "FLOAT[, ]",
                A: "FLOAT[, ]",
                B: "FLOAT[, ]",
            ):
                Y1 = op.LinearRegression(X, A, B, domain='custom')
                Y = op.Abs(Y1)
                op.Identity(Y, outputs=["Y"])
                return Y


            def make_custom_LinearRegression(g: "GraphBuilder"):
                gr = GraphBuilder({'': 14}, as_function=True)
                x = gr.make_tensor_input('x')
                a = gr.make_tensor_input('a')
                b = gr.make_tensor_input('b')
                op = gr.op
                xa = op.MatMul(x, a)
                y = op.Add(xa, b)
                gr.make_tensor_output(y)
                g.add_function(builder=gr)
                return gr


            def mm() -> "ModelProto":
                g = GraphBuilder({'': 14, 'custom': 1}, ir_version=11)
                g.make_tensor_input("X", TensorProto.FLOAT, ('', ''))
                g.make_tensor_input("A", TensorProto.FLOAT, ('', ''))
                g.make_tensor_input("B", TensorProto.FLOAT, ('', ''))
                example(g.op, "X", "A", "B")
                g.make_tensor_output("Y", TensorProto.FLOAT, ()__SUFFIX__)
                make_custom_LinearRegression(g)
                model = g.to_onnx()
                return model


            model = mm()
        """
            )
            .strip("\n")
            .replace("__SUFFIX__", ", is_dimension=False, indexed=False")
        )
        self.assertEqual(expected, code.strip("\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
