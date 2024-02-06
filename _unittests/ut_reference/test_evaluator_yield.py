import unittest
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.parser import parse_model
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.reference import (
    YieldEvaluator,
    ResultType,
    DistanceExecution,
    ResultExecution,
    compare_onnx_execution,
)
from onnx_array_api.reference.evaluator_yield import make_summary


class TestArrayTensor(ExtTestCase):
    def test_make_summary(self):
        a = np.arange(12).reshape(3, 4)
        v = make_summary(a)
        self.assertEqual(v, "DMVE")
        a = np.arange(12)
        v = make_summary(a)
        self.assertEqual(v, "DMVE")
        a = np.arange(12).astype(np.float32)
        v = make_summary(a)
        self.assertEqual(v, "DMVE")
        a = np.arange(13)
        a[-1] = 0
        v = make_summary(a)
        self.assertEqual(v, "GWMA")

    def test_evaluator_yield(self):
        new_domain = "custom_domain"
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,
            "LinearRegression",
            ["X", "A", "B"],
            ["Y"],
            [node1, node2],
            opset_imports,
            [],
        )

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
        )

        cst = np.arange(4).reshape((-1, 2)).astype(np.float32)
        yield_eval = YieldEvaluator(onnx_model)
        results = list(
            yield_eval.enumerate_results(None, {"A": cst, "B": cst, "X": cst})
        )
        expected = [
            (
                ResultType.INPUT,
                "A",
                np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                None,
            ),
            (
                ResultType.INPUT,
                "B",
                np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                None,
            ),
            (
                ResultType.INPUT,
                "X",
                np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                None,
            ),
            (
                ResultType.RESULT,
                "Y1",
                np.array([[2.0, 4.0], [8.0, 14.0]], dtype=np.float32),
                "LinearRegression",
            ),
            (
                ResultType.RESULT,
                "Y",
                np.array([[2.0, 4.0], [8.0, 14.0]], dtype=np.float32),
                "Abs",
            ),
            (
                ResultType.OUTPUT,
                "Y",
                np.array([[2.0, 4.0], [8.0, 14.0]], dtype=np.float32),
                None,
            ),
        ]
        self.assertEqual(len(expected), len(results))
        for a, b in zip(expected, results):
            self.assertEqual(len(a), len(b))
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[2].tolist(), b[2].tolist())
            self.assertEqual(a[3], b[3])

    def test_evaluator_yield_summary(self):
        new_domain = "custom_domain"
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,
            "LinearRegression",
            ["X", "A", "B"],
            ["Y"],
            [node1, node2],
            opset_imports,
            [],
        )

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
        )

        cst = np.arange(4).reshape((-1, 2)).astype(np.float32)
        yield_eval = YieldEvaluator(onnx_model)
        results = list(
            yield_eval.enumerate_summarized(None, {"A": cst, "B": cst, "X": cst})
        )
        expected = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]
        self.assertEqual(len(expected), len(results))
        for a, b in zip(expected, results):
            self.assertEqual(len(a), len(b))
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[2], b[2])
            self.assertEqual(a[3], b[3])
            self.assertEqual(a[4], b[4])
            self.assertEqual(a[5], b[5])

    def test_distance_pair(self):
        el1 = (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None)
        el2 = el1
        dc = DistanceExecution()
        self.assertEqual(dc.distance_pair(el1, el2), 0)
        el2 = (ResultType.INPUT, np.dtype("float16"), (2, 2), "ABCD", None)
        self.assertEqual(dc.distance_pair(el1, el2), 2)
        el2 = (ResultType.OUTPUT, np.dtype("float16"), (2, 2, 4), "GBCD", "Abs")
        self.assertEqual(dc.distance_pair(el1, el2), 1130)
        el2 = (ResultType.OUTPUT, np.dtype("float16"), (2, 3), "GBCD", "Abs")
        self.assertEqual(dc.distance_pair(el1, el2), 1021)

    def test_distance_sequence_0(self):
        expected = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]

        dc = DistanceExecution()
        d, align = dc.distance_sequence(expected, expected)
        self.assertEqual(d, 0)
        self.assertEqual(align, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    def test_distance_sequence_ins(self):
        s1 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]
        s2 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]

        dc = DistanceExecution()
        d, align = dc.distance_sequence(s1, s2)
        self.assertEqual(d, dc.insert_cost)
        self.assertEqual(align, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4)])
        d, align = dc.distance_sequence(s2, s1)
        self.assertEqual(d, dc.insert_cost)
        self.assertEqual(align, [(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5)])

    def test_distance_sequence_equal(self):
        s1 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]
        s2 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Z"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]

        dc = DistanceExecution()
        d, align = dc.distance_sequence(s1, s2)
        self.assertEqual(d, 0)
        self.assertEqual(align, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    def test_distance_sequence_diff(self):
        s1 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]
        s2 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIP", "Abs", "Z"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]

        dc = DistanceExecution()
        d, align = dc.distance_sequence(s1, s2)
        self.assertEqual(d, 1)
        self.assertEqual(align, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    def test_distance_sequence_diff2(self):
        s1 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]
        s2 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 3), "CEIP", "Abs", "Z"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIP", None, "Y"),
        ]

        dc = DistanceExecution()
        d, align = dc.distance_sequence(s1, s2)
        self.assertEqual(d, 5)
        self.assertEqual(align, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    def test_distance_sequence_str(self):
        s1 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 3), "ABCD", None, "X"),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Exp", "H"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs", "Y"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIO", None, "Y"),
        ]
        s2 = [
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "A"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "B"),
            (ResultType.INPUT, np.dtype("float32"), (2, 2), "ABCD", None, "X"),
            (
                ResultType.RESULT,
                np.dtype("float32"),
                (2, 2),
                "CEIO",
                "LinearRegression",
                "Y1",
            ),
            (ResultType.RESULT, np.dtype("float32"), (2, 3), "CEIP", "Abs", "Z"),
            (ResultType.OUTPUT, np.dtype("float32"), (2, 2), "CEIP", None, "Y"),
        ]
        s1 = [ResultExecution(*s) for s in s1]
        s2 = [ResultExecution(*s) for s in s2]

        dc = DistanceExecution()
        d, align = dc.distance_sequence(s1, s2)
        self.assertEqual(d, 1008)
        self.assertEqual(
            align, [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4), (6, 5)]
        )
        text = dc.to_str(s1, s2, align)
        self.assertIn("OUTPUT", text)
        expected = """
            1=|INPUTfloat322x2ABCDA|INPUTfloat322x2ABCDA
            2=|INPUTfloat322x2ABCDB|INPUTfloat322x2ABCDB
            3~|INPUTfloat322x3ABCDX|INPUTfloat322x2ABCDX
            4-|RESULTfloat322x2CEIOExpH|
            5=|RESULTfloat322x2CEIOLinearRegrY1|RESULTfloat322x2CEIOLinearRegrY1
            6~|RESULTfloat322x2CEIOAbsY|RESULTfloat322x3CEIPAbsZ
            7~|OUTPUTfloat322x2CEIOY|OUTPUTfloat322x2CEIPY
        """.replace(
            "            ", ""
        ).strip(
            "\n "
        )
        self.maxDiff = None
        self.assertEqual(expected, text.replace(" ", "").strip("\n"))

    def test_compare_execution(self):
        m1 = parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }"""
        )
        m2 = parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                z = Mul(x, x)
            }"""
        )
        res1, res2, align, dc = compare_onnx_execution(m1, m2)
        text = dc.to_str(res1, res2, align)
        self.assertIn("CAAA Constant", text)
        self.assertEqual(len(align), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
