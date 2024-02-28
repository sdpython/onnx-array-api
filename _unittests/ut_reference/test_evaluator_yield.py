import unittest
import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
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
            001=|INPUTfloat322:2x2ABCDA|INPUTfloat322:2x2ABCDA
            002=|INPUTfloat322:2x2ABCDB|INPUTfloat322:2x2ABCDB
            003~|INPUTfloat322:2x3ABCDX|INPUTfloat322:2x2ABCDX
            004-|RESULTfloat322:2x2CEIOExpH|
            005=|RESULTfloat322:2x2CEIOLinearRegressioY1|RESULTfloat322:2x2CEIOLinearRegressioY1
            006~|RESULTfloat322:2x2CEIOAbsY|RESULTfloat322:2x3CEIPAbsZ
            007~|OUTPUTfloat322:2x2CEIOY|OUTPUTfloat322:2x2CEIPY
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

    def test_compare_execution_discrepancies(self):
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
        res1, res2, align, dc = compare_onnx_execution(m1, m2, keep_tensor=True)
        text = dc.to_str(res1, res2, align)
        print(text)
        self.assertIn("CAAA Constant", text)
        self.assertIn("| a=", text)
        self.assertIn(" r=", text)

    def test_no_execution(self):
        model = make_model(
            make_graph(
                [
                    make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, "N"])],
                [
                    from_array(np.array([0], dtype=np.int64), name="zero"),
                    from_array(np.array([1], dtype=np.int64), name="un"),
                    from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model)
        res1, res2, align, dc = compare_onnx_execution(model, model, mode="nodes")
        text = dc.to_str(res1, res2, align)
        self.assertIn("012 = | NODE", text)

        model2 = make_model(
            make_graph(
                [
                    make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    make_node("MatMul", ["xm1", "xm2c"], ["xm"]),
                    make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, [32, 128]),
                    make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 128, 64]),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 32, "N"])],
                [
                    from_array(np.array([0], dtype=np.int64), name="zero"),
                    from_array(np.array([1], dtype=np.int64), name="un"),
                    from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        check_model(model2)
        res1, res2, align, dc = compare_onnx_execution(model, model2, mode="nodes")
        text = dc.to_str(res1, res2, align)
        self.assertIn("012 = | NODE", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
