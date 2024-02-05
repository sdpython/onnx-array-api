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
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.reference import YieldEvaluator, ResultType


class TestArrayTensor(ExtTestCase):
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
            (ResultType.INPUT, "A", np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)),
            (ResultType.INPUT, "B", np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)),
            (ResultType.INPUT, "X", np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)),
            (ResultType.RESULT, "Y1", np.array([[2.0, 4.0], [8.0, 14.0]], dtype=np.float32)),
            (ResultType.RESULT, "Y", np.array([[2.0, 4.0], [8.0, 14.0]], dtype=np.float32)),
            (ResultType.OUTPUT, "Y", np.array([[2.0, 4.0], [8.0, 14.0]], dtype=np.float32)),
        ]
        self.assertEqual(len(expected), len(results))
        for a, b in zip(expected, results):
            self.assertEqual(len(a), len(b))
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[2].tolist(), b[2].tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
