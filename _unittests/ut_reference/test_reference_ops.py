import unittest
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
    make_opsetid,
)
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.reference import ExtendedReferenceEvaluator


class TestReferenceOps(ExtTestCase):

    def test_fused_matmul(self):
        model = make_model(
            make_graph(
                [make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")],
                "name",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, None),
                    make_tensor_value_info("Y", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), make_opsetid("com.microsoft", 1)],
        )
        ref = ExtendedReferenceEvaluator(model)
        a = np.arange(4).reshape(-1, 2)
        got = ref.run(None, {"X": a, "Y": a})
        self.assertEqualArray(a @ a, got[0])

    def test_fused_matmul11(self):
        model = make_model(
            make_graph(
                [
                    make_node(
                        "FusedMatMul",
                        ["X", "Y"],
                        ["Z"],
                        transA=1,
                        transB=1,
                        domain="com.microsoft",
                    )
                ],
                "name",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, None),
                    make_tensor_value_info("Y", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), make_opsetid("com.microsoft", 1)],
        )
        ref = ExtendedReferenceEvaluator(model)
        a = np.arange(4).reshape(-1, 2)
        got = ref.run(None, {"X": a, "Y": a})
        self.assertEqualArray(a.T @ a.T, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
