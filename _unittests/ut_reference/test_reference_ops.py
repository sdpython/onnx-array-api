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

    def test_memcpy(self):
        model = make_model(
            make_graph(
                [
                    make_node("MemcpyToHost", ["X"], ["Z"]),
                    make_node("MemcpyFromHost", ["X"], ["Z"]),
                ],
                "name",
                [make_tensor_value_info("X", TensorProto.FLOAT, None)],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        a = np.arange(4).reshape(-1, 2).astype(np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": a})
        self.assertEqualArray(a, got[0])

    def test_quick_gelu(self):
        from onnxruntime import InferenceSession

        for alpha in [0.0, 2.0]:
            model = make_model(
                make_graph(
                    [
                        make_node(
                            "QuickGelu",
                            ["X"],
                            ["Z"],
                            domain="com.microsoft",
                            alpha=alpha,
                        )
                    ],
                    "name",
                    [make_tensor_value_info("X", TensorProto.FLOAT, None)],
                    [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
                ),
                opset_imports=[make_opsetid("", 18), make_opsetid("com.microsoft", 1)],
                ir_version=9,
            )
            sess = InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            a = np.arange(4).reshape(-1, 2).astype(np.float32)
            expected = sess.run(None, {"X": a})
            ref = ExtendedReferenceEvaluator(model)
            got = ref.run(None, {"X": a})
            self.assertEqualArray(expected[0], got[0])

    def test_scatter_elements(self):
        model = make_model(
            make_graph(
                [
                    make_node(
                        "ScatterElements",
                        ["data", "indices", "updates"],
                        ["Z"],
                        axis=3,
                        reduction="add",
                    )
                ],
                "name",
                [
                    make_tensor_value_info("data", TensorProto.FLOAT, None),
                    make_tensor_value_info("indices", TensorProto.INT64, None),
                    make_tensor_value_info("updates", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18)],
        )
        data = np.zeros(2**4, dtype=np.float32).reshape((2, 2, 2, 2))
        indices = np.array([[[[0]]]], dtype=np.int64)
        updates = np.array([[[[1]]]], dtype=np.float32)
        y = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
        ).reshape((2, 2, 2, 2))
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"data": data, "indices": indices, "updates": updates})
        self.assertEqualArray(y, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
