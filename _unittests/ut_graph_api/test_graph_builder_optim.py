import os
import unittest
import onnx
from onnx.inliner import inline_local_functions
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.graph_api.graph_builder import GraphBuilder


class TestGraphBuilderOptim(ExtTestCase):
    def test_wcheck_afiles(self):
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
