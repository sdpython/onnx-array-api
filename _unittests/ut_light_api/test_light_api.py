import unittest
import numpy as np
from onnx import ModelProto
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.light_api import start, OnnxGraph, Var


class TestLightApi(ExtTestCase):
    def test_neg(self):
        onx = start()
        self.assertIsInstance(onx, OnnxGraph)
        r = repr(onx)
        self.assertEqual("OnnxGraph()", r)
        v = start().vin("X")
        self.assertIsInstance(v, Var)
        s = str(v)
        self.assertEqual("X:FLOAT", s)
        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(-a, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
