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
        self.assertEqual(["X"], v.parent.input_names)
        s = str(v)
        self.assertEqual("X:FLOAT", s)
        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(-a, got)

    def test_add(self):
        onx = start()
        onx = (
            start().vin("X").vin("Y").bring("X", "Y").Add().rename("Z").vout().to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "Y": a + 1})[0]
        self.assertEqualArray(a * 2 + 1, got)

    def test_add_constant(self):
        onx = start()
        onx = (
            start()
            .vin("X")
            .cst(np.array([1], dtype=np.float32), "one")
            .bring("X", "one")
            .Add()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "Y": a + 1})[0]
        self.assertEqualArray(a + 1, got)

    def test_left_bring(self):
        onx = start()
        onx = (
            start()
            .vin("X")
            .cst(np.array([1], dtype=np.float32), "one")
            .left_bring("X")
            .Add()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "Y": a + 1})[0]
        self.assertEqualArray(a + 1, got)

    def test_right_bring(self):
        onx = (
            start()
            .vin("S")
            .vin("X")
            .right_bring("S")
            .Reshape()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "S": np.array([-1], dtype=np.int64)})[0]
        self.assertEqualArray(a.ravel(), got)

    def test_reshape_1(self):
        onx = (
            start()
            .vin("X")
            .vin("S")
            .bring("X", "S")
            .Reshape()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "S": np.array([-1], dtype=np.int64)})[0]
        self.assertEqualArray(a.ravel(), got)

    def test_reshape_2(self):
        x = start().vin("X").vin("S").v("X")
        self.assertIsInstance(x, Var)
        self.assertEqual(x.name, "X")
        g = start()
        g.vin("X").vin("S").v("X").reshape("S").rename("Z").vout()
        self.assertEqual(["Z"], g.output_names)
        onx = start().vin("X").vin("S").v("X").reshape("S").rename("Z").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "S": np.array([-1], dtype=np.int64)})[0]
        self.assertEqualArray(a.ravel(), got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
