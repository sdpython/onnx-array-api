import unittest
from textwrap import dedent
import numpy as np
from onnx import ModelProto, TensorProto
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.light_api import start, translate
from onnx_array_api.light_api.emitter import EventType

OPSET_API = min(19, onnx_opset_version() - 1)


class TestTranslate(ExtTestCase):
    def test_event_type(self):
        self.assertEqual(
            EventType.to_str(EventType.INITIALIZER), "EventType.INITIALIZER"
        )

    def test_exp(self):
        onx = start(opset=19).vin("X").Exp().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Exp", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

        code = translate(onx)
        expected = dedent(
            """
        (
            start(opset=19)
            .vin('X', elem_type=TensorProto.FLOAT)
            .bring('X')
            .Exp()
            .rename('Y')
            .bring('Y')
            .vout(elem_type=TensorProto.FLOAT)
            .to_onnx()
        )"""
        ).strip("\n")
        self.assertEqual(expected, code)

        onx2 = (
            start(opset=19)
            .vin("X", elem_type=TensorProto.FLOAT)
            .bring("X")
            .Exp()
            .rename("Y")
            .bring("Y")
            .vout(elem_type=TensorProto.FLOAT)
            .to_onnx()
        )
        ref = ReferenceEvaluator(onx2)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

    def test_transpose(self):
        onx = (
            start(opset=19)
            .vin("X")
            .reshape((-1, 1))
            .Transpose(perm=[1, 0])
            .rename("Y")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Transpose", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(a.reshape((-1, 1)).T, got)

        code = translate(onx)
        expected = dedent(
            """
            (
                start(opset=19)
                .cst(np.array([-1,  1]).astype(int64))
                .rename('r')
                .vin('X', elem_type=TensorProto.FLOAT)
                .bring('X', 'r')
                .Reshape()
                .rename('r0_0')
                .bring('r0_0')
                .Transpose(perm=[1, 0])
                .rename('Y')
                .bring('Y')
                .vout(elem_type=TensorProto.FLOAT)
                .to_onnx()
            )"""
        ).strip("\n")
        self.assertEqual(expected, code)

    def test_topk_reverse(self):
        onx = (
            start(opset=19)
            .vin("X", np.float32)
            .vin("K", np.int64)
            .bring("X", "K")
            .TopK(largest=0)
            .rename("Values", "Indices")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        x = np.array([[0, 1, 2, 3], [9, 8, 7, 6]], dtype=np.float32)
        k = np.array([2], dtype=np.int64)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(np.array([[0, 1], [6, 7]], dtype=np.float32), got[0])
        self.assertEqualArray(np.array([[0, 1], [3, 2]], dtype=np.int64), got[1])

        code = translate(onx)
        expected = dedent(
            """
            (
                start(opset=19)
                .vin('X', elem_type=TensorProto.FLOAT)
                .vin('K', elem_type=TensorProto.INT64)
                .bring('X', 'K')
                .TopK(axis=-1, largest=0, sorted=1)
                .rename('Values', 'Indices')
                .bring('Values')
                .vout(elem_type=TensorProto.FLOAT)
                .bring('Indices')
                .vout(elem_type=TensorProto.FLOAT)
                .to_onnx()
            )"""
        ).strip("\n")
        self.assertEqual(expected, code)


if __name__ == "__main__":
    # TestLightApi().test_topk()
    unittest.main(verbosity=2)
