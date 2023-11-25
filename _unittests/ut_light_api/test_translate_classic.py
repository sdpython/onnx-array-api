import unittest
import os
from textwrap import dedent
import numpy as np
from onnx import ModelProto, TensorProto, load
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx.helper import (
    make_tensor_value_info,
    make_node,
    make_graph,
    make_model,
    make_opsetid,
)
from onnx.checker import check_model
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.light_api import start, translate

OPSET_API = min(19, onnx_opset_version() - 1)


class TestTranslateClassic(ExtTestCase):
    def test_check_code(self):
        opset_imports = [
            make_opsetid("", 19),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(make_tensor_value_info("X", TensorProto.FLOAT, shape=[]))
        nodes.append(make_node("Exp", ["X"], ["Y"]))
        outputs.append(make_tensor_value_info("Y", TensorProto.FLOAT, shape=[]))
        graph = make_graph(
            nodes,
            "onename",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = make_model(graph, functions=functions, opset_imports=opset_imports)
        check_model(model)

    def test_exp(self):
        onx = start(opset=19).vin("X").Exp().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Exp", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

        code = translate(onx, api="onnx")

        expected = dedent(
            """
        opset_imports = [
            make_opsetid('', 19),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(make_tensor_value_info('X', TensorProto.FLOAT, shape=[]))
        nodes.append(
            make_node(
                'Exp',
                ['X'],
                ['Y']
            )
        )
        outputs.append(make_tensor_value_info('Y', TensorProto.FLOAT, shape=[]))
        graph = make_graph(
            nodes,
            'light_api',
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = make_model(
            graph,
            functions=functions,
            opset_imports=opset_imports
        )"""
        ).strip("\n")
        self.maxDiff = None
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

        code = translate(onx, api="onnx")
        expected = dedent(
            """
            opset_imports = [
                make_opsetid('', 19),
            ]
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []
            functions = []
            initializers.append(
                from_array(
                    np.array([-1, 1], dtype=np.int64),
                    name='r'
                )
            )
            inputs.append(make_tensor_value_info('X', TensorProto.FLOAT, shape=[]))
            nodes.append(
                make_node(
                    'Reshape',
                    ['X', 'r'],
                    ['r0_0']
                )
            )
            nodes.append(
                make_node(
                    'Transpose',
                    ['r0_0'],
                    ['Y'],
                    perm=[1, 0]
                )
            )
            outputs.append(make_tensor_value_info('Y', TensorProto.FLOAT, shape=[]))
            graph = make_graph(
                nodes,
                'light_api',
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            model = make_model(
                graph,
                functions=functions,
                opset_imports=opset_imports
            )"""
        ).strip("\n")
        self.maxDiff = None
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

        code = translate(onx, api="onnx")
        expected = dedent(
            """
            opset_imports = [
                make_opsetid('', 19),
            ]
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []
            functions = []
            inputs.append(make_tensor_value_info('X', TensorProto.FLOAT, shape=[]))
            inputs.append(make_tensor_value_info('K', TensorProto.INT64, shape=[]))
            nodes.append(
                make_node(
                    'TopK',
                    ['X', 'K'],
                    ['Values', 'Indices'],
                    axis=-1,
                    largest=0,
                    sorted=1
                )
            )
            outputs.append(make_tensor_value_info('Values', TensorProto.FLOAT, shape=[]))
            outputs.append(make_tensor_value_info('Indices', TensorProto.FLOAT, shape=[]))
            graph = make_graph(
                nodes,
                'light_api',
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            model = make_model(
                graph,
                functions=functions,
                opset_imports=opset_imports
            )"""
        ).strip("\n")
        self.maxDiff = None
        self.assertEqual(expected, code)

    def test_fft(self):
        data = os.path.join(
            os.path.dirname(__file__), "_data", "stft_inlined_batch_1.onnx"
        )
        onx = load(data)
        code = translate(onx, api="onnx")
        try:
            compile(code, "<string>", mode="exec")
        except Exception as e:
            new_code = "\n".join(
                [f"{i+1:04} {line}" for i, line in enumerate(code.split("\n"))]
            )
            raise AssertionError(f"ERROR {e}\n{new_code}")

    def test_aionnxml(self):
        onx = (
            start(opset=19, opsets={"ai.onnx.ml": 3})
            .vin("X")
            .reshape((-1, 1))
            .rename("USE")
            .ai.onnx.ml.Normalizer(norm="MAX")
            .rename("Y")
            .vout()
            .to_onnx()
        )
        code = translate(onx, api="onnx")
        print(code)
        expected = dedent(
            """
            opset_imports = [
                make_opsetid('', 19),
                make_opsetid('ai.onnx.ml', 3),
            ]
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []
            functions = []
            initializers.append(
                from_array(
                    np.array([-1, 1], dtype=np.int64),
                    name='r'
                )
            )
            inputs.append(make_tensor_value_info('X', TensorProto.FLOAT, shape=[]))
            nodes.append(
                make_node(
                    'Reshape',
                    ['X', 'r'],
                    ['USE']
                )
            )
            nodes.append(
                make_node(
                    'Normalizer',
                    ['USE'],
                    ['Y'],
                    domain='ai.onnx.ml',
                    norm='MAX'
                )
            )
            outputs.append(make_tensor_value_info('Y', TensorProto.FLOAT, shape=[]))
            graph = make_graph(
                nodes,
                'light_api',
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            model = make_model(
                graph,
                functions=functions,
                opset_imports=opset_imports
            )"""
        ).strip("\n")
        self.maxDiff = None
        self.assertEqual(expected, code)


if __name__ == "__main__":
    # TestLightApi().test_topk()
    unittest.main(verbosity=2)
