import inspect
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import scipy
from numpy.testing import assert_allclose
from onnx import FunctionProto, ModelProto, TensorProto
from onnx.backend.test.case.node.pad import pad_impl
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator
from onnx.shape_inference import infer_shapes

from onnx_array_api.ext_test_case import ExtTestCase, ignore_warnings, skipif_ci_windows
from onnx_array_api.reference import ExtendedReferenceEvaluator
from onnx_array_api.npx import ElemType, eager_onnx, jit_onnx
from onnx_array_api.npx.npx_core_api import (
    cst,
    make_tuple,
    npxapi_function,
    npxapi_inline,
)
from onnx_array_api.npx.npx_functions import absolute as absolute_inline
from onnx_array_api.npx.npx_functions import all as all_inline
from onnx_array_api.npx.npx_functions import arange as arange_inline
from onnx_array_api.npx.npx_functions import arccos as arccos_inline
from onnx_array_api.npx.npx_functions import arccosh as arccosh_inline
from onnx_array_api.npx.npx_functions import arcsin as arcsin_inline
from onnx_array_api.npx.npx_functions import arcsinh as arcsinh_inline
from onnx_array_api.npx.npx_functions import arctan as arctan_inline
from onnx_array_api.npx.npx_functions import arctanh as arctanh_inline
from onnx_array_api.npx.npx_functions import argmin as argmin_inline
from onnx_array_api.npx.npx_functions import cdist as cdist_inline
from onnx_array_api.npx.npx_functions import ceil as ceil_inline
from onnx_array_api.npx.npx_functions import clip as clip_inline
from onnx_array_api.npx.npx_functions import compress as compress_inline
from onnx_array_api.npx.npx_functions import compute as compute_inline
from onnx_array_api.npx.npx_functions import concat as concat_inline
from onnx_array_api.npx.npx_functions import copy as copy_inline
from onnx_array_api.npx.npx_functions import cos as cos_inline
from onnx_array_api.npx.npx_functions import cosh as cosh_inline
from onnx_array_api.npx.npx_functions import cumsum as cumsum_inline
from onnx_array_api.npx.npx_functions import det as det_inline
from onnx_array_api.npx.npx_functions import dot as dot_inline
from onnx_array_api.npx.npx_functions import einsum as einsum_inline
from onnx_array_api.npx.npx_functions import equal as equal_inline
from onnx_array_api.npx.npx_functions import erf as erf_inline
from onnx_array_api.npx.npx_functions import exp as exp_inline
from onnx_array_api.npx.npx_functions import expand_dims as expand_dims_inline
from onnx_array_api.npx.npx_functions import expit as expit_inline
from onnx_array_api.npx.npx_functions import floor as floor_inline
from onnx_array_api.npx.npx_functions import hstack as hstack_inline
from onnx_array_api.npx.npx_functions import identity as identity_inline
from onnx_array_api.npx.npx_functions import isnan as isnan_inline
from onnx_array_api.npx.npx_functions import linspace as linspace_inline
from onnx_array_api.npx.npx_functions import log as log_inline
from onnx_array_api.npx.npx_functions import log1p as log1p_inline
from onnx_array_api.npx.npx_functions import matmul as matmul_inline
from onnx_array_api.npx.npx_functions import pad as pad_inline
from onnx_array_api.npx.npx_functions import reciprocal as reciprocal_inline
from onnx_array_api.npx.npx_functions import relu as relu_inline
from onnx_array_api.npx.npx_functions import round as round_inline
from onnx_array_api.npx.npx_functions import sigmoid as sigmoid_inline
from onnx_array_api.npx.npx_functions import sign as sign_inline
from onnx_array_api.npx.npx_functions import sin as sin_inline
from onnx_array_api.npx.npx_functions import sinh as sinh_inline
from onnx_array_api.npx.npx_functions import sqrt as sqrt_inline
from onnx_array_api.npx.npx_functions import squeeze as squeeze_inline
from onnx_array_api.npx.npx_functions import take as take_inline
from onnx_array_api.npx.npx_functions import tan as tan_inline
from onnx_array_api.npx.npx_functions import tanh as tanh_inline
from onnx_array_api.npx.npx_functions import topk as topk_inline
from onnx_array_api.npx.npx_functions import transpose as transpose_inline
from onnx_array_api.npx.npx_functions import unsqueeze as unsqueeze_inline
from onnx_array_api.npx.npx_functions import vstack as vstack_inline
from onnx_array_api.npx.npx_functions import where as where_inline
from onnx_array_api.npx.npx_functions_test import (
    _min_max,
    _min_max_inline,
    absolute,
    addition,
    argmin,
    concat,
    copy,
    log1p,
    negative,
    relu,
    topk,
)
from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor
from onnx_array_api.npx.npx_types import (
    Bool,
    DType,
    Float32,
    Float64,
    Int64,
    OptParType,
    TensorType,
    OptTensorType,
)
from onnx_array_api.npx.npx_var import Input, Var

DEFAULT_OPSET = onnx_opset_version()


class TestNpx(ExtTestCase):
    def test_shape_inference(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.UNDEFINED, [None, None])
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])
        graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        shapes = infer_shapes(onnx_model)
        output = shapes.graph.output[0]
        self.assertEqual(output.type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_tensor(self):
        dt = TensorType["float32", "F32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEmpty(dt.shape)
        self.assertEqual(dt.type_name(), "TensorType['float32', 'F32']")

        dt = TensorType["float32", "F32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "TensorType['float32', 'F32']")

        dt = TensorType[np.float32, "F32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "TensorType['float32', 'F32']")
        self.assertEmpty(dt.shape)

        dt = TensorType[np.str_, "TEXT"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.str_)
        self.assertEqual(dt.type_name(), "TensorType[strings, 'TEXT']")
        self.assertEmpty(dt.shape)

        self.assertRaise(lambda: TensorType[None], TypeError)
        self.assertRaise(lambda: TensorType[{np.float32, np.str_}], TypeError)

    def test_opt_tensor(self):
        dt = OptTensorType["float32", "F32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEmpty(dt.shape)
        self.assertEqual(dt.type_name(), "OptTensorType['float32', 'F32']")

        dt = OptTensorType["float32", "F32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "OptTensorType['float32', 'F32']")

        dt = OptTensorType[np.float32, "F32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "OptTensorType['float32', 'F32']")
        self.assertEmpty(dt.shape)

        dt = OptTensorType[np.str_, "TEXT"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.str_)
        self.assertEqual(dt.type_name(), "OptTensorType[strings, 'TEXT']")
        self.assertEmpty(dt.shape)

        self.assertRaise(lambda: TensorType[None], TypeError)
        self.assertRaise(lambda: TensorType[{np.float32, np.str_}], TypeError)

    def test_superset(self):
        t1 = TensorType[ElemType.numerics, "T"]
        t2 = TensorType[ElemType.float64, "F64"]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[None]
        t2 = Float32[None]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[5]
        t2 = Float32[5]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[None]
        t2 = Float32[5]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32["N"]
        t2 = Float32[5]
        self.assertTrue(t1.issuperset(t2))
        t1 = TensorType[ElemType.int64, "I"]
        t2 = Int64[1]
        self.assertTrue(t1.issuperset(t2))

    def test_sig(self):
        def local1(
            x: TensorType[ElemType.floats, "T"],
        ) -> TensorType[ElemType.floats, "T"]:
            return x

        def local2(
            x: TensorType[ElemType.floats, "T"],
        ) -> TensorType[ElemType.floats, "T"]:
            return x

        def local3(x: Float32["N", 1]) -> Float32["N", 1]:
            return x

        def local4(x: Float64["N", 1]) -> Int64["N", 1]:
            return x

        self.assertNotEmpty(local1)
        self.assertNotEmpty(local2)
        self.assertNotEmpty(local3)
        self.assertNotEmpty(local4)

    def test_numpy_abs(self):
        f = absolute(Input())
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", absolute.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", absolute.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", absolute.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_neg(self):
        f = absolute(negative(Input()))
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(-x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_log1p(self):
        f = log1p(Input())
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([5, 6], dtype=np.float64)
        y = np.log1p(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_neg_constraint_input(self):
        f = absolute(negative(Input()))
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        self.assertRaise(lambda: f.to_onnx(), RuntimeError)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(-x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_two_inputs(self):
        f = absolute(addition(Input(), Input()))
        self.assertIsInstance(f, Var)
        self.assertIn("Signature", addition.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", addition.__doc__)
        self.assertIn("y: TensorType[numerics, 'T']", addition.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", addition.__doc__)
        self.assertRaise(lambda: f.to_onnx(), RuntimeError)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([2.5], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x, "I__1": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_parameter_argmin(self):
        f = argmin(Input())
        self.assertIsInstance(f, Var)
        self.assertIn("Signature", argmin.__doc__)
        self.assertIn("x: TensorType[numerics, 'T'],", argmin.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", argmin.__doc__)
        self.assertIn("axis: OptParType[int],", argmin.__doc__)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        if DEFAULT_OPSET > 18:
            z = np.argmin(x, axis=0)
            self.assertEqualArray(z, got[0])
        else:
            # bug in onnx==1.13
            self._warns.append(
                (
                    __file__,
                    inspect.getframeinfo(inspect.currentframe()).lineno,
                    "ReferenceEvaluator:test_numpy_parameter_argmin: "
                    "axis not taken into account",
                )
            )
            self.assertIn(0, got[0].ravel().tolist())

    def test_numpy_relu(self):
        f = relu(Input())
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        z = np.where(x >= 0, x, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_numpy_concat2(self):
        f = concat(Input(), Input())
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        z = np.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"I__0": x1, "I__1": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2_inline(self):
        f = concat_inline(Input("A"), Input("B"))
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        z = np.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_numpy_concat1_2(self):
        f = concat(Input(), concat(Input(), Input()))
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        z = np.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onx)
        feeds = {"I__2": x1, "I__0": x2, "I__1": x3}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_numpy_concat1_2_names(self):
        f = concat(Input("A"), concat(Input("B"), Input("C")))
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        z = np.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2, "C": x3}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_numpy_concat2_2(self):
        f = concat(
            concat(Input("A"), Input("B")), concat(Input("C"), Input("D"), Input("E"))
        )
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        x4 = np.array([[10, 20]], dtype=np.float64)
        x5 = np.array([[100, 200]], dtype=np.float64)
        z = np.vstack([x1, x2, x3, x4, x5])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2, "C": x3, "D": x4, "E": x5}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_abs_a0(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_a0_true(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={(0, True): Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_aN(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None], "r__0": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_inline(self):
        f = absolute_inline(Input())
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", absolute.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", absolute.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", absolute.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        self.assertNotIn("functions {", str(onx))
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_addition_op(self):
        f = absolute(addition(copy(Input("A")), Input("B")))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_inline(self):
        f = absolute_inline(copy_inline(Input("A")) + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator(self):
        f = absolute(copy(Input("A")) + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_input_inline(self):
        f = absolute_inline(Input("A") + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_input(self):
        f = absolute(Input("A") + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_backend_0(self):
        def impl(A, B):
            return absolute_inline(copy_inline(A) + B)

        f = impl(Input("A"), Input("B"))

        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x, y)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64), y.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_backend_1(self):
        def impl(A, B):
            return absolute(copy(A) + B)

        f = impl(Input("A"), Input("B"))

        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x, y)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64), y.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_backend_parameters(self):
        def impl(A, axis=1):
            return argmin_inline(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_backend_parameters_xapi(self):
        @npxapi_inline
        def impl(A, axis=1):
            return argmin_inline(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_backend_parameters_no_inline(self):
        def impl(A, axis=1):
            return argmin(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x}
        got = ref.run(None, feeds)
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_backend_parameters_no_inline_xapi(self):
        @npxapi_function
        def impl(
            A: TensorType[ElemType.numerics, "T"],
            axis: OptParType[int] = 1,
        ) -> TensorType[ElemType.numerics, "T"]:
            return argmin(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x}
        got = ref.run(None, feeds)
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertIsInstance(f.versions, dict)
        self.assertEqual(len(f.versions), 1)
        res = f(x, axis=0)
        self.assertEqual(len(f.versions), 2)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqual(len(f.versions), 3)
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqual(len(f.versions), 4)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

        # versions
        self.assertIsInstance(f.onxs, dict)
        self.assertEqual(len(f.onxs), 4)
        keys = list(sorted(f.onxs))
        self.assertIsInstance(f.onxs[keys[0]], ModelProto)
        k = keys[-1]
        self.assertEqual(len(k), 4)
        self.assertEqual(k[1:], ("axis", int, 0))

    def test_numpy_topk(self):
        f = topk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", topk.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", topk.__doc__)
        self.assertIn("k: TensorType['int64', (1,), 'I']", topk.__doc__)
        self.assertIn(
            ") -> TupleType[TensorType[numerics, 'T'], TensorType['int64', 'I']]",
            topk.__doc__,
        )
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

    def test_numpy_topk_function(self):
        def mytopk(x, k):
            f = topk(x, k)
            return f

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

        f = jit_onnx(topk)
        res = f(x, k)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(y, res[0])
        self.assertEqualArray(z, res[1])

    def test_numpy_topk_function_indices(self):
        def mytopk(x, k):
            f = topk(x, k)
            return f[1]

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={"X": Float64[None], "K": Int64[1], (0, False): Int64[None]}
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    def test_numpy_topk_inline(self):
        f = topk_inline(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", topk.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", topk.__doc__)
        self.assertIn("k: TensorType['int64', (1,), 'I']", topk.__doc__)
        self.assertIn(
            ") -> TupleType[TensorType[numerics, 'T'], TensorType['int64', 'I']]",
            topk.__doc__,
        )
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

    def test_numpy_topk_function_inline(self):
        def mytopk(x, k):
            f = topk_inline(x, k)
            return f

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

        f = jit_onnx(topk)
        res = f(x, k)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(y, res[0])
        self.assertEqualArray(z, res[1])

    def test_numpy_topk_function_indices_inline(self):
        def mytopk(x, k):
            f = topk_inline(x, k)
            return f[1]

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={"X": Float64[None], "K": Int64[1], (0, False): Int64[None]}
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    def test_numpy_min_max(self):
        def myf(x):
            f = _min_max(x)
            return f

        f = myf(Input("X"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                (0, False): Float64[None],
                (1, False): Float64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        z1 = np.array([[-7]], dtype=np.float64)
        z2 = np.array([[7]], dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(z1, got[0])
        self.assertEqualArray(z2, got[1])

        f = jit_onnx(myf)
        res = f(x)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(z1, res[0])
        self.assertEqualArray(z2, res[1])

    def test_numpy_min_max_inline(self):
        def myf(x):
            f = _min_max_inline(x)
            return f

        f = myf(Input("X"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                (0, False): Float64[None],
                (1, False): Float64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        z1 = np.array([[-7]], dtype=np.float64)
        z2 = np.array([[7]], dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(z1, got[0])
        self.assertEqualArray(z2, got[1])

        f = jit_onnx(myf)
        res = f(x)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(z1, res[0])
        self.assertEqualArray(z2, res[1])

    def test_eager_numpy_type(self):
        def impl(A):
            self.assertIsInstance(A, EagerNumpyTensor)
            b = absolute(A)
            self.assertIsInstance(b, EagerNumpyTensor)
            c = absolute_inline(A)
            self.assertIsInstance(c, EagerNumpyTensor)
            return c

        e = eager_onnx(impl)
        self.assertEqual(len(e.versions), 0)

        # Float64
        x = np.array([0, 1, -2], dtype=np.float64)
        z = np.abs(x)
        res = e(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # again
        res = e(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

    def test_eager_numpy_type_op(self):
        def impl(A):
            self.assertIsInstance(A, EagerNumpyTensor)
            b = absolute(A) + A
            self.assertIsInstance(b, EagerNumpyTensor)
            return b

        e = eager_onnx(impl)
        self.assertEqual(len(e.versions), 0)

        # Float64
        x = np.array([0, 1, -2], dtype=np.float64)
        z = np.abs(x) + x
        res = e(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # again
        res = e(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

    def test_eager_numpy(self):
        def impl(A):
            print("A")
            b = absolute(A)
            print("B")
            c = b - A
            print("C")
            return c

        with redirect_stdout(StringIO()):
            f = impl(Input("A"))
            onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x) - x
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        with redirect_stdout(StringIO()):
            res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        with redirect_stdout(StringIO()):
            res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

        e = eager_onnx(impl)

        # Float64
        s = StringIO()
        with redirect_stdout(s):
            res = e(x)
        text = s.getvalue()
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)
        self.assertStartsWith("A\nB\nC\n", text)

        # Int64
        s = StringIO()
        with redirect_stdout(s):
            res = e(x.astype(np.int64))
        text = s.getvalue()
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqual("A\nB\nC\n", text)

    def common_numpy_op(self, msg, fct, use_int=False):
        if use_int:
            dtype = np.int64
            otype = Float64
        else:
            dtype = np.float64
            otype = Int64
        with self.subTest(msg=msg, op=fct):
            f = copy(fct(copy(Input("A")), Input("B")))
            self.assertIsInstance(f, Var)
            onx = f.to_onnx(constraints={"A": otype[None], "B": otype[None]})
            x = np.array([-5, 6], dtype=dtype)
            y = np.array([15, -16], dtype=dtype)
            z = fct(x, y)
            ref = ReferenceEvaluator(onx)
            got = ref.run(None, {"A": x, "B": y})
            try:
                self.assertEqualArray(z, got[0])
            except AssertionError as e:
                # with open("debug_bin.onnx", "wb") as f:
                #     f.write(onx.SerializeToString())
                raise AssertionError(f"Discrepancies with\n{onx}") from e

    def test_numpy_op_op(self):
        self.common_numpy_op("+", lambda x, y: x + y)
        self.common_numpy_op("-", lambda x, y: x - y)
        self.common_numpy_op("*", lambda x, y: x * y)
        self.common_numpy_op("/", lambda x, y: x / y)
        self.common_numpy_op("@", lambda x, y: x @ y)
        self.common_numpy_op("%", lambda x, y: x % y, True)

    def test_numpy_op_cmp(self):
        self.common_numpy_op("<", lambda x, y: x < y)
        self.common_numpy_op("<=", lambda x, y: x <= y)
        self.common_numpy_op(">", lambda x, y: x > y)
        self.common_numpy_op(">=", lambda x, y: x >= y)
        self.common_numpy_op("==", lambda x, y: x == y)
        self.common_numpy_op("!=", lambda x, y: x != y)

    def test_numpy_op_neg(self):
        self.common_numpy_op("-", lambda x, y: (-x) != y)

    def test_numpy_op_shift(self):
        self.common_numpy_op("<<", lambda x, y: x << y, True)
        self.common_numpy_op(">>", lambda x, y: x >> y, True)

    def test_numpy_op_bit(self):
        self.common_numpy_op("&", lambda x, y: x & y, True)
        self.common_numpy_op("|", lambda x, y: x | y, True)
        self.common_numpy_op("|", lambda x, y: x ^ y, True)
        self.common_numpy_op("~", lambda x, y: (~x) | y, True)

    def common_numpy_op_right(self, msg, fct, use_int=False):
        if use_int:
            dtype = np.int64
            otype = Float64
        else:
            dtype = np.float64
            otype = Int64
        if msg == "@":
            ccc = np.array([[1, 1]], dtype=dtype).T
            x = np.array([[-5, 6]], dtype=dtype)
        else:
            ccc = 1
            x = np.array([-5, 6], dtype=dtype)
        with self.subTest(msg=msg, op=fct):
            z = fct(ccc, x)
            f = copy(fct(ccc, copy(Input("A"))))
            self.assertIsInstance(f, Var)
            onx = f.to_onnx(constraints={"A": otype[None]})
            ref = ReferenceEvaluator(onx)
            got = ref.run(None, {"A": x})
            try:
                self.assertEqualArray(z, got[0])
            except AssertionError as e:
                # with open("debug_bin.onnx", "wb") as f:
                #     f.write(onx.SerializeToString())
                raise AssertionError(f"Discrepancies with\n{onx}") from e

    def test_numpy_op_op_right(self):
        self.common_numpy_op_right("+", lambda x, y: x + y)
        self.common_numpy_op_right("-", lambda x, y: x - y)
        self.common_numpy_op_right("*", lambda x, y: x * y)
        self.common_numpy_op_right("/", lambda x, y: x / y)
        self.common_numpy_op_right("%", lambda x, y: x % y, True)
        self.common_numpy_op_right("<", lambda x, y: x < y)
        self.common_numpy_op_right("<=", lambda x, y: x <= y)
        self.common_numpy_op_right(">", lambda x, y: x > y)
        self.common_numpy_op_right(">=", lambda x, y: x >= y)
        self.common_numpy_op_right("==", lambda x, y: x == y)
        self.common_numpy_op_right("!=", lambda x, y: x != y)
        self.common_numpy_op_right("&", lambda x, y: x & y, True)
        self.common_numpy_op_right("|", lambda x, y: x | y, True)
        self.common_numpy_op_right("|", lambda x, y: x ^ y, True)
        self.common_numpy_op_right("~", lambda x, y: (~x) | y, True)

    def test_shape(self):
        f = absolute_inline(Input("A").reshape(copy_inline(Input("A")).shape))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x.reshape(x.shape))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_shape_t(self):
        f = absolute_inline(Input("A").reshape(copy_inline(Input("A")).T.shape))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.reshape(x.T.shape))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_astype(self):
        f = absolute_inline(copy_inline(Input("A")).astype(DType(TensorProto.FLOAT)))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.astype(np.float32))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_astype_dtype(self):
        f = absolute_inline(copy_inline(Input("A")).astype(DType(7)))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5.4, 6.6]], dtype=np.float64)
        z = np.abs(x.astype(np.int64))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_astype_int(self):
        f = absolute_inline(copy_inline(Input("A")).astype(DType(1)))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.astype(np.float32))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_sum(self):
        f = absolute_inline(copy_inline(Input("A")).sum())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.sum())
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_copy(self):
        f = absolute_inline(Input("A").copy())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_flatten(self):
        f = absolute_inline(Input("A").flatten())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6], [-5, 6]], dtype=np.float64)
        z = np.abs(x.flatten())
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_sum_axis(self):
        f = absolute_inline(copy_inline(Input("A")).sum(axis=1, keepdims=1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.sum(axis=1, keepdims=1))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_numpy_op_bin_reduce(self):
        self.common_numpy_op(
            "and", lambda x, y: (x.sum() == y.sum()) & (((-x).sum()) == y.sum())
        )
        self.common_numpy_op(
            "or", lambda x, y: (x.sum() == y.sum()) | (((-x).sum()) == y.sum())
        )
        self.common_numpy_op(
            "xor", lambda x, y: (x.sum() == y.sum()) ^ (((-x).sum()) == y.sum())
        )

    def common_test_inline(self, fonx, fnp, tcst=0):
        f = fonx(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, 0.2], dtype=np.float64)
        x = x + tcst
        y = fnp(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0], atol=1e-10)

    def common_test_inline_bin(self, fonx, fnp, tcst=0):
        f = fonx(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Float64[None], 1: Float64[None], (0, False): Float64[None]}
        )
        x = np.array([[0.1, 0.2], [0.6, 10]], dtype=np.float64)
        y = np.array([[-1, 2], [-0.7, 0.1]], dtype=np.float64)
        x = x + tcst
        z = fnp(x, y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0], atol=1e-10)

    def test_arccos(self):
        self.common_test_inline(arccos_inline, np.arccos)

    def test_arccosh(self):
        self.common_test_inline(arccosh_inline, np.arccosh, tcst=1)

    def test_arcsin(self):
        self.common_test_inline(arcsin_inline, np.arcsin)

    def test_arcsinh(self):
        self.common_test_inline(arcsinh_inline, np.arcsinh)

    def test_arctan(self):
        self.common_test_inline(arctan_inline, np.arctan)

    def test_arctanh(self):
        self.common_test_inline(arctanh_inline, np.arctanh)

    def test_ceil(self):
        self.common_test_inline(ceil_inline, np.ceil)

    def test_clip(self):
        # 1
        f = clip_inline(Input("A"), cst(0), cst(1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, 0, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

        # 2
        f = clip_inline(Input("A"), cst(0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, 0, None)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_clip_int(self):
        f = clip_inline(Input("A"), 0, 1)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, 0, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_clip_none(self):
        f = clip_inline(Input("A"), None, cst(0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, None, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    @skipif_ci_windows("Unstable on Windows.")
    def test_arange_inline(self):
        # arange(5)
        f = arange_inline(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[None], (0, False): Int64[None]})
        x = np.array(5, dtype=np.int64)
        y = np.arange(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

        # arange(1, 5)
        f = arange_inline(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[1], 1: Int64[1], (0, False): Int64[None]})
        x1 = np.array(1, dtype=np.int64)
        x2 = np.array(5, dtype=np.int64)
        y = np.arange(x1, x2)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x1, "B": x2})
        self.assertEqualArray(y, got[0])

        # arange(1, 5, 2)
        f = arange_inline(Input("A"), Input("B"), Input("C"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Int64[1], 1: Int64[1], 2: Int64[1], (0, False): Int64[None]}
        )
        x1 = np.array(1, dtype=np.int64)
        x2 = np.array(5, dtype=np.int64)
        x3 = np.array(2, dtype=np.int64)
        y = np.arange(x1, x2, x3)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x1, "B": x2, "C": x3})
        self.assertEqualArray(y, got[0])

    @skipif_ci_windows("Unstable on Windows.")
    def test_arange_inline_dtype(self):
        # arange(1, 5, 2), dtype
        f = arange_inline(Input("A"), Input("B"), Input("C"), dtype=np.float64)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Int64[1], 1: Int64[1], 2: Int64[1], (0, False): Int64[None]}
        )
        x1 = np.array(1, dtype=np.int64)
        x2 = np.array(5, dtype=np.int64)
        x3 = np.array(2, dtype=np.int64)
        y = np.arange(x1, x2, x3, dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x1, "B": x2, "C": x3})
        self.assertEqual(y.dtype, got[0].dtype)
        self.assertEqualArray(y, got[0])

    def test_cos(self):
        self.common_test_inline(cos_inline, np.cos)

    def test_cosh(self):
        self.common_test_inline(cosh_inline, np.cosh)

    def test_compress_float32(self):
        x = np.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=np.float32)
        cond = np.array([False, True])

        axes = [0, 1, None]
        for axis in axes:
            with self.subTest(axis=axis):
                z = np.compress(cond, x, axis=axis)
                f = compress_inline(Input("A"), Input("B"), axis=axis)
                onx = f.to_onnx(constraints={"A": Bool[None], "B": Float32[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": cond, "B": x})
                self.assertEqualArray(z, got[0])

    def test_cumsum(self):
        x = np.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=np.float32)
        axis = np.array([1])

        z = np.cumsum(x, axis[0])
        f = cumsum_inline(Input("A"), Input("B"))
        onx = f.to_onnx(constraints={"A": Float32[None], "B": Int64[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": axis})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_cumsum_no_axis(self):
        x = np.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=np.float32)

        z = np.cumsum(x)
        f = cumsum_inline(Input("A"))
        onx = f.to_onnx(constraints={"A": Float32[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_det(self):
        self.common_test_inline(det_inline, np.linalg.det, tcst=np.identity(2))

    def test_dot(self):
        self.common_test_inline_bin(dot_inline, np.dot)

    def test_einsum(self):
        equation = "ij,jk->ik"
        self.common_test_inline_bin(
            lambda x, y: einsum_inline(x, y, equation=equation),
            lambda x, y: np.einsum(equation, x, y),
        )

    def test_equal(self):
        self.common_test_inline_bin(equal_inline, np.equal)

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_erf(self):
        self.common_test_inline(erf_inline, scipy.special.erf)

    def test_exp(self):
        self.common_test_inline(exp_inline, np.exp)

    def test_expand_dims(self):
        f = expand_dims_inline(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Float64[None], 1: Int64[None], (0, False): Float64[None]}
        )
        x = np.array([[0.1, 0.2], [0.6, 10]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.int64)
        z = np.expand_dims(x, tuple(y))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_expit(self):
        self.common_test_inline(expit_inline, scipy.special.expit)

    def test_floor(self):
        self.common_test_inline(floor_inline, np.floor)

    def test_hstack(self):
        f = hstack_inline(Input("A"), Input("B"))
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2], [10, 20]], dtype=np.float64)
        z = np.hstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_identity(self):
        f = identity_inline(n=2, dtype=np.float64)
        onx = f.to_onnx(constraints={(0, False): Float64[None]})
        self.assertIn('name: "dtype"', str(onx))
        z = np.identity(2).astype(np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {})
        self.assertEqualArray(z, got[0])

    def test_identity_uint8(self):
        f = identity_inline(n=2, dtype=np.uint8)
        onx = f.to_onnx(constraints={(0, False): Float64[None]})
        self.assertIn('name: "dtype"', str(onx))
        z = np.identity(2).astype(np.uint8)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {})
        self.assertEqualArray(z, got[0])

    def test_isnan(self):
        self.common_test_inline(isnan_inline, np.isnan)

    def test_log(self):
        self.common_test_inline(log_inline, np.log)

    def test_log1p(self):
        self.common_test_inline(log1p_inline, np.log1p)

    def test_matmul(self):
        self.common_test_inline_bin(matmul_inline, np.matmul)

    def test_pad_1(self):
        x = np.random.randn(1, 3, 4, 5).astype(np.float64)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
        value = np.array(1.2, dtype=np.float64)

        for mode in ["constant", "reflect", "edge", "wrap"]:
            with self.subTest(mode=mode):
                z = pad_impl(x, pads, mode, 1.2)
                f = pad_inline(
                    copy_inline(Input("A")), cst(pads), cst(value), mode=mode
                )
                self.assertIsInstance(f, Var)
                onx = f.to_onnx(constraints={"A": Float64[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x})
                self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_pad_2(self):
        x = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
        value = np.array(1.2, dtype=np.float64)
        axes = np.array([1, 2, 3, 4], dtype=np.int64)

        for mode in ["constant", "reflect", "edge"]:
            with self.subTest(mode=mode):
                z = pad_impl(x, pads, mode, value, axes)
                f = pad_inline(
                    copy_inline(Input("A")), cst(pads), cst(value), cst(axes), mode=mode
                )
                self.assertIsInstance(f, Var)
                onx = f.to_onnx(constraints={"A": Float64[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x})
                self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_pad_3(self):
        x = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
        axes = np.array([1, 2, 3, 4], dtype=np.int64)

        for mode in ["constant", "reflect", "edge"]:
            with self.subTest(mode=mode):
                z = pad_impl(x, pads, mode, 0, axes)
                f = pad_inline(
                    copy_inline(Input("A")), cst(pads), None, cst(axes), mode=mode
                )
                self.assertIsInstance(f, Var)
                onx = f.to_onnx(constraints={"A": Float64[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x})
                self.assertEqualArray(z, got[0])

    def common_reduce(self, fct):
        f = absolute_inline(fct(copy_inline(Input("A"))))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(fct(x))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(DEFAULT_OPSET < 19, reason="ReferenceEvaluator bugged.")
    def test_reduce_sum(self):
        self.common_reduce(lambda x: x.sum())

    def test_reduce_mean(self):
        self.common_reduce(lambda x: x.mean())

    def test_reduce_min(self):
        self.common_reduce(lambda x: x.min())

    def test_reduce_max(self):
        self.common_reduce(lambda x: x.max())

    def test_reduce_prod(self):
        self.common_reduce(lambda x: x.prod())

    def test_relu(self):
        self.common_test_inline(relu_inline, lambda x: np.where(x > 0, x, 0))

    def test_reciprocal(self):
        self.common_test_inline(reciprocal_inline, np.reciprocal)

    def test_round(self):
        self.common_test_inline(round_inline, np.round)

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_sigmoid(self):
        self.common_test_inline(sigmoid_inline, scipy.special.expit)

    def test_sign(self):
        self.common_test_inline(sign_inline, np.sign)

    def test_sin(self):
        self.common_test_inline(sin_inline, np.sin)

    def test_sinh(self):
        self.common_test_inline(sinh_inline, np.sinh)

    def test_sqrt(self):
        self.common_test_inline(sqrt_inline, np.sqrt)

    def test_squeeze(self):
        axis = np.array([1], dtype=np.int64)
        f = squeeze_inline(copy_inline(Input("A")), cst(axis))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64).T
        z = np.squeeze(x, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_squeeze_noaxis(self):
        f = squeeze_inline(copy_inline(Input("A")))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.squeeze(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_tan(self):
        self.common_test_inline(tan_inline, np.tan)

    def test_tanh(self):
        self.common_test_inline(tanh_inline, np.tanh)

    def test_transpose(self):
        f = transpose_inline(copy_inline(Input("A")), perm=(1, 0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64).T
        z = np.transpose(x, (1, 0))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_unsqueeze(self):
        axis = np.array([1], dtype=np.int64)
        f = unsqueeze_inline(copy_inline(Input("A")), cst(axis))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64).T
        z = np.expand_dims(x, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_vstack(self):
        f = vstack_inline(Input("A"), Input("B"))
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2], [10, 20]], dtype=np.float64)
        z = np.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_where(self):
        zero = np.array([0], dtype=np.float64)
        f = where_inline(copy_inline(Input("A")) >= cst(zero), Input("A"), cst(zero))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64).T
        z = np.where(x >= 0, x, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_types(self):
        one = np.array([1], dtype=np.float64)

        def impl(x):
            return absolute_inline(copy_inline(x) + cst(one))

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_numpy_operator_types_array(self):
        one = np.array([1], dtype=np.float64)

        def impl(x):
            return absolute_inline(copy_inline(x) + one)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_numpy_operator_types_int(self):
        one = 1

        def impl(x):
            return absolute_inline(copy_inline(x) + one)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_numpy_operator_types_int_right(self):
        one = 1

        def impl(x):
            return absolute_inline(one + copy_inline(x))

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def common_test_indices_int_tuple_slice(self, indices):
        def impl(x):
            return copy_inline(x)[indices]

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(63).reshape((9, 7)).astype(dtype=np.float64)
        z = x[indices]
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_indices_int_tuple_slice(self):
        self.common_test_indices_int_tuple_slice(1)
        self.common_test_indices_int_tuple_slice((1, 2))
        self.common_test_indices_int_tuple_slice(slice(0, 2))
        self.common_test_indices_int_tuple_slice((slice(0, 2), slice(4, 6)))
        self.common_test_indices_int_tuple_slice((slice(0, 2), 5))
        self.common_test_indices_int_tuple_slice((5, slice(0, 2)))
        self.common_test_indices_int_tuple_slice((5, slice(0, 7, 2)))

    def test_filter(self):
        def impl(x):
            y = copy_inline(x)
            ind = (y == 2) | (y == 8)
            return y[ind]

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(63).reshape((9, 7)).astype(dtype=np.float64)
        z = x[(x == 2) | (x == 8)]
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_int(self):
        def impl(x):
            y = copy_inline(x)
            return y.set[5](-6)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[5] = -6
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_slice(self):
        def impl(x):
            y = copy_inline(x)
            return y.set[5:8](np.array([-6, -7, -8]))

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[5:8] = np.array([-6, -7, -8])
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_where(self):
        def impl(x):
            y = copy_inline(x)
            return y.set[x == 5](-7)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[x == 5] = -7
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_where_set(self):
        def impl(x):
            y = copy_inline(x)
            y[x == 5] = -7
            return y()

        self.assertEmpty(Input("A").current_var_)
        i = Input("A")
        self.assertEqual(id(i), id(i.self_var))
        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[x == 5] = -7
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_where_set_2(self):
        def impl(x):
            y = copy_inline(x)
            y[x == 5] = -7
            return y

        self.assertEmpty(Input("A").current_var_)
        i = Input("A")
        self.assertEqual(id(i), id(i.self_var))
        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[x == 5] = -7
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_cdist(self):
        from scipy.spatial.distance import cdist as scipy_cdist

        for metric in ["euclidean", "sqeuclidean"]:
            with self.subTest(metric=metric):

                def impl(xa, xb, metric=metric):
                    return cdist_inline(xa, xb, metric=metric)

                onx = impl(Input("A"), Input("B"), metric=metric).to_onnx(
                    constraints={
                        "A": Float64[None],
                        "B": Float64[None],
                        (0, False): Float64[None],
                    }
                )
                x = np.arange(10).reshape((5, 2)).astype(dtype=np.float64)
                y = np.arange(14).reshape((7, 2)).astype(dtype=np.float64) * 10
                z = scipy_cdist(x, y, metric=metric)
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x, "B": y})
                self.assertEqualArray(z, got[0])

                f = jit_onnx(impl)

                # Float64
                res = f(x, y)
                self.assertEqualArray(z, res)
                self.assertEqual(res.dtype, np.float64)

                # Float32
                res = f(x.astype(np.float32), y.astype(np.float32))
                self.assertEqualArray(z.astype(np.float32), res)
                self.assertEqual(res.dtype, np.float32)

    def test_onnx_in_var_node_proto(self):
        def impl(xa, xb):
            return xa + xb

        onx_base = impl(Input("A"), Input("B")).to_onnx(
            constraints={
                "A": Float32[None],
                "B": Float32[None],
                (0, False): Float32[None],
            }
        )
        self.assertIn("Add", str(onx_base))

        def impl2(x):
            return compute_inline(
                x,
                cst(np.array([5, 6], dtype=np.float32)).astype(x),
                proto=onx_base.graph.node[0],
            )

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )
        self.assertIn("Add", str(onx))

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = x + np.array([5, 6], dtype=np.float32)
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_onnx_in_var_model_proto(self):
        from scipy.spatial.distance import cdist as scipy_cdist

        metric = "sqeuclidean"

        def impl(xa, xb):
            return cdist_inline(xa, xb, metric=metric)

        onx_base = impl(Input("xa"), Input("xb")).to_onnx(
            constraints={
                "xa": Float32[None],
                "xb": Float32[None],
                (0, False): Float32[None],
            }
        )
        self.assertNotIn("ai.onnx.ml", str(onx_base))

        def impl2(x):
            return compute_inline(
                x,
                cst(np.arange(4).reshape((2, 2)).astype(np.float32)).astype(x),
                proto=onx_base,
                name="mycdist",
            )

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = scipy_cdist(
            x, np.arange(4).reshape((2, 2)).astype(np.float32), metric=metric
        ).astype(np.float32)
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    def test_onnx_in_var_function_proto(self):
        def impl(xa, xb):
            return (xa - xb) ** 2

        onx_base = impl(Input("xa"), Input("xb")).to_onnx(
            constraints={
                "xa": Float32[None],
                "xb": Float32[None],
                (0, False): Float32[None],
            },
            as_function=True,
            name="diff_square",
            domain="local_f",
        )
        self.assertIsInstance(onx_base, FunctionProto)
        self.assertNotIn("ai.onnx.ml", str(onx_base))

        def impl2(x):
            return compute_inline(
                x,
                cst(np.arange(2).reshape((1, 2)).astype(np.float32)).astype(x),
                proto=onx_base,
                name="mycdist",
            )

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = ((x - np.arange(2).reshape((1, 2))) ** 2).astype(np.float32)
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    def test_onnx_in_var_model_proto_if(self):
        def _make_model():
            X = make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
            Z = make_tensor_value_info("Z", TensorProto.UNDEFINED, ["N"])
            one = make_tensor_value_info("one", TensorProto.FLOAT, ["N"])

            graph1 = make_graph([], "then", [], [X])
            graph2 = make_graph([], "else", [], [one])

            graph_def = make_graph(
                [
                    make_node("ReduceSum", ["X"], ["Xred"]),
                    make_node("Constant", [], ["one"], value_floats=[1.0]),
                    make_node("CastLike", ["one", "Xred"], ["one_c"]),
                    make_node("Greater", ["Xred", "one_c"], ["cond"]),
                    make_node(
                        "If", ["cond"], ["Z_c"], then_branch=graph1, else_branch=graph2
                    ),
                    make_node("CastLike", ["Z_c", "X"], ["Z"]),
                ],
                "test",
                [X],
                [Z],
            )

            model_def = make_model(
                graph_def,
                producer_name="npx",
                ir_version=7,
                producer_version="0.1",
                opset_imports=[make_operatorsetid("", 15)],
            )
            return model_def

        def impl2(x):
            return compute_inline(x, proto=_make_model(), name="myif")

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = x
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    def test_kmeans(self):
        def compute_labels(X, centers):
            dist = cdist_inline(X, centers)
            return argmin_inline(dist, axis=1)

        onx = compute_labels(Input("X"), Input("centers")).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
            }
        )

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)

        f = jit_onnx(compute_labels)

        # float64
        res = f(x, centers)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)

    def test_kmeans_distance(self):
        def compute_labels(X, centers):
            dist = cdist_inline(X, centers)
            labels = argmin_inline(dist, axis=1)
            return make_tuple(labels, dist)

        onx = compute_labels(Input("X"), Input("centers")).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)
        self.assertEqual(got[1].dtype, np.float64)

        f = jit_onnx(compute_labels)

        # float64
        res, dist = f(x, centers)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)
        self.assertEqualArray(got[1], dist)

    def test_kmeans_distance_calls(self):
        def build_distance(X, centers, use_sqrt=False):
            dist = cdist_inline(X, centers, metric="sqeuclidean")
            if use_sqrt:
                return sqrt_inline(dist)
            return dist

        def compute_labels(X, centers):
            dist = build_distance(X, centers, True)
            labels = argmin_inline(dist, axis=1)
            return make_tuple(labels, dist)

        onx = compute_labels(Input("X"), Input("centers")).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )
        self.assertIn('"Sqrt"', str(onx))

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)
        self.assertEqual(got[1].dtype, np.float64)

        f = jit_onnx(compute_labels)
        self.assertEqual(len(f.onxs), 0)
        self.assertEqual(f.n_versions, 0)

        # float64
        res, dist = f(x, centers)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)
        self.assertEqualArray(got[1], dist)
        self.assertEqual(f.n_versions, 1)
        self.assertEqual(len(f.available_versions), 1)
        self.assertEqual(f.available_versions, [((np.float64, 2), (np.float64, 2))])
        key = ((DType(TensorProto.DOUBLE), 2), (DType(TensorProto.DOUBLE), 2))
        onx = f.get_onnx(key)
        self.assertIsInstance(onx, ModelProto)
        self.assertRaise(lambda: f.get_onnx(2), ValueError)
        onx = f.get_onnx()
        self.assertIsInstance(onx, ModelProto)

    def test_kmeans_distance_calls_args(self):
        def build_distance(X, centers, use_sqrt=False):
            dist = cdist_inline(X, centers, metric="sqeuclidean")
            if use_sqrt:
                return sqrt_inline(dist)
            return dist

        def compute_labels(X, centers, use_sqrt=False):
            dist = build_distance(X, centers, use_sqrt)
            labels = argmin_inline(dist, axis=1)
            return make_tuple(labels, dist)

        onx = compute_labels(Input("X"), Input("centers"), use_sqrt=False).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )
        self.assertNotIn('"Sqrt"', str(onx))

        onx = compute_labels(Input("X"), Input("centers"), use_sqrt=True).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )
        self.assertIn('"Sqrt"', str(onx))

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)
        self.assertEqual(got[1].dtype, np.float64)

        f = jit_onnx(compute_labels)
        self.assertEqual(len(f.onxs), 0)
        self.assertEqual(f.n_versions, 0)

        # float64
        res, dist = f(x, centers, use_sqrt=True)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)
        self.assertEqualArray(got[1], dist)
        self.assertEqual(f.n_versions, 1)
        self.assertEqual(len(f.available_versions), 1)
        key = (
            (DType(TensorProto.DOUBLE), 2),
            (DType(TensorProto.DOUBLE), 2),
            "use_sqrt",
            bool,
            True,
        )
        self.assertEqual(f.available_versions, [key])
        onx = f.get_onnx(key)
        self.assertIsInstance(onx, ModelProto)
        self.assertRaise(lambda: f.get_onnx(2), ValueError)
        onx = f.get_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn('"Sqrt"', str(onx))

    def test_eager_cst(self):
        def l2_loss(x, y):
            d = x - y
            return d ** cst(2)

        jitted_myloss = jit_onnx(l2_loss)
        eager_myloss = eager_onnx(l2_loss)

        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

        res_jit = jitted_myloss(x, y)
        res_eager = eager_myloss(x, y)
        assert_allclose(res_jit, res_eager)

    def test_eager_cst_nocst(self):
        def l2_loss(x, y):
            return (x - y) ** 2

        jitted_myloss = jit_onnx(l2_loss)
        eager_myloss = eager_onnx(l2_loss)

        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

        res_jit = jitted_myloss(x, y)
        res_eager = eager_myloss(x, y)
        assert_allclose(res_jit, res_eager)

    def test_eager_cst_index(self):
        def l1_loss(x, y):
            return absolute_inline(x - y).sum()

        def l2_loss(x, y):
            return ((x - y) ** 2).sum()

        def myloss(x, y):
            x0 = x[:, 0]
            a = l1_loss(x0, y[:, 0])
            b = l2_loss(x[:, 1], y[:, 1])
            return a + b

        jitted_myloss = jit_onnx(myloss)
        eager_myloss = eager_onnx(myloss)

        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

        res_jit = jitted_myloss(x, y)
        res_eager = eager_myloss(x, y)
        assert_allclose(res_jit, res_eager)

    def test_take(self):
        data = np.random.randn(3, 3).astype(np.float32)
        indices = np.array([[0, 2]])
        y = np.take(data, indices, axis=1)

        f = take_inline(Input("A"), Input("B"), axis=1)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None], "B": Int64[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": data, "B": indices})
        self.assertEqualArray(y, got[0])

    def test_numpy_all(self):
        data = np.array([[1, 0], [1, 1]]).astype(np.bool_)
        y = np.all(data, axis=1)

        f = all_inline(Input("A"), axis=1)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Bool[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": data})
        self.assertEqualArray(y, got[0])

    def test_numpy_all_zeros(self):
        data = np.zeros((0,), dtype=np.bool_)
        y = np.all(data)

        f = all_inline(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Bool[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": data})
        self.assertEqualArray(y, got[0])

    @unittest.skipIf(True, reason="ReduceMin does not support shape[axis] == 0")
    def test_numpy_all_zeros_axis_0(self):
        data = np.zeros((0, 1), dtype=np.bool_)
        y = np.all(data, axis=0)

        f = all_inline(Input("A"), axis=0)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Bool[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": data})
        self.assertEqualArray(y, got[0])

    def test_numpy_all_empty_axis_1(self):
        data = np.zeros((0, 1), dtype=np.bool_)
        y = np.all(data, axis=1)

        f = all_inline(Input("A"), axis=1)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Bool[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": data})
        self.assertEqualArray(y, got[0])

    def test_get_item_b(self):
        a = EagerNumpyTensor(np.array([True], dtype=np.bool_))
        i = a[0]
        self.assertEqualArray(i.numpy(), a.numpy()[0])

    def test_get_item_i8(self):
        a = EagerNumpyTensor(np.array([5, 6], dtype=np.int8))
        i = a[0]
        self.assertEqualArray(i.numpy(), a.numpy()[0])

    @ignore_warnings(RuntimeWarning)
    def test_linspace_big_inline(self):
        # linspace(5, 0, 1) --> [5] even with endpoint=True
        f = linspace_inline(Input("A"), Input("B"), Input("C"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                0: Int64[None],
                1: Int64[None],
                2: Int64[None],
                (0, False): Int64[None],
            }
        )

        start = np.array(16777217.0, dtype=np.float64)
        stop = np.array(0.0, dtype=np.float64)
        num = np.array(1, dtype=np.int64)
        y = np.linspace(start, stop, num)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"A": start, "B": stop, "C": num})
        self.assertEqualArray(y, got[0])

    @ignore_warnings(RuntimeWarning)
    def test_linspace_inline(self):
        # linspace(0, 5, 1)
        f = linspace_inline(Input("A"), Input("B"), Input("C"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                0: Int64[None],
                1: Int64[None],
                2: Int64[None],
                (0, False): Int64[None],
            }
        )

        start = np.array(0, dtype=np.float64)
        stop = np.array(5, dtype=np.float64)
        num = np.array(1, dtype=np.int64)
        y = np.linspace(start, stop, num)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"A": start, "B": stop, "C": num})
        self.assertEqualArray(y, got[0])


if __name__ == "__main__":
    TestNpx().test_linspace_inline()
    unittest.main(verbosity=2)
