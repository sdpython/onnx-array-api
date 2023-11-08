import unittest
from typing import Callable, Optional
import numpy as np
from onnx import ModelProto
from onnx.defs import (
    get_all_schemas_with_history,
    onnx_opset_version,
    OpSchema,
    get_schema,
    SchemaError,
)
from onnx.reference import ReferenceEvaluator
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.light_api import start, OnnxGraph, Var
from onnx_array_api.light_api._op_var import OpsVar
from onnx_array_api.light_api._op_vars import OpsVars

OPSET_API = min(19, onnx_opset_version() - 1)


def make_method(schema: OpSchema) -> Optional[Callable]:
    if schema.min_output != schema.max_output:
        return None

    kwargs = []
    names = []
    defaults_none = []
    for v in schema.attributes.values():
        names.append(v.name)
        if v.default_value is None:
            kwargs.append(f"{v.name}=None")
        elif v.type.value == OpSchema.AttrType.FLOAT:
            kwargs.append(f"{v.name}: float={v.default_value.f}")
        elif v.type.value == OpSchema.AttrType.INT:
            kwargs.append(f"{v.name}: int={v.default_value.i}")
        elif v.type.value == OpSchema.AttrType.INTS:
            kwargs.append(f"{v.name}: Optional[List[int]]=None")
            defaults_none.append(
                f"        {v.name} = {v.name} or {v.default_value.ints}"
            )
        elif v.type.value == OpSchema.AttrType.STRING:
            kwargs.append(f"{v.name}: str={v.default_value.s!r}")
        else:
            raise AssertionError(
                f"Operator {schema.domain}:{schema.name} has attribute "
                f"{v.name!r} with type {v.type}."
            )

    if max(schema.min_output, schema.max_output) > 1:
        ann = "Vars"
    else:
        ann = "Var"
    code = [f'    def {schema.name}(self, {", ".join(kwargs)})->"{ann}":']
    if defaults_none:
        code.extend(defaults_none)

    n_inputs = schema.max_input
    eol = ", ".join(f"{n}={n}" for n in names)
    if schema.domain == "":
        if n_inputs == 1:
            code.append(f'        return self.make_node("{schema.name}", self, {eol})')
        else:
            code.append(
                f'        return self.make_node("{schema.name}", *self.vars_, {eol})'
            )
    else:
        raise AssertionError(
            f"Not implemented yet for operator {schema.domain}:{schema.name}."
        )

    return "\n".join(code)


class TestLightApi(ExtTestCase):
    def list_ops_missing(self, n_inputs):
        schemas = {}
        for schema in get_all_schemas_with_history():
            if (
                schema.domain != ""
                or "Sequence" in schema.name
                or "Optional" in schema.name
            ):
                continue
            key = schema.domain, schema.name
            if key not in schemas or schemas[key].since_version < schema.since_version:
                schemas[key] = schema
        expected = set(_[1] for _ in list(sorted(schemas)))
        missing = []
        for ex in expected:
            if (
                not hasattr(Var, ex)
                and not hasattr(OpsVar, ex)
                and not hasattr(OpsVars, ex)
            ):
                missing.append(ex)
        if missing:
            methods = []
            new_missing = []
            for m in sorted(missing):
                try:
                    schema = get_schema(m, OPSET_API)
                except SchemaError:
                    continue
                if m in {
                    "Constant",
                    "ConstantOfShape",
                    "If",
                    "Max",
                    "MaxPool",
                    "Mean",
                    "Min",
                    "StringNormalizer",
                    "Sum",
                    "TfIdfVectorizer",
                    "Unique",
                    # 2
                    "BatchNormalization",
                    "Dropout",
                    "GRU",
                    "LSTM",
                    "LayerNormalization",
                    "Loop",
                    "RNN",
                    "Scan",
                    "SoftmaxCrossEntropyLoss",
                    "Split",
                }:
                    continue
                if schema.min_input == schema.max_input == 1:
                    if n_inputs != 1:
                        continue
                else:
                    if n_inputs == 1:
                        continue
                code = make_method(schema)
                if code is not None:
                    methods.append(code)
                    methods.append("")
                new_missing.append(m)
            text = "\n".join(methods)
            if len(new_missing) > 0:
                raise AssertionError(
                    f"n_inputs={n_inputs}: missing method for operators "
                    f"{new_missing}\n{text}"
                )

    def test_list_ops_missing(self):
        self.list_ops_missing(1)
        self.list_ops_missing(2)

    def test_list_ops_uni(self):
        schemas = {}
        for schema in get_all_schemas_with_history():
            if (
                schema.domain != ""
                or "Sequence" in schema.name
                or "Optional" in schema.name
            ):
                continue
            if (
                schema.min_input
                == schema.max_input
                == 1
                == schema.max_output
                == schema.min_output
                and len(schema.attributes) == 0
            ):
                key = schema.domain, schema.name
                if (
                    key not in schemas
                    or schemas[key].since_version < schema.since_version
                ):
                    schemas[key] = schema
        expected = set(_[1] for _ in list(sorted(schemas)))
        for ex in expected:
            self.assertHasAttr(OpsVar, ex)

    def test_list_ops_bi(self):
        schemas = {}
        for schema in get_all_schemas_with_history():
            if (
                schema.domain != ""
                or "Sequence" in schema.name
                or "Optional" in schema.name
            ):
                continue
            if (
                (schema.min_input == schema.max_input == 2)
                and (1 == schema.max_output == schema.min_output)
                and len(schema.attributes) == 0
            ):
                key = schema.domain, schema.name
                if (
                    key not in schemas
                    or schemas[key].since_version < schema.since_version
                ):
                    schemas[key] = schema
        expected = set(_[1] for _ in list(sorted(schemas)))
        for ex in expected:
            self.assertHasAttr(OpsVars, ex)

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

    def test_exp(self):
        onx = start().vin("X").Exp().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn("Exp", str(onx))
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a})[0]
        self.assertEqualArray(np.exp(a), got)

    def test_transpose(self):
        onx = (
            start()
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

    def test_mul(self):
        onx = start()
        onx = (
            start().vin("X").vin("Y").bring("X", "Y").Mul().rename("Z").vout().to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        a = np.arange(10).astype(np.float32)
        got = ref.run(None, {"X": a, "Y": a + 1})[0]
        self.assertEqualArray(a * (a + 1), got)

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

    def test_operator_float(self):
        for f in [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x, y: x == y,
            lambda x, y: x < y,
            lambda x, y: x <= y,
            lambda x, y: x > y,
            lambda x, y: x >= y,
            lambda x, y: x != y,
            lambda x, y: x @ y,
        ]:
            g = start()
            x = g.vin("X")
            y = g.vin("Y")
            onx = f(x, y).rename("Z").vout().to_onnx()
            self.assertIsInstance(onx, ModelProto)
            ref = ReferenceEvaluator(onx)
            a = np.arange(10).astype(np.float32)
            got = ref.run(None, {"X": a, "Y": a + 1})[0]
            self.assertEqualArray(f(a, a + 1), got)

    def test_operator_int(self):
        for f in [
            lambda x, y: x % y,
            lambda x, y: x**y,
        ]:
            g = start()
            x = g.vin("X", np.int64)
            y = g.vin("Y", np.int64)
            onx = f(x, y).rename("Z").vout(np.int64).to_onnx()
            self.assertIsInstance(onx, ModelProto)
            ref = ReferenceEvaluator(onx)
            a = np.arange(10).astype(np.int64)
            got = ref.run(None, {"X": a, "Y": a + 1})[0]
            self.assertEqualArray(f(a, a + 1), got)

    def test_operator_bool(self):
        for f in [
            lambda x, y: x != y,
        ]:
            g = start()
            x = g.vin("X", np.bool_)
            y = g.vin("Y", np.bool_)
            onx = f(x, y).rename("Z").vout(np.bool_).to_onnx()
            self.assertIsInstance(onx, ModelProto)
            ref = ReferenceEvaluator(onx)
            a = (np.arange(10).astype(np.int64) % 2).astype(np.bool_)
            b = (np.arange(10).astype(np.int64) % 3).astype(np.bool_)
            got = ref.run(None, {"X": a, "Y": b})[0]
            self.assertEqualArray(f(a, b), got)

    def test_topk(self):
        onx = (
            start()
            .vin("X", np.float32)
            .vin("K", np.int64)
            .bring("X", "K")
            .TopK()
            .rename("Values", "Indices")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, ModelProto)
        ref = ReferenceEvaluator(onx)
        x = np.array([[0, 1, 2, 3], [9, 8, 7, 6]], dtype=np.float32)
        k = np.array([2], dtype=np.int64)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(np.array([[3, 2], [9, 8]], dtype=np.float32), got[0])
        self.assertEqualArray(np.array([[3, 2], [0, 1]], dtype=np.int64), got[1])

    def test_topk_reverse(self):
        onx = (
            start()
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


if __name__ == "__main__":
    # TestLightApi().test_topk()
    unittest.main(verbosity=2)
