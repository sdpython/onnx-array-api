import sys
import unittest
from typing import Any, Dict, List, Optional
from difflib import unified_diff
import packaging.version as pv
import numpy
from numpy.testing import assert_allclose
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto, TensorProto, __version__ as onnx_version
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference.op_run import to_array_extended
from onnx.numpy_helper import from_array, to_array
from onnx.backend.base import Device, DeviceType
from onnx_array_api.reference import ExtendedReferenceEvaluator
from onnx_array_api.translate_api.make_helper import make_node_extended
from onnx_array_api.translate_api import translate
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

verbosity = 10 if "-v" in sys.argv or "--verbose" in sys.argv else 0


class ReferenceImplementationError(RuntimeError):
    "Fails, export cannot be compared."
    pass


class ExportWrapper:
    apis = ["onnx", "light"]

    def __init__(self, model):
        self.model = model
        self.expected_sess = ExtendedReferenceEvaluator(self.model, verbose=verbosity)

    @property
    def input_names(self):
        return self.expected_sess.input_names

    @property
    def input_types(self):
        return self.expected_sess.input_types

    @property
    def output_names(self):
        return self.expected_sess.output_names

    @property
    def output_types(self):
        return self.expected_sess.output_types

    def run(
        self, names: Optional[List[str]], feeds: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        try:
            expected = self.expected_sess.run(names, feeds)
        except (RuntimeError, AssertionError, TypeError, KeyError) as e:
            raise ReferenceImplementationError(
                f"ReferenceImplementation fails with {onnx_simple_text_plot(self.model)}"
                f"\n--RAW--\n{self.model}"
            ) from e

        for api in self.apis:
            try:
                code = translate(self.model, api=api)
            except NotImplementedError:
                continue
            except ValueError as e:
                raise AssertionError(
                    f"Unable to translate model for api {api!r}, "
                    f"\n--BASE--\n{onnx_simple_text_plot(self.model)}"
                    f"\n--EXPECTED--\n{expected}"
                ) from e
            try:
                code_compiled = compile(code, "<string>", mode="exec")
            except Exception as e:
                new_code = "\n".join(
                    [f"{i+1:04} {line}" for i, line in enumerate(code.split("\n"))]
                )
                raise AssertionError(f"ERROR {e}\n{new_code}")

            locs = {
                "np": numpy,
                "to_array": to_array,
                "to_array_extended": to_array_extended,
                "from_array": from_array,
                "TensorProto": TensorProto,
                "make_function": make_function,
                "make_opsetid": make_opsetid,
                "make_model": make_model,
                "make_graph": make_graph,
                "make_node": make_node,
                "make_node_extended": make_node_extended,
                "make_tensor_value_info": make_tensor_value_info,
            }
            globs = locs.copy()
            try:
                exec(code_compiled, globs, locs)
            except (TypeError, NameError, ValueError) as e:
                new_code = "\n".join(
                    [f"{i+1:04} {line}" for i, line in enumerate(code.split("\n"))]
                )
                raise AssertionError(
                    f"Unable to executed code for api {api!r}\n{new_code}"
                ) from e
            export_model = locs["model"]
            ref = ExtendedReferenceEvaluator(export_model, verbose=verbosity)
            try:
                got = ref.run(names, feeds)
            except (TypeError, AttributeError) as e:
                diff = "\n".join(
                    unified_diff(
                        str(self.model).split("\n"),
                        str(export_model).split("\n"),
                        fromfile="before",
                        tofile="after",
                    )
                )
                raise AssertionError(
                    f"Unable to run the exported model for api {api!r}, "
                    f"\n--BASE--\n{onnx_simple_text_plot(self.model)}"
                    f"\n--EXP[{api}]--\n{onnx_simple_text_plot(export_model)}"
                    f"\n--CODE--\n{code}"
                    f"\n--FEEDS--\n{feeds}"
                    f"\n--EXPECTED--\n{expected}"
                    f"\n--DIFF--\n{diff}"
                ) from e
            if len(expected) != len(got):
                raise AssertionError(
                    f"Unexpected number of outputs for api {api!r}, "
                    f"{len(expected)} != {len(got)}."
                    f"\n--BASE--\n{onnx_simple_text_plot(self.model)}"
                    f"\n--EXP[{api}]--\n{onnx_simple_text_plot(export_model)}"
                )
            for a, b in zip(expected, got):
                if not isinstance(a, numpy.ndarray):
                    continue
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise AssertionError(
                        f"Shape or type discrepancies for api {api!r}."
                        f"\n--BASE--\n{onnx_simple_text_plot(self.model)}"
                        f"\n--EXP[{api}]--\n{onnx_simple_text_plot(export_model)}"
                    )
                if a.dtype in (numpy.str_, object, numpy.object_) or isinstance(
                    a.dtype, getattr(getattr(numpy, "dtypes", None), "StrDType", type)
                ):
                    if a.tolist() != b.tolist():
                        raise AssertionError(
                            f"Text discrepancies for api {api!r} with a.dtype={a.dtype} "
                            f"and b.dtype={b.dtype}"
                            f"\n--BASE--\n{onnx_simple_text_plot(self.model)}"
                            f"\n--EXP[{api}]--\n{onnx_simple_text_plot(export_model)}"
                        )
                    continue
                try:
                    assert_allclose(a, b, atol=1e-3)
                except (AssertionError, TypeError) as e:
                    raise AssertionError(
                        f"Discrepancies for api {api!r} with a.dtype={a.dtype} "
                        f"and b.dtype={b.dtype} (type-dtype={type(a.dtype)})"
                        f"\n--BASE--\n{onnx_simple_text_plot(self.model)}"
                        f"\n--EXP[{api}]--\n{onnx_simple_text_plot(export_model)}"
                    ) from e

            return expected


class ExportBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.input_names):
                feeds = dict(zip(self._session.input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(
                    self._session.input_names, self._session.input_types
                ):
                    shape = tuple(d.dim_value for d in tshape.tensor_type.shape.dim)
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        return outs


class ExportBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def create_inference_session(cls, model):
        return ExportWrapper(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> ExportBackendRep:
        if isinstance(model, ExportWrapper):
            return ExportBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


backend_test = onnx.backend.test.BackendTest(ExportBackend, __name__)

# The following tests are too slow with the reference implementation (Conv).
backend_test.exclude(
    "(FLOAT8|BFLOAT16|INT4|_opt_|_3d_|_momentum_|_4d_|int4"
    "|test_adagrad"
    "|test_adam"
    "|test_ai_onnx_ml_"
    "|test_cast_FLOAT16"
    "|test_cast_FLOAT_to_STRING"
    "|test_castlike_FLOAT16"
    "|test_castlike_FLOAT_to_STRING"
    "|test_bernoulli"
    "|test_bvlc_alexnet"
    "|test_conv"  # too long
    "|test_gradient_"
    "|test_densenet121"
    "|test_inception_v1"
    "|test_inception_v2"
    "|test_loop11_"
    "|test_loop16_seq_none"
    "|test_MaxPool2d"
    "|test_quantizelinear_e"
    "|test_resnet50"
    "|test_sequence_model"
    "|test_scan_sum"
    "|test_scatter_with_axis"
    "|test_scatter_without_axis"
    "|test_shufflenet"
    "|test_squeezenet"
    "|test_vgg19"
    "|test_zfnet512"
    "|test_range_float_type_positive_delta_expanded"
    "|test_range_int32_type_negative_delta_expanded"
    ")"
)

if pv.Version(onnx_version) < pv.Version("1.16.0"):
    backend_test.exclude("(test_strnorm|test_range_)")

# The following tests cannot pass because they consists in generating random number.
backend_test.exclude("(test_bernoulli)")

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == "__main__":
    res = unittest.main(verbosity=2, exit=False)
    tests_run = res.result.testsRun
    errors = len(res.result.errors)
    skipped = len(res.result.skipped)
    unexpected_successes = len(res.result.unexpectedSuccesses)
    expected_failures = len(res.result.expectedFailures)
    print("---------------------------------")
    print(
        f"tests_run={tests_run} errors={errors} skipped={skipped} "
        f"unexpected_successes={unexpected_successes} "
        f"expected_failures={expected_failures}"
    )
