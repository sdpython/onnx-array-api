from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from onnx import ModelProto, TensorProto
from onnx.defs import onnx_opset_version
from onnxruntime import InferenceSession, RunOptions, get_available_providers
from onnxruntime.capi._pybind_state import OrtDevice as C_OrtDevice
from onnxruntime.capi._pybind_state import OrtMemType
from onnxruntime.capi._pybind_state import OrtValue as C_OrtValue
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from ..npx.npx_tensors import EagerTensor, JitTensor
from ..npx.npx_types import DType, TensorType


class OrtTensor:
    """
    Default backend based on
    :class:`onnxruntime.InferenceSession`.
    Data is not copied.

    :param input_names: input names
    :param onx: onnx model
    """

    CPU = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
    CUDA0 = C_OrtDevice(C_OrtDevice.cuda(), OrtMemType.DEFAULT, 0)
    providers = [
        c
        for c in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if c in get_available_providers()
    ]

    @staticmethod
    def from_array(
        value: np.ndarray, device: Optional[C_OrtDevice] = None
    ) -> "OrtTensor":
        """
        Creates an instance of :class:`OrtTensor` from a numpy array.
        Relies on `ortvalue_from_numpy`.
        A copy of the data in the Numpy object is held by the
        :epkg:`C_OrtValue` only if the device is **not cpu**.
        Any expression such as `from_array(x.copy())`, or
        `from_array(x.astype(np.float32))`, ... creates an intermediate
        variable scheduled to be deleted by the garbage collector
        as soon as the function returns. In that case, the buffer
        holding the values is deleted and the instance `OrtTenor`
        is no longer equal to the original value:
        `assert_allclose(value, tensor.numpy())` is false.
        `value` must remain alive as long as the `OrtTensor` is.

        :param value: value
        :param device: CPU, GPU, value such as `OrtTensor.CPU`,
            `OrtTensor.CUDA0`
        :return: instance of :class:`OrtTensor`
        """
        if device is None:
            device = OrtTensor.CPU
        return OrtTensor(C_OrtValue.ortvalue_from_numpy(value, device), _hold=value)

    def numpy(self) -> np.ndarray:
        """
        Converts the :epkg:`OrtValue` into numpy array.
        """
        return self._tensor.numpy()

    class Evaluator:
        """
        Wraps class :class:`onnxruntime.InferenceSession`
        to have a signature closer to python function.

        :param tensor_class: class tensor such as :class:`NumpyTensor
            <onnx_array_api.npx.npx_numpy_tensors.NumpyTensor>`
        :param input_names: input names
        :param onx: onnx model
        :param f: unused except in error messages
        :param _hold: :epkg:`onnxruntime` does not copy the data if it comes
            from a numpy array on CPU it does not hold any reference on it.
            *_hold* is used to stored the underlying numpy array hosting the
            data for an OrtTensor if it comes from it. It ensures
            the garbage collector does not remove it.
        """

        def __init__(
            self,
            tensor_class: type,
            input_names: List[str],
            onx: ModelProto,
            f: Callable = None,
        ):
            try:
                self.ref = InferenceSession(
                    onx.SerializeToString(),
                    providers=tensor_class.providers,
                )
            except InvalidArgument as e:
                if (
                    len(onx.graph.output) == 1
                    and onx.graph.output[0].type.tensor_type.elem_type
                    == TensorProto.UNDEFINED
                ):
                    # ShapeInference cannot use python function for unknown node type.
                    # Let's give the only output the same type as the first
                    # input.
                    onx.graph.output[0].type.tensor_type.elem_type = onx.graph.input[
                        0
                    ].type.tensor_type.elem_type
                    self.ref = InferenceSession(
                        onx.SerializeToString(),
                        providers=tensor_class.providers,
                    )
                else:
                    if len(onx.graph.node) <= 3:
                        raise RuntimeError(
                            f"Unable to create an InferenceSession with model {onx}."
                        ) from e
                    raise e
            self.input_names = input_names
            self.tensor_class = tensor_class
            self.output_names = [output.name for output in self.ref._outputs_meta]
            self.run_options = RunOptions()
            self._f = f

        def run(self, *inputs: List["OrtTensor"]) -> List["OrtTensor"]:
            """
            Executes the function.

            :param inputs: function inputs
            :return: outputs
            """
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs but got "
                    f"len(inputs)={len(inputs)}, f={self._f}."
                )
            feeds = {}
            for name, inp in zip(self.input_names, inputs):
                feeds[name] = inp.value
            res = self.ref._sess.run_with_ort_values(
                feeds, self.output_names, self.run_options
            )
            return list(map(inputs[0].__class__, res))

    def __init__(
        self,
        tensor: Union[C_OrtValue, "OrtTensor", np.ndarray],
        _hold: Optional[np.ndarray] = None,
    ):
        if isinstance(tensor, C_OrtValue):
            self._tensor = tensor
            self._hold = _hold
        elif isinstance(tensor, OrtTensor):
            self._tensor = tensor._tensor
            self._hold = _hold
        elif isinstance(tensor, np.ndarray):
            if _hold is not None:
                raise RuntimeError(
                    "tensor cannot be a numpy array and _hold be not None."
                )
            self._tensor = C_OrtValue.ortvalue_from_numpy(tensor, OrtTensor.CPU)
            self._hold = tensor
        else:
            raise ValueError(f"An OrtValue is expected not {type(tensor)}.")

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(OrtTensor.from_array({self.numpy()!r}))"

    @property
    def device_name(self):
        return self._tensor.device_name()

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        "Returns the shape of the tensor."
        return tuple(self._tensor.shape())

    @property
    def dtype(self) -> DType:
        "Returns the element type of this tensor."
        return DType(self._tensor.element_type())

    @property
    def key(self) -> Any:
        "Unique key for a tensor of the same type."
        return (self.dtype, len(self.shape))

    @property
    def value(self) -> C_OrtValue:
        "Returns the value of this tensor as a numpy array."
        return self._tensor

    @property
    def tensor_type(self) -> TensorType:
        "Returns the tensor type of this tensor."
        return TensorType[self.dtype]

    @property
    def dims(self):
        """
        Returns the dimensions of the tensor.
        First dimension is the batch dimension if the tensor
        has more than one dimension. It is always left undefined.
        """
        if len(self._tensor.shape()) <= 1:
            # a scalar (len==0) or a 1D tensor
            return tuple(self._tensor.shape())
        return (None, *tuple(self.shape[1:]))

    def tensor_type_dims(self, name: str) -> TensorType:
        """
        Returns the tensor type of this tensor.
        This property is used to define a key used to cache a jitted function.
        Same keys keys means same ONNX graph.
        Different keys usually means same ONNX graph but different
        input shapes.

        :param name: name of the constraint
        """
        dt = self.dtype
        return TensorType[dt, self.dims, name]

    @classmethod
    def create_function(
        cls: Any, input_names: List[str], onx: ModelProto, f: Callable
    ) -> Callable:
        """
        Creates a python function calling the onnx backend
        used by this class.

        :param onx: onnx model
        :return: python function
        """
        return cls.Evaluator(cls, input_names, onx, f=f)


class OrtCommon:
    """
    Common methods to jit and eager mode.
    """

    @classmethod
    def get_opsets(cls, opsets):
        if opsets is None:
            return {"": min(onnx_opset_version(), 18), "com.microsoft": 1}
        if "com.microsoft" in opsets:
            return opsets
        opsets = opsets.copy()
        opsets.update({"com.microsoft": 1})
        return opsets

    @classmethod
    def get_ir_version(cls, ir_version):
        if ir_version is None:
            return 8
        return min(ir_version, 8)


class EagerOrtTensor(OrtTensor, OrtCommon, EagerTensor):
    """
    Defines a value for :epkg:`onnxruntime` as a backend.
    """

    def __array_namespace__(self, api_version: Optional[str] = None):
        """
        Returns the module holding all the available functions.
        """
        if api_version is None or api_version == "2022.12":
            from onnx_array_api.array_api import onnx_ort

            return onnx_ort
        raise ValueError(
            f"Unable to return an implementation for api_version={api_version!r}."
        )


class JitOrtTensor(OrtTensor, OrtCommon, JitTensor):
    """
    Defines a value for :epkg:`onnxruntime` as a backend.
    """

    pass
