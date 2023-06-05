from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from onnx import ModelProto
from onnx.helper import np_dtype_to_tensor_dtype
from onnx.reference import ReferenceEvaluator

from .npx_tensors import EagerTensor, JitTensor
from .npx_types import DType, TensorType


class NumpyTensor:
    """
    Default backend based on
    :func:`onnx.reference.ReferenceEvaluator`.

    :param input_names: input names
    :param onx: onnx model
    """

    class Evaluator:
        """
        Wraps class :class:`onnx.reference.ReferenceEvaluator`
        to have a signature closer to python function.
        """

        def __init__(self, tensor_class: type, input_names: List[str], onx: ModelProto):
            self.ref = ReferenceEvaluator(onx)
            self.input_names = input_names
            self.tensor_class = tensor_class

        def run(self, *inputs: List["NumpyTensor"]) -> List["NumpyTensor"]:
            """
            Executes the function.

            :param inputs: function inputs
            :return: outputs
            """
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs but got {len(inputs)}, "
                    f"self.input_names={self.input_names}, inputs={inputs}."
                )
            feeds = {}
            for name, inp in zip(self.input_names, inputs):
                feeds[name] = inp.value
            res = self.ref.run(None, feeds)
            return list(map(self.tensor_class, res))

    def __init__(self, tensor: np.ndarray):
        if isinstance(tensor, np.ndarray):
            self._tensor = tensor
        elif isinstance(tensor, NumpyTensor):
            self._tensor = tensor._tensor
        elif isinstance(
            tensor,
            (
                np.float16,
                np.float32,
                np.float64,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
                np.uint64,
                np.uint32,
                np.uint16,
                np.uint8,
                np.bool_,
            ),
        ):
            self._tensor = np.array(tensor)
        else:
            raise TypeError(f"A numpy array is expected not {type(tensor)}.")

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}({self._tensor!r})"

    def numpy(self):
        "Returns the array converted into a numpy array."
        return self._tensor

    @property
    def dtype(self) -> DType:
        "Returns the element type of this tensor."
        return DType(np_dtype_to_tensor_dtype(self._tensor.dtype))

    @property
    def key(self) -> Any:
        "Unique key for a tensor of the same type."
        return (self.dtype, len(self._tensor.shape))

    @property
    def value(self) -> np.ndarray:
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
        has more than one dimension.
        """
        if len(self._tensor.shape) == 0:
            return (0,)
        if len(self._tensor.shape) == 1:
            return self._tensor.shape
        return (None,) + self._tensor.shape[1:]

    @property
    def ndim(self):
        "Returns the number of dimensions (rank)."
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        "Returns the shape of the tensor."
        return self._tensor.shape

    @property
    def tensor_type_dims(self) -> TensorType:
        """
        Returns the tensor type of this tensor.
        This property is used to define a key used to cache a jitted function.
        Same keys keys means same ONNX graph.
        Different keys usually means same ONNX graph but different
        input shapes.
        """
        return TensorType[self.dtype, self.dims]

    @classmethod
    def create_function(cls: Any, input_names: List[str], onx: ModelProto) -> Callable:
        """
        Creates a python function calling the onnx backend
        used by this class.

        :param onx: onnx model
        :return: python function
        """
        return cls.Evaluator(cls, input_names, onx)

    @classmethod
    def get_opsets(cls, opsets):
        """
        Updates the opsets for a given backend.
        This method should be overloaded.
        By default, it returns opsets.
        """
        return opsets

    @classmethod
    def get_ir_version(cls, ir_version):
        """
        Updates the IR version.
        This method should be overloaded.
        By default, it returns ir_version.
        """
        return ir_version

    # The class should support whatever Var supports.
    # This part is not yet complete.


class EagerNumpyTensor(NumpyTensor, EagerTensor):
    """
    Defines a value for a specific backend.
    """

    def __array_namespace__(self, api_version: Optional[str] = None):
        """
        Returns the module holding all the available functions.
        """
        if api_version is None or api_version == "2022.12":
            from onnx_array_api.array_api import onnx_numpy

            return onnx_numpy
        raise ValueError(
            f"Unable to return an implementation for api_version={api_version!r}."
        )


class JitNumpyTensor(NumpyTensor, JitTensor):
    """
    Defines a value for a specific backend.
    """

    pass
