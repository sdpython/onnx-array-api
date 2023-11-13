import warnings
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from onnx import ModelProto, TensorProto
from ..reference import ExtendedReferenceEvaluator
from .._helpers import np_dtype_to_tensor_dtype
from .npx_tensors import EagerTensor, JitTensor
from .npx_types import DType, TensorType


class NumpyTensor:
    """
    Default backend based on
    :func:`onnx_array_api.reference.ExtendedReferenceEvaluator`.

    :param input_names: input names
    :param onx: onnx model
    """

    class Evaluator:
        """
        Wraps class :class:`onnx_array_api.reference.ExtendedReferenceEvaluator`
        to have a signature closer to python function.

        :param tensor_class: class tensor such as :class:`NumpyTensor`
        :param input_names: input names
        :param onx: onnx model
        :param f: unused except in error messages
        """

        def __init__(
            self,
            tensor_class: type,
            input_names: List[str],
            onx: ModelProto,
            f: Callable,
        ):
            self.ref = ExtendedReferenceEvaluator(onx)
            self.input_names = input_names
            self.tensor_class = tensor_class
            self._f = f

        def run(self, *inputs: List["NumpyTensor"]) -> List["NumpyTensor"]:
            """
            Executes the function.

            :param inputs: function inputs
            :return: outputs
            """
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs but got {len(inputs)}, "
                    f"self.input_names={self.input_names}, "
                    f"inputs={inputs}, f={self._f}."
                )
            feeds = {}
            for name, inp in zip(self.input_names, inputs):
                if inp is None:
                    feeds[name] = None
                    continue
                if not isinstance(inp, (EagerTensor, JitTensor)):
                    raise TypeError(
                        f"Unexpected type {type(inp)} for input {name!r}, "
                        f"inp={inp!r}, f={self._f}."
                    )
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

    def __len__(self):
        "usual"
        return len(self._tensor)

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
        has more than one dimension. It is always left undefined.
        """
        if len(self._tensor.shape) <= 1:
            # a scalar (len==0) or a 1D tensor
            return self._tensor.shape
        return (None, *tuple(self.shape[1:]))

    @property
    def ndim(self):
        "Returns the number of dimensions (rank)."
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        "Returns the shape of the tensor."
        return self._tensor.shape

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

    def __bool__(self):
        "Implicit conversion to bool."
        if self.dtype != DType(TensorProto.BOOL):
            raise TypeError(
                f"Conversion to bool only works for bool scalar, not for {self!r}."
            )
        if self.shape == (0,):
            return False
        if self.shape:
            warnings.warn(
                f"Conversion to bool only works for scalar, not for {self!r}, "
                f"bool(...)={bool(self._tensor)}."
            )
            try:
                return bool(self._tensor)
            except ValueError as e:
                raise ValueError(f"Unable to convert {self} to bool.") from e
        return bool(self._tensor)

    def __int__(self):
        "Implicit conversion to int."
        if self.shape:
            raise ValueError(
                f"Conversion to bool only works for scalar, not for {self!r}."
            )
        if self.dtype not in {
            DType(TensorProto.INT64),
            DType(TensorProto.INT32),
            DType(TensorProto.INT16),
            DType(TensorProto.INT8),
            DType(TensorProto.UINT64),
            DType(TensorProto.UINT32),
            DType(TensorProto.UINT16),
            DType(TensorProto.UINT8),
        }:
            raise TypeError(
                f"Conversion to int only works for int scalar, "
                f"not for dtype={self.dtype}."
            )
        return int(self._tensor)

    def __float__(self):
        "Implicit conversion to float."
        if self.shape:
            raise ValueError(
                f"Conversion to bool only works for scalar, not for {self!r}."
            )
        if self.dtype not in {
            DType(TensorProto.FLOAT),
            DType(TensorProto.DOUBLE),
            DType(TensorProto.FLOAT16),
            DType(TensorProto.BFLOAT16),
        }:
            raise TypeError(
                f"Conversion to float only works for float scalar, "
                f"not for dtype={self.dtype}."
            )
        return float(self._tensor)

    def __iter__(self):
        """
        The :epkg:`Array API` does not define this function (2022/12).
        This method raises an exception with a better error message.
        """
        warnings.warn(
            f"Iterators are not implemented in the generic case. "
            f"Every function using them cannot be converted into ONNX "
            f"(tensors - {type(self)})."
        )
        for row in self._tensor:
            yield self.__class__(row)


class JitNumpyTensor(NumpyTensor, JitTensor):
    """
    Defines a value for a specific backend.
    """

    pass
