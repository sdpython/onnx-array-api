from typing import Any, Optional

import numpy as np

from .npx_types import OptParType, ParType, TupleType


class ArrayApiError(RuntimeError):
    """
    Raised when a function is not supported by the :epkg:`Array API`.
    """

    pass


class BaseArrayApi:
    """
    List of supported method by a tensor.
    """

    def __array_namespace__(self, api_version: Optional[str] = None):
        """
        This method must be overloaded.
        """
        raise NotImplementedError("Method '__array_namespace__' must be implemented.")

    def generic_method(self, method_name, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Method {method_name!r} must be overwritten "
            f"for class {self.__class__.__name__!r}. "
            f"Method 'generic_method' can be overwritten "
            f"as well to change the behaviour "
            f"for all methods supported by class BaseArrayApi."
        )

    def numpy(self) -> np.ndarray:
        return self.generic_method("numpy")

    def __neg__(self) -> "BaseArrayApi":
        return self.generic_method("__neg__")

    def __invert__(self) -> "BaseArrayApi":
        return self.generic_method("__invert__")

    def __add__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__add__", ov)

    def __radd__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__radd__", ov)

    def __sub__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__sub__", ov)

    def __rsub__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rsub__", ov)

    def __mul__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__mul__", ov)

    def __rmul__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rmul__", ov)

    def __matmul__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        res = self.generic_method("__matmul__", ov)
        # TODO: It works with float32 but not float64.
        return res

    def __truediv__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__truediv__", ov)

    def __rtruediv__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rtruediv__", ov)

    def __mod__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__mod__", ov)

    def __rmod__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rmod__", ov)

    def __pow__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__pow__", ov)

    def __rpow__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rpow__", ov)

    def __lt__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__lt__", ov)

    def __le__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__le__", ov)

    def __gt__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__gt__", ov)

    def __ge__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__ge__", ov)

    def __eq__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__eq__", ov)

    def __ne__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__ne__", ov)

    def __lshift__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__lshift__", ov)

    def __rshift__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rshift__", ov)

    def __and__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__and__", ov)

    def __rand__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rand__", ov)

    def __or__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__or__", ov)

    def __ror__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__ror__", ov)

    def __xor__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__xor__", ov)

    def __rxor__(self, ov: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("__rxor__", ov)

    @property
    def T(self) -> "BaseArrayApi":
        return self.generic_method("T")

    def astype(self, dtype: Any) -> "BaseArrayApi":
        return self.generic_method("astype", dtype=dtype)

    @property
    def shape(self) -> "BaseArrayApi":
        return self.generic_method("shape")

    def reshape(self, shape: "BaseArrayApi") -> "BaseArrayApi":
        return self.generic_method("reshape", shape)

    def sum(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "BaseArrayApi":
        return self.generic_method("sum", axis=axis, keepdims=keepdims)

    def mean(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "BaseArrayApi":
        return self.generic_method("mean", axis=axis, keepdims=keepdims)

    def min(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "BaseArrayApi":
        return self.generic_method("min", axis=axis, keepdims=keepdims)

    def max(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "BaseArrayApi":
        return self.generic_method("max", axis=axis, keepdims=keepdims)

    def prod(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "BaseArrayApi":
        return self.generic_method("prod", axis=axis, keepdims=keepdims)

    def copy(self) -> "BaseArrayApi":
        return self.generic_method("copy")

    def flatten(self) -> "BaseArrayApi":
        return self.generic_method("flatten")

    def __getitem__(self, index: Any) -> "BaseArrayApi":
        return self.generic_method("__getitem__", index)

    def __setitem__(self, index: Any, values: Any):
        return self.generic_method("__setitem__", index, values)
