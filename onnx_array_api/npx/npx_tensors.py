from typing import Any

import numpy as np
from onnx.helper import np_dtype_to_tensor_dtype

from .npx_array_api import ArrayApi


class JitTensor:
    """
    Defines a value for a specific jit mode
    """

    pass


class EagerTensor(ArrayApi):
    """
    Defines a value for a specific eager mode.
    An eager tensor must overwrite every call to a method listed in class
    :class:`ArrayApi`.
    """

    def __iter__(self):
        """
        This is not implementation in the generic case.
        This method raises an exception with a better error message.
        """
        raise RuntimeError(
            "Iterators are not implemented in the generic case. "
            "It may be enabled for the eager mode but it might fail "
            "when a whole function is converted into ONNX."
        )

    @staticmethod
    def _op_impl(*inputs, method_name=None):
        # avoids circular imports.
        from .npx_var import Var

        for i, x in enumerate(inputs):
            if not isinstance(x, Var):
                raise TypeError(f"Input {i} must be a Var not {type(x)}.")
        meth = getattr(Var, method_name)
        return meth(*inputs)

    @staticmethod
    def _reduce_impl(x, axes, keepdims=0, method_name=None):
        # avoids circular imports.
        from .npx_var import Var

        if not isinstance(x, Var):
            raise TypeError(f"Input 0 must be a Var not {type(x)}.")
        meth = getattr(Var, method_name)
        return meth(x, axes, keepdims=keepdims)

    @staticmethod
    def _reduce_impl_noaxes(x, keepdims=0, method_name=None):
        # avoids circular imports.
        from .npx_var import Var

        if not isinstance(x, Var):
            raise TypeError(f"Input 0 must be a Var not {type(x)}.")
        meth = getattr(Var, method_name)
        return meth(x, keepdims=keepdims)

    @staticmethod
    def _getitem_impl_var(obj, index, method_name=None):
        # avoids circular imports.
        from .npx_var import Var

        if not isinstance(obj, Var):
            raise TypeError(f"obj must be a Var not {type(obj)}.")
        meth = getattr(Var, method_name)
        return meth(obj, index)

    @staticmethod
    def _astype_impl(x, dtype: int = None, method_name=None):
        # avoids circular imports.
        if dtype is None:
            raise ValueError("dtype cannot be None.")
        from .npx_var import Var

        if not isinstance(x, Var):
            raise TypeError(f"Input 0 must be a Var not {type(x)}.")
        meth = getattr(Var, "astype")
        return meth(x, dtype)

    @staticmethod
    def _getitem_impl_tuple(obj, index=None, method_name=None):
        # avoids circular imports.
        from .npx_var import Var

        if not isinstance(obj, Var):
            raise TypeError(f"obj must be a Var not {type(obj)}.")
        meth = getattr(Var, method_name)
        return meth(obj, index)

    def _generic_method_getitem(self, method_name, *args: Any, **kwargs: Any) -> Any:
        # avoids circular imports.
        from .npx_jit_eager import eager_onnx

        if len(args) != 1:
            raise ValueError(
                f"Unexpected number of argument {len(args)}, it should be one."
            )
        if isinstance(args[0], tuple):
            eag = eager_onnx(
                EagerTensor._getitem_impl_tuple, self.__class__, bypass_eager=True
            )
            res = eag(self, index=args[0], method_name=method_name, already_eager=True)
        else:
            eag = eager_onnx(
                EagerTensor._getitem_impl_var, self.__class__, bypass_eager=True
            )
            res = eag(self, args[0], method_name=method_name, already_eager=True)
        if isinstance(res, tuple) and len(res) == 1:
            return res[0]
        return res

    def _generic_method_operator(self, method_name, *args: Any, **kwargs: Any) -> Any:
        # avoids circular imports.
        from .npx_jit_eager import eager_onnx

        if len(args) not in (0, 1):
            raise ValueError(
                f"An operator must have zero or one argument not {len(args)}."
            )
        if len(kwargs) not in (0, 1):
            raise ValueError(f"Operators do not support parameters {len(kwargs)}.")

        # let's cast numpy arrays into constants.
        new_args = []
        for a in args:
            if isinstance(a, np.ndarray):
                new_args.append(self.__class__(a.astype(self.dtype)))
            else:
                new_args.append(a)

        eag = eager_onnx(EagerTensor._op_impl, self.__class__, bypass_eager=True)
        res = eag(self, *new_args, method_name=method_name, already_eager=True)
        if isinstance(res, tuple) and len(res) == 1:
            return res[0]
        return res

    def _generic_method_reduce(self, method_name, *args: Any, **kwargs: Any) -> Any:
        # avoids circular imports.
        from .npx_jit_eager import eager_onnx

        if len(args) not in (0, 1):
            raise ValueError(
                f"An operator must have zero or one argument not {len(args)}."
            )

        if "axis" in kwargs:
            axes = kwargs["axis"]
            del kwargs["axis"]
        else:
            axes = None
        if axes is None:
            eag = eager_onnx(
                EagerTensor._reduce_impl_noaxes, self.__class__, bypass_eager=True
            )
            res = eag(self, method_name=method_name, already_eager=True, **kwargs)
        else:
            eag = eager_onnx(
                EagerTensor._reduce_impl, self.__class__, bypass_eager=True
            )
            res = eag(self, axes, method_name=method_name, already_eager=True, **kwargs)
        if isinstance(res, tuple) and len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def _np_dtype_to_tensor_dtype(dtype):
        if dtype == int:
            dtype = np.dtype("int64")
        elif dtype == float:
            dtype = np.dtype("float64")
        return np_dtype_to_tensor_dtype(dtype)

    def _generic_method_astype(self, method_name, *args: Any, **kwargs: Any) -> Any:
        # avoids circular imports.
        from .npx_jit_eager import eager_onnx
        from .npx_var import Var

        if len(args) != 1:
            raise ValueError(f"astype takes only one argument not {len(args)}.")

        dtype = (
            args[0]
            if isinstance(args[0], (int, Var))
            else self._np_dtype_to_tensor_dtype(args[0])
        )
        eag = eager_onnx(EagerTensor._astype_impl, self.__class__, bypass_eager=True)
        res = eag(self, dtype, method_name=method_name, already_eager=True, **kwargs)
        if isinstance(res, tuple) and len(res) == 1:
            return res[0]
        return res

    def generic_method(self, method_name, *args: Any, **kwargs: Any) -> Any:
        """
        The method converts the method into an ONNX graph build by the
        corresponding method in class Var.
        """
        # avoids circular imports.
        from .npx_var import Var

        if not hasattr(Var, method_name):
            raise AttributeError(
                f"Class Var does not implement method {method_name!r}. "
                f"This method cannot be converted into an ONNX graph."
            )
        if method_name == "__getitem__":
            return self._generic_method_getitem(method_name, *args, **kwargs)

        if method_name == "__setitem__":
            return ArrayApi.generic_method(self, method_name, *args, **kwargs)

        if method_name in {"mean", "sum", "min", "max", "prod"}:
            return self._generic_method_reduce(method_name, *args, **kwargs)

        if method_name == "astype":
            return self._generic_method_astype(method_name, *args, **kwargs)

        if method_name.startswith("__") and method_name.endswith("__"):
            return self._generic_method_operator(method_name, *args, **kwargs)

        return ArrayApi.generic_method(self, method_name, *args, **kwargs)
