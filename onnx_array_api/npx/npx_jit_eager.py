from inspect import signature
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .npx_tensors import EagerTensor, JitTensor
from .npx_types import TensorType
from .npx_var import Cst, Input, Var

logger = getLogger("onnx-array-api")


class JitEager:
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    :param ir_version: defines the IR version to use
    """

    def __init__(
        self,
        f: Callable,
        tensor_class: type,
        target_opsets: Optional[Dict[str, int]] = None,
        output_types: Optional[Dict[Any, TensorType]] = None,
        ir_version: Optional[int] = None,
    ):
        self.f = f
        self.tensor_class = tensor_class
        self.versions = {}
        self.onxs = {}
        self.target_opsets = tensor_class.get_opsets(target_opsets)
        self.output_types = output_types
        self.ir_version = tensor_class.get_ir_version(ir_version)
        # parameters necessary after the function was converting to
        # onnx to remember an input in fact a mandatory parameter.
        self.n_inputs_ = 0
        self.input_to_kwargs_ = None
        self.method_name_ = None

    def info(self, prefix: Optional[str] = None, method_name: Optional[str] = None):
        """
        Logs a status.
        """
        if prefix is None:
            logger.info("")
            return
        logger.info(
            "%s [%s.%s] nx=%d ni=%d kw=%d f=%s.%s cl=%s me=%s",
            prefix,
            self.__class__.__name__,
            method_name[:6],
            len(self.onxs),
            self.n_inputs_,
            0 if self.input_to_kwargs_ is None else 1,
            self.f.__module__,
            self.f.__name__,
            self.tensor_class.__name__,
            self.method_name_ or "",
        )

    def status(self, me: str) -> str:
        """
        Returns a short string indicating the status.
        """
        return (
            f"[{self.__class__.__name__}.{me[:6]}]"
            f"nx={len(self.onxs)} "
            f"ni={self.n_inputs_} "
            f"kw={0 if self.input_to_kwargs_ is None else 1} "
            f"f={self.f.__module__}.{self.f.__name__} "
            f"cl={self.tensor_class.__name__} "
            f"me={self.method_name_ or ''}"
        )

    @property
    def n_versions(self):
        """
        Returns the number of jitted functions.
        There is one per type and number of dimensions.
        """
        return len(self.onxs)

    @property
    def available_versions(self):
        """
        Returns the key used to distinguish between every jitted version.
        """
        return list(sorted(self.onxs))

    def get_onnx(self, key: Optional[int] = None):
        """
        Returns the jitted function associated to one key.
        If key is None, the assumes there is only one available jitted function
        and it returns it.
        """
        if key is None:
            if len(self.onxs) != 1:
                raise ValueError(
                    f"There is more than one jitted function. "
                    f"The key must be specified among "
                    f"{self.available_versions!r}."
                )
            return self.onxs[self.available_versions[0]]
        if key not in self.onxs:
            raise ValueError(
                f"Not jitted function indexed with "
                f"key={key!r} in {self.available_versions!r}."
            )
        return self.onxs[key]

    @staticmethod
    def make_key(*values, **kwargs):
        """
        Builds a key based on the input types and parameters.
        Every set of inputs or parameters producing the same
        key (or signature) must use the same compiled ONNX.
        """
        res = []
        for iv, v in enumerate(values):
            if isinstance(v, (Var, EagerTensor, JitTensor)):
                res.append(v.key)
            elif isinstance(v, (int, float)):
                res.append(v)
            elif isinstance(v, slice):
                res.append(("slice", v.start, v.stop, v.step))
            elif isinstance(v, type):
                res.append(("type", v.__name__))
            elif isinstance(v, tuple):
                subkey = []
                for sk in v:
                    if isinstance(sk, slice):
                        res.append(("slice", sk.start, sk.stop, sk.step))
                    elif isinstance(sk, (int, float)):
                        res.append(("slice", sk))
                    else:
                        raise TypeError(f"Input {iv} cannot have such tuple: {v}.")
                res.append(tuple(subkey))
            else:
                raise TypeError(
                    f"Unable to build a key, input {iv} has type {type(v)}."
                )
        if kwargs:
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (int, float, str, type)):
                    res.append(k)
                    res.append(v)
                elif isinstance(v, tuple):
                    newv = []
                    for t in v:
                        if isinstance(t, Var):
                            raise NotImplementedError(
                                f"Cannot support Var in argument {k!r}."
                            )
                        if isinstance(t, slice):
                            newv.append(("slice", t.start, t.stop, t.step))
                        else:
                            newv.append(t)
                    res.append(tuple(newv))
                else:
                    raise TypeError(
                        f"Type {type(v)} is not yet supported, "
                        f"v={v} and parameter {k!r}."
                    )
        key = tuple(res)
        return key

    def to_jit(self, *values, **kwargs):
        """
        Converts the function into ONNX based on the provided inputs
        and parameters. It then wraps it by calling
        `self.tensor_class.create_function`.
        The onnx graph built by the function defines the input
        types and the expected number of dimensions.
        """
        self.info("+", "to_jit")
        annotations = self.f.__annotations__
        if len(annotations) > 0:
            input_to_kwargs = {}
            names = list(annotations.keys())
            annot_values = list(annotations.values())
            constraints = {}
            new_kwargs = {}
            for i, (v, iname) in enumerate(zip(values, names)):
                if isinstance(v, (EagerTensor, JitTensor)) and (
                    i >= len(annot_values) or issubclass(annot_values[i], TensorType)
                ):
                    constraints[iname] = v.tensor_type_dims
                else:
                    new_kwargs[iname] = v
                    input_to_kwargs[i] = iname
            if self.input_to_kwargs_ is None:
                self.n_inputs_ = len(values) - len(input_to_kwargs)
                self.input_to_kwargs_ = input_to_kwargs
            elif self.input_to_kwargs_ != input_to_kwargs:
                raise RuntimeError(
                    f"Unexpected input and argument. Previous call produced "
                    f"self.input_to_kwargs_={self.input_to_kwargs_} and "
                    f"input_to_kwargs={input_to_kwargs} for function {self.f} "
                    f"from module {self.f.__module__!r}."
                )
        elif self.input_to_kwargs_:
            constraints = {}
            new_kwargs = {}
            for i, (v, iname) in enumerate(zip(values, names)):
                if (
                    isinstance(v, (EagerTensor, JitTensor))
                    and (
                        i >= len(annot_values)
                        or issubclass(annot_values[i], TensorType)
                    )
                    and i not in self.input_to_kwargs_
                ):
                    constraints[iname] = v.tensor_type_dims
                else:
                    new_kwargs[iname] = v
        else:
            names = [f"x{i}" for i in range(len(values))]
            new_kwargs = {}
            constraints = {
                iname: v.tensor_type_dims
                for i, (v, iname) in enumerate(zip(values, names))
                if isinstance(v, (EagerTensor, JitTensor))
            }
            self.n_inputs_ = len(values)
            self.input_to_kwargs_ = {}

        if self.output_types is not None:
            constraints.update(self.output_types)

        inputs = [
            Input(iname) for iname, v in zip(names, values) if iname in constraints
        ]
        names = [i.name for i in inputs]
        if len(new_kwargs) > 0:
            # An attribute is not named in the numpy API
            # but is the ONNX definition.
            if len(kwargs) == 0:
                kwargs = new_kwargs
            else:
                kwargs = kwargs.copy()
                kwargs.update(kwargs)

        var = self.f(*inputs, **kwargs)

        onx = var.to_onnx(
            constraints=constraints,
            target_opsets=self.target_opsets,
            ir_version=self.ir_version,
        )
        exe = self.tensor_class.create_function(names, onx)
        self.info("-", "to_jit")
        return onx, exe

    def cast_to_tensor_class(self, inputs: List[Any]) -> List[EagerTensor]:
        """
        Wraps input into `self.tensor_class`.

        :param inputs: python inputs (including numpy)
        :return: wrapped inputs
        """
        values = []
        for i, a in enumerate(inputs):
            try:
                values.append(self.tensor_class(a))
            except TypeError as e:
                raise TypeError(
                    f"Unable to convert input {i}, with type {type(a)}."
                ) from e
        return values

    def cast_from_tensor_class(
        self, results: List[EagerTensor]
    ) -> Union[Any, Tuple[Any]]:
        """
        Wraps input from `self.tensor_class` to python types.

        :param results: python inputs (including numpy)
        :return: wrapped inputs
        """
        if isinstance(results, (tuple, list)):
            if len(results) == 1:
                return results[0].value
            return tuple(r.value for r in results)
        return results.value

    def move_input_to_kwargs(
        self, values: List[Any], kwargs: Dict[str, Any]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Mandatory parameters not usually not named. Some inputs must
        be moved to the parameter list before calling ONNX.

        :param values: list of inputs
        :param kwargs: dictionary of arguments
        :return: new values, new arguments
        """
        if self.input_to_kwargs_ is None:
            # if self.bypass_eager or self.f.__annotations__:
            #    return values, kwargs
            raise RuntimeError(
                f"self.input_to_kwargs_ is not initialized for function {self.f} "
                f"from module {self.f.__module__!r}."
            )
        if len(self.input_to_kwargs_) == 0:
            return values, kwargs
        new_values = []
        new_kwargs = kwargs.copy()
        for i, v in enumerate(values):
            if i in self.input_to_kwargs_:
                new_kwargs[self.input_to_kwargs_[i]] = v
            else:
                new_values.append(v)
        return tuple(new_values), new_kwargs

    def jit_call(self, *values, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.
        """
        self.info("+", "jit_call")
        if self.input_to_kwargs_ is None:
            # No jitting was ever called.
            onx, fct = self.to_jit(*values, **kwargs)
            if self.input_to_kwargs_ is None:
                raise RuntimeError(
                    f"Attribute 'input_to_kwargs_' should be set for "
                    f"function {self.f} form module {self.f.__module__!r}."
                )
        else:
            onx, fct = None, None

        values, kwargs = self.move_input_to_kwargs(values, kwargs)
        key = self.make_key(*values, **kwargs)
        if self.method_name_ is None and "method_name" in key:
            pos = list(key).index("method_name")
            self.method_name_ = key[pos + 1]

        if onx is not None:
            # First jitting.
            self.versions[key] = fct
            self.onxs[key] = onx
        elif key in self.versions:
            # Already jitted.
            fct = self.versions[key]
        else:
            # One version was already jitted but types or parameter
            # are different.
            onx, fct = self.to_jit(*values, **kwargs)
            self.versions[key] = fct
            self.onxs[key] = onx
        try:
            res = fct.run(*values)
        except Exception as e:
            from ..plotting.text_plot import onnx_simple_text_plot

            text = onnx_simple_text_plot(self.onxs[key])
            raise RuntimeError(
                f"Unable to run function for key={key!r}, "
                f"types={[type(x) for x in values]}, "
                f"dtypes={[x.dtype for x in values]}, "
                f"shapes={[x.shape for x in values]}, "
                f"kwargs={kwargs}, "
                f"self.input_to_kwargs_={self.input_to_kwargs_}, "
                f"f={self.f} from module {self.f.__module__!r} "
                f"onnx=\n---\n{text}\n---\n{self.onxs[key]}"
            ) from e
        self.info("-", "jit_call")
        return res


class JitOnnx(JitEager):
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    :param ir_version: defines the IR version to use
    """

    def __init__(
        self,
        f: Callable,
        tensor_class: type = None,
        target_opsets: Optional[Dict[str, int]] = None,
        output_types: Optional[Dict[Any, TensorType]] = None,
        ir_version: Optional[int] = None,
    ):
        if tensor_class is None:
            from .npx_numpy_tensors import JitNumpyTensor

            tensor_class = JitNumpyTensor
        JitEager.__init__(
            self,
            f,
            tensor_class,
            target_opsets=target_opsets,
            output_types=output_types,
            ir_version=ir_version,
        )

    def __call__(self, *args, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.
        The method first wraps the inputs with `self.tensor_class`
        and converts them into python types just after.
        """
        self.info("+", "__call__")
        values = self.cast_to_tensor_class(args)
        res = self.jit_call(*values, **kwargs)
        res = self.cast_from_tensor_class(res)
        self.info("-", "jit_call")
        return res


class EagerOnnx(JitEager):
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    :param bypass_eager: this parameter must be true if the function
        has not annotation and is not decorated by `xapi_inline` or
        `xapi_function`
    :param ir_version: defines the IR version to use
    """

    def __init__(
        self,
        f: Callable,
        tensor_class: type = None,
        target_opsets: Optional[Dict[str, int]] = None,
        output_types: Optional[Dict[Any, TensorType]] = None,
        ir_version: Optional[int] = None,
        bypass_eager: bool = False,
    ):
        if tensor_class is None:
            from .npx_numpy_tensors import EagerNumpyTensor

            tensor_class = EagerNumpyTensor
        JitEager.__init__(
            self,
            f,
            tensor_class,
            target_opsets=target_opsets,
            output_types=output_types,
            ir_version=ir_version,
        )
        self.has_eager_parameter = "eager" in set(p for p in signature(f).parameters)
        self._eager_cache = False
        self.bypass_eager = bypass_eager

    def _preprocess_constants(self, *args):
        """
        An input may be a constant. It needs to be replaced by a tensor.
        """
        modified = False
        new_args = []
        for i, n in enumerate(args):
            if isinstance(n, self.tensor_class):
                new_args.append(n)
            elif isinstance(n, Cst):
                new_args.append(self.tensor_class(n.inputs[0]))
                modified = True
            elif isinstance(n, tuple):
                if all(map(lambda x: isinstance(x, int), n)):
                    new_args.append(
                        self.tensor_class(np.array(list(n), dtype=np.int64))
                    )
                elif any(map(lambda t: isinstance(t, Var), n)):
                    raise TypeError(
                        f"Unexpected types in tuple "
                        f"({[type(t) for t in n]}) for input {i}, "
                        f"function {self.f} from module {self.f.__module__!r}."
                    )
                else:
                    raise TypeError(
                        f"Unsupported tuple {n!r} for input {i}, "
                        f"function {self.f} from module {self.f.__module__!r}."
                    )
            elif isinstance(n, np.ndarray):
                new_args.append(self.tensor_class(n))
                modified = True
            elif isinstance(n, (int, float)):
                new_args.append(self.tensor_class(np.array(n)))
                modified = True
            elif n in (int, float):
                # usually used to cast
                new_args.append(n)
            elif n is None:
                new_args.append(n)
            else:
                raise TypeError(
                    f"Unexpected type ({type(n)}) for input {i}, "
                    f"function {self.f} from module {self.f.__module__!r}."
                )
        if modified:
            return tuple(new_args)
        return args

    def __call__(self, *args, already_eager=False, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.

        :param already_eager: already in eager mode, inputs must be of type
            EagerTensor and the returned outputs must be the same
        """
        self.info()
        self.info("+", "__call__")
        if already_eager:
            if any(
                map(
                    lambda t: t is not None
                    and not isinstance(
                        t,
                        (EagerTensor, Cst, int, float, tuple, slice, type, np.ndarray),
                    ),
                    args,
                )
            ):
                raise TypeError(
                    f"One of the input is not an EagerTensor or a constant, "
                    f"types are {[type(t) for t in args]} for function "
                    f"{self.f} from module {self.f.__module__!r}."
                )
            values_tensor = args
        else:
            values_tensor = self.cast_to_tensor_class(args)

        values = self._preprocess_constants(*values_tensor)

        if self._eager_cache or self.bypass_eager:
            # The function was already converted into onnx
            # reuse it or create a new one for different types.
            res = self.jit_call(*values, **kwargs)
            self.info("-", "1__call__")
        else:
            # tries to call the version
            try:
                res = self.f(*values)
            except (AttributeError, TypeError) as e:
                inp1 = ", ".join(map(str, map(type, args)))
                inp2 = ", ".join(map(str, map(type, values)))
                raise TypeError(
                    f"Unexpected types, input types are {inp1} " f"and {inp2}."
                ) from e

            if isinstance(res, EagerTensor) or (
                isinstance(res, tuple) and isinstance(res[0], EagerTensor)
            ):
                if already_eager:
                    raise TypeError(
                        f"EagerTensor ({type(res)}) is not "
                        f"expected for function {self.f} "
                        f"from module {self.f.__module__!r}, "
                        f"type of first input is {type(args[0])}."
                    )
            elif isinstance(res, Var) or any(map(lambda x: isinstance(x, Var), res)):
                # The function returns instance of type Var.
                # It does not support eager mode and needs
                # to be converted into onnx.
                res = self.jit_call(*values, **kwargs)
                self._eager_cache = True
            self.info("-", "2__call__")
        if already_eager:
            return tuple(res)
        return self.cast_from_tensor_class(res)


def jit_onnx(*args, **kwargs):
    """
    Returns an instance of :class:`JitOnnx`.
    """
    return JitOnnx(*args, **kwargs)


def eager_onnx(*args, **kwargs):
    """
    Returns an instance of :class:`EagerOnnx`.
    """
    return EagerOnnx(*args, **kwargs)
