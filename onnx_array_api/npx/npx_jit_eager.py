from inspect import signature
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto
from .npx_tensors import EagerTensor, JitTensor
from .npx_types import DType, OptTensorType, TensorType
from .npx_var import Cst, Input, Var

logger = getLogger("onnx-array-api")


class JitEager:
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvaluator`
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
        self.kwargs_to_input_ = None
        self.method_name_ = None

    def info(
        self,
        prefix: Optional[str] = None,
        method_name: Optional[str] = None,
        already_eager: Optional[bool] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        key: Optional[Tuple[Any, ...]] = None,
        onx: Optional[ModelProto] = None,
        output: Optional[Any] = None,
    ):
        """
        Logs a status.
        """
        if prefix is None:
            logger.info("")
            return
        if key is None:
            logger.info(
                "%s [%s.%s] nx=%d ni=%d ikw=%d kwi=%d f=%s.%s "
                "cl=%s me=%s mekw=%s ae=%s",
                prefix,
                self.__class__.__name__,
                method_name[:6],
                len(self.onxs),
                self.n_inputs_,
                0 if self.input_to_kwargs_ is None else 1,
                0 if self.kwargs_to_input_ is None else 1,
                self.f.__module__,
                self.f.__name__,
                self.tensor_class.__name__,
                self.method_name_ or "",
                "" if kwargs is None else kwargs.get("method_name", ""),
                "" if already_eager is None else (1 if already_eager else 0),
            )
        if method_name in ("jit_call", "jit_call_key") and (
            args is not None or kwargs is not None
        ):
            if key is not None:
                logger.debug("---- key=%s", key)
            logger.debug(
                "---- [%s] [%s]",
                "" if args is None else str(args),
                "" if kwargs is None else str(kwargs),
            )
        if output is not None:
            logger.debug("==== [%s]", output)

    def status(self, me: str) -> str:
        """
        Returns a short string indicating the status.
        """
        return (
            f"[{self.__class__.__name__}.{me[:6]}]"
            f"nx={len(self.onxs)} "
            f"ni={self.n_inputs_} "
            f"ikw={0 if self.input_to_kwargs_ is None else 1} "
            f"kwi={0 if self.kwargs_to_input_ is None else 1} "
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

    def make_key(self, *values: List[Any], **kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
        """
        Builds a key based on the input types and parameters.
        Every set of inputs or parameters producing the same
        key (or signature) must use the same compiled ONNX.

        :param values: values given to the function
        :param kwargs: parameters
        :return: tuple of mutable keys
        """
        res = []
        for iv, v in enumerate(values):
            if isinstance(v, (Var, EagerTensor, JitTensor)):
                if iv in self.kwargs_to_input_:
                    raise RuntimeError(
                        f"Input {iv} should be a constant to be moved "
                        f"to the attribute list, v={v}."
                    )
                res.append(v.key)
            elif isinstance(v, (int, float, bool, DType)):
                if iv in self.kwargs_to_input_:
                    res.append(self.kwargs_to_input_[iv])
                res.append(type(v))
                res.append(v)
            elif isinstance(v, slice):
                if iv in self.kwargs_to_input_:
                    raise NotImplementedError(
                        f"Input {iv} should be a constant to be moved "
                        f"to the attribute list, v={v}."
                    )
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
            elif v is None:
                if iv in self.kwargs_to_input_:
                    res.append(self.kwargs_to_input_[iv])
                res.append(None)
            else:
                raise TypeError(
                    f"Unable to build a key, input {iv} has type {type(v)}."
                )
        if kwargs:
            for k, v in sorted(kwargs.items()):
                if k in self.kwargs_to_input_:
                    res.append(type(v))
                    res.append(v)
                elif isinstance(v, (int, float, str, type, bool, DType)):
                    res.append(k)
                    res.append(type(v))
                    res.append(v)
                elif isinstance(v, slice):
                    if (
                        isinstance(v.start, Var)
                        or isinstance(v.stop, Var)
                        or isinstance(v.step, Var)
                    ):
                        raise NotImplementedError(
                            f"Cannot support Var in argument {k!r}."
                        )
                    res.append(k)
                    res.append(type(v))
                    res.append((v.start, v.stop, v.step))
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
                elif v is None:
                    # optional parameter or inputs
                    pass
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
        self.info("+", "to_jit", args=values, kwargs=kwargs)
        annotations = self.f.__annotations__
        if annotations:
            input_to_kwargs = {}
            kwargs_to_input = {}
            names = list(annotations.keys())
            annot_values = list(annotations.values())
            constraints = {}
            new_kwargs = {}
            for i, (v, iname) in enumerate(zip(values, names)):
                if i < len(annot_values) and not isinstance(annot_values[i], type):
                    raise TypeError(
                        f"annotation {i} is not a type but is "
                        f"{type(annot_values[i])!r}, "
                        f"annot_values[i]={annot_values[i]!r}, "
                        f"for function {self.f} "
                        f"from module {self.f.__module__!r}."
                    )
                if isinstance(v, (EagerTensor, JitTensor)) and (
                    i >= len(annot_values) or issubclass(annot_values[i], TensorType)
                ):
                    constraints[iname] = v.tensor_type_dims(annot_values[i].name)
                elif (
                    v is None
                    and i < len(annot_values)
                    and issubclass(annot_values[i], OptTensorType)
                ):
                    constraints[iname] = annot_values[i]
                    kwargs_to_input[iname] = i, annot_values[i]
                elif (
                    v is not None
                    and i < len(annot_values)
                    and issubclass(annot_values[i], TensorType)
                ):
                    constraints[iname] = annot_values[i]
                    kwargs_to_input[iname] = i, annot_values[i]
                else:
                    new_kwargs[iname] = v
                    input_to_kwargs[i] = iname
                    if iname == "shape":
                        raise RuntimeError(
                            f"Inconsistency for function {self.f}, iname={iname!r}, "
                            f"i={i}, v={v!r}, annot_values={annot_values}."
                        )

            if self.input_to_kwargs_ is None:
                self.n_inputs_ = (
                    len(values) - len(input_to_kwargs) + len(kwargs_to_input)
                )
                self.input_to_kwargs_ = input_to_kwargs
                self.kwargs_to_input_ = kwargs_to_input
            elif (
                self.input_to_kwargs_ != input_to_kwargs
                or self.input_to_kwargs_ != input_to_kwargs
            ):
                raise RuntimeError(
                    f"Unexpected input and argument. Previous call produced "
                    f"self.input_to_kwargs_={self.input_to_kwargs_}, "
                    f"self.kwargs_to_input_={self.kwargs_to_input_}, "
                    f"self.n_inputs_={self.n_inputs_} and "
                    f"input_to_kwargs={input_to_kwargs}, "
                    f"kwargs_to_input={kwargs_to_input} for function {self.f} "
                    f"from module {self.f.__module__!r}, "
                    f"len(values)={len(values)}, kwargs={kwargs!r}."
                )
        elif self.input_to_kwargs_ or self.kwargs_to_input_:
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
                    constraints[iname] = v.tensor_type_dims(iname)
                else:
                    new_kwargs[iname] = v
        else:
            names = [f"x{i}" for i in range(len(values))]
            new_kwargs = {}
            constraints = {
                iname: v.tensor_type_dims(iname)
                for i, (v, iname) in enumerate(zip(values, names))
                if isinstance(v, (EagerTensor, JitTensor))
            }
            self.n_inputs_ = len(values)
            self.input_to_kwargs_ = {}
            self.kwargs_to_input_ = {}

        if self.output_types is not None:
            constraints.update(self.output_types)

        inputs = [
            Input(iname, annotation=constraints[iname])
            for iname, v in zip(names, values)
            if iname in constraints
        ]
        names = [i.name for i in inputs]
        if new_kwargs:
            # An attribute is not named in the numpy API
            # but is the ONNX definition.
            if not kwargs:
                kwargs = new_kwargs
            else:
                kwargs = kwargs.copy()
                kwargs.update(new_kwargs)
        self.info("=", "to_jit", args=inputs, kwargs=kwargs)
        try:
            var = self.f(*inputs, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"Unexpected error, inputs={inputs}, kwargs={kwargs}, "
                f"self.input_to_kwargs_={self.input_to_kwargs_}, "
                f"self.kwargs_to_input_={self.kwargs_to_input_}."
            ) from e

        onx = var.to_onnx(
            constraints=constraints,
            target_opsets=self.target_opsets,
            ir_version=self.ir_version,
        )
        if values and not values[0].shape:
            inps = onx.graph.input[0]
            shape = []
            for d in inps.type.tensor_type.shape.dim:
                v = d.dim_value if d.dim_value > 0 else d.dim_param
                shape.append(v)
            if shape:
                raise RuntimeError(
                    f"Shape mismatch, values[0]={values[0]} "
                    f"and inputs={onx.graph.input}."
                )

        exe = self.tensor_class.create_function(names, onx, f=self.f)
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
        if not self.input_to_kwargs_:
            return values, kwargs
        new_values = []
        new_kwargs = kwargs.copy()
        for i, v in enumerate(values):
            if i in self.input_to_kwargs_:
                new_kwargs[self.input_to_kwargs_[i]] = v
            else:
                new_values.append(v)
        if "shape" in new_kwargs:
            raise RuntimeError(
                f"Inconsistency for function {self.f}, "
                f"values={values}, kwargs={kwargs}, ",
                f"new_values={new_values}, new_kwargs={new_kwargs}, "
                f"self.input_to_kwargs_={self.input_to_kwargs_}",
            )
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
        self.info("+", "jit_call", args=values, kwargs=kwargs)
        if self.input_to_kwargs_ is None:
            # No jitting was ever called.
            try:
                onx, fct = self.to_jit(*values, **kwargs)
            except TypeError as e:
                raise TypeError(
                    f"ERROR with self.f={self.f}, "
                    f"values={values!r}, kwargs={kwargs!r}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Undefined ERROR with self.f={self.f}, "
                    f"values={values!r}, kwargs={kwargs!r}"
                ) from e
            if self.input_to_kwargs_ is None:
                raise RuntimeError(
                    f"Attribute 'input_to_kwargs_' should be set for "
                    f"function {self.f} form module {self.f.__module__!r}."
                )
            if self.kwargs_to_input_ is None:
                raise RuntimeError(
                    f"Attribute 'kwargs_to_input_' should be set for "
                    f"function {self.f} form module {self.f.__module__!r}."
                )
        else:
            onx, fct = None, None

        values, kwargs = self.move_input_to_kwargs(values, kwargs)
        key = self.make_key(*values, **kwargs)
        self.info("=", "jit_call_key", key=key, args=values, kwargs=kwargs)
        if self.method_name_ is None and "method_name" in key:
            pos = list(key).index("method_name")
            self.method_name_ = key[pos + 2]

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

            def catch_len(x):
                try:
                    return len(x)
                except TypeError:
                    return 0

            raise RuntimeError(
                f"Unable to run function for key={key!r}, "
                f"types={[type(x) for x in values]}, "
                f"dtypes={[getattr(x, 'dtype', type(x)) for x in values]}, "
                f"shapes={[getattr(x, 'shape', catch_len(x)) for x in values]}, "
                f"kwargs={kwargs}, "
                f"self.input_to_kwargs_={self.input_to_kwargs_}, "
                f"f={self.f} from module {self.f.__module__!r} "
                f"onnx=\n---\n{text}\n---\n{self.onxs[key]}"
            ) from e
        self.info("-", "jit_call", output=res)
        return res


class JitOnnx(JitEager):
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvaluator`
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
        self.info("+", "__call__", args=args, kwargs=kwargs)
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
        if None, it defaults to :class:`onnx.reference.ReferenceEvaluator`
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

    allowed_input_types = (
        EagerTensor,
        Cst,
        int,
        bool,
        float,
        tuple,
        slice,
        type,
        # np.ndarray,
        DType,
    )

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
                    modified = True
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
            elif isinstance(n, (int, float, bool)):
                new_args.append(self.tensor_class(np.array(n)))
                modified = True
            elif isinstance(n, DType):
                new_args.append(n)
            elif n in (int, float, bool):
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
        self.info(
            "+", "__call__", already_eager=already_eager, args=args, kwargs=kwargs
        )
        if already_eager:
            if any(
                map(
                    lambda t: t is not None
                    and not isinstance(
                        t,
                        EagerOnnx.allowed_input_types,
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
            self.info(
                "-", "1__call__", already_eager=already_eager, args=args, kwargs=kwargs
            )
        else:
            # tries to call the version
            try:
                res = self.f(*values, **kwargs)
            except (AttributeError, TypeError) as e:
                inp1 = ", ".join(map(str, map(lambda a: type(a).__name__, args)))
                inp2 = ", ".join(map(str, map(lambda a: type(a).__name__, values)))
                raise TypeError(
                    f"Unexpected types, input types are args=[{inp1}], "
                    f"values=[{inp2}], kwargs={kwargs}. "
                    f"(values = self._preprocess_constants(args)) "
                    f"args={args}, values={values}"
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
            self.info(
                "-", "2__call__", already_eager=already_eager, args=args, kwargs=kwargs
            )
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
    Returns an instance of :class:`EagerOnnx
    <onnx_array_api.npx.npx_jit_eager.EagerOnnx>`.
    """
    return EagerOnnx(*args, **kwargs)
