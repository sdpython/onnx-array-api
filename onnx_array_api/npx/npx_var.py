from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto
from .npx_array_api import BaseArrayApi, ArrayApiError
from .npx_constants import DEFAULT_OPSETS, ONNX_DOMAIN
from .npx_types import (
    DType,
    ElemType,
    OptParType,
    ParType,
    TensorType,
    OptTensorType,
    TupleType,
)


class Par:
    """
    Defines a named parameter.

    :param name: parameter name
    :param dtype: parameter type (bool, int, str, float)
    :param value: value of the parameter if known
    :param parent_op: node type it belongs to
    """

    def __init__(
        self,
        name: str,
        dtype: ParType,
        value: Optional[Any] = None,
        parent_op: Optional[Tuple[str, str, int]] = None,
    ):
        if not issubclass(dtype, ParType):
            raise TypeError(
                f"dtype for parameter {name!r} must be of ParType not {dtype}."
            )
        if parent_op is None:
            raise ValueError(f"parent_op must be filled for paramenter {name!r}.")
        self.name = name
        self.dtype = dtype
        self.value = value
        self.parent_op = parent_op

    def __repr__(self):
        "usual"
        if self.value is None:
            return (
                f"{self.__class__.__name__}({self.name!r}, {self.dtype.type_name()}, "
                f"parent_op={self.parent_op!r})"
            )
        return (
            f"{self.__class__.__name__}"
            f"({self.name!r}, {self.dtype.type_name()}, {self.value!r}, "
            f"parent_op={self.parent_op!r})"
        )

    @property
    def onnx_type(self):
        "Returns the corresponding onnx type."
        return self.dtype.onnx_type()

    def __eq__(self, x):
        "Should not be used."
        raise NotImplementedError("__eq__ should not be used.")

    def __neq__(self, x):
        "Should not be used."
        raise NotImplementedError("__neq__ should not be used.")

    def __lt__(self, x):
        "Should not be used."
        raise NotImplementedError("__lt__ should not be used.")

    def __gt__(self, x):
        "Should not be used."
        raise NotImplementedError("__gt__ should not be used.")

    def __le__(self, x):
        "Should not be used."
        raise NotImplementedError("__le__ should not be used.")

    def __ge__(self, x):
        "Should not be used."
        raise NotImplementedError("__ge__ should not be used.")


class ManyIdentity:
    """
    Holds several instances of :class:`Var <onnx_array_api.npx.npx_var.Var>`.
    """

    def __init__(self, *inputs, input_indices=None):
        self.inputs = inputs
        self.onnx_op = None
        if input_indices is None:
            self.input_indices = [0 for i in self.inputs]
        else:
            self.input_indices = input_indices
        self.n_var_outputs = len(self.inputs)
        self.onnx_op_kwargs = {}
        self._prefix = "ManyIdentity_"

    def __repr__(self) -> str:
        "usual"
        args = list(map(repr, self.inputs))
        if max(self.input_indices) > 0:
            args.append(f"input_indices={self.input_indices}")
        s = ", ".join(args)
        return f"{self.__class__.__name__}({s})"

    def __len__(self):
        "Returns the number of merged variables."
        return len(self.inputs)

    def __getitem__(self, i):
        "Returns the ith elements."
        return self.inputs[i]

    def to_onnx(
        self,
        target_opsets: Optional[Dict[str, int]] = None,
        as_function: bool = False,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        constraints: Optional[Dict[Any, TensorType]] = None,
        ir_version: Optional[int] = None,
    ) -> Union[ModelProto, FunctionProto, List[Any]]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`, if None,
            it is replaced by `DEFAULT_OPSETS`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :param attributes: function attributes if any
        :param constraints: specifies a precise type for the type
            constraints when a function allows more than one type,
            this works if there is only one variable to be converted
        :return: ModelProto, FunctionProto
        """
        from .npx_graph_builder import _GraphBuilder

        # Var.to_onnx
        if target_opsets is None:
            target_opsets = DEFAULT_OPSETS.copy()
        g = _GraphBuilder(
            target_opsets,
            as_function=as_function,
            name=name,
            domain=domain,
            attributes=attributes,
            constraints=constraints,
            ir_version=ir_version,
        )
        done = set()
        outputs = []
        for var in self.inputs:
            vs = var._get_vars()
            for var2 in vs:
                key = id(var2)
                if key in done:
                    continue
                g.append(var2)
                done.add(key)
            outputs.append(vs[-1])
        onx = g.to_onnx(output_vars=outputs)
        if as_function:
            if len(outputs) != len(onx.output):
                raise RuntimeError(
                    f"Mismatch number of outputs, expecting {len(outputs)}, "
                    f"got ({len(onx.output)})."
                )
            if g.functions_:
                return [g.functions_, onx]
            return onx

        if len(outputs) != len(onx.graph.output):
            raise RuntimeError(
                f"Mismatch number of outputs, expecting {len(outputs)}, "
                f"got ({len(onx.graph.output)})."
            )
        return onx


class Var(BaseArrayApi):
    """
    Defines a variable, a result...

    :param inputs: list of inputs
    :param op: apply on operator on the inputs
    :param inline: True to reduce the use of function and inline
        small functions, this only applies if *op* is a function
    :param n_var_outputs: number of the operator outputs
    :param input_indices: to select a specific output from the input
        operator
    :param kwargs: operator attributes

    Private attribute:

    :param onnx_input_type_: names given to the variables
    """

    def __array_namespace__(self, api_version: Optional[str] = None):
        """
        Raises an exception if called.
        """
        raise RuntimeError(
            f"This function should never be called for class {type(self)}. "
            f"It should be called for an eager tensor."
        )

    @staticmethod
    def get_cst_var():
        from .npx_core_api import cst, var

        return cst, var

    class _setter_do:
        def __init__(self, parent: "Var", *args):
            self.parent = parent.self_var
            self.args = args

        def __call__(self, new_values):
            """
            Returns a copy of `self.parent` where values
            whose indices are indicated by `args` and new
            values by `new_values`.
            """
            if len(self.args) == 1 and isinstance(self.args[0], (int, slice)):
                return self._setitem1_slice(self.args[0], new_values)
            if len(self.args) == 1 and isinstance(self.args[0], Var):
                return self._setitem1_where(self.args[0], new_values)
            raise NotImplementedError(
                f"This expression is not yet implemented for args={self.args}."
            )

        def _setitem1_where(self, index, new_values):
            cst, var = Var.get_cst_var()
            if isinstance(new_values, (int, float, bool)):
                new_values = np.array(new_values)
            if isinstance(new_values, np.ndarray):
                value = var(cst(new_values), self.parent, op="CastLike")
            elif isinstance(new_values, Var):
                value = new_values
            else:
                raise TypeError(f"Unexpected type for new_values: {type(new_values)}.")
            return var(index, value, self.parent, op="Where")

        def _setitem1_slice(self, index, new_values):
            cst, var = Var.get_cst_var()

            if isinstance(index, slice):
                start = 0 if index.start is None else index.start
                stop = index.stop
                step = index.step
            elif isinstance(index, int):
                start, stop, step = index, index + 1, 1
            else:
                raise NotImplementedError(
                    f"Unable to assign new values due to "
                    f"unexpected type {type(index)!r}."
                )

            inp = self.parent
            if stop is None and isinstance(new_values, np.ndarray):
                stop = start + new_values.size
            if stop is None:
                raise NotImplementedError(f"No implementation if stop is  {stop}.")
            indices = np.arange(start, stop, step or 1).astype(np.int64)
            if isinstance(new_values, np.ndarray):
                values = new_values
            else:
                values = np.full(indices.shape, new_values)
            return var(inp, cst(indices), cst(values), op="ScatterElements", axis=0)

    class _setter:
        def __init__(self, parent: "Var"):
            self.parent = parent

        def __getitem__(self, *args):
            return Var._setter_do(self.parent, *args)

    def __init__(
        self,
        *inputs: List[Any],
        op: Optional[
            Union[Callable, str, Tuple[str, str], FunctionProto, ModelProto, NodeProto]
        ] = None,
        dtype: Optional[Union[type, DType]] = None,
        inline: bool = False,
        n_var_outputs: int = 1,
        input_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        self.inputs = list(inputs)
        self.n_var_outputs = n_var_outputs
        self.inline = inline
        self._annotation = None
        if op is None:
            self.onnx_op = None  # a constant
        elif isinstance(op, tuple):
            self.onnx_op = op  # domain, operator name
        elif isinstance(op, str):
            self.onnx_op = ("", op)  # operator name
        elif isinstance(op, (FunctionProto, ModelProto, NodeProto)):
            self.onnx_op = (ONNX_DOMAIN, op)
        else:
            self.onnx_op = (None, op)  # function to call

        self.onnx_op_kwargs = kwargs
        self._prefix = None
        if isinstance(dtype, DType):
            # regular parameter
            self.onnx_op_kwargs["dtype"] = dtype
        elif hasattr(dtype, "type_name"):
            self.dtype = dtype
        elif dtype is None:
            self.dtype = None
        else:
            raise TypeError(f"Unexpected type {type(dtype)} for dtype.")

        updates = {}
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, type):
                raise TypeError(f"Unexpected type for input {i} - {inp}.")
            if isinstance(inp, Var):
                updates[i] = inp.self_var
            if not isinstance(inp, np.ndarray):
                continue
            if inp.size > 0 and isinstance(inp.ravel()[0], (np.ndarray, Var)):
                raise TypeError(
                    f"Unexpected type for input {i}: {type(inp)}, "
                    f"{inp.ravel()[0]}, op={op!r}"
                )
        # This step is needed when Var.__setitem__ was called to
        # modify the variable.
        for i, v in updates.items():
            self.inputs[i] = v
        self.inputs = tuple(self.inputs)
        if input_indices is None:
            self.input_indices = [0 for i in self.inputs]
        elif not isinstance(input_indices, list):
            raise TypeError(
                f"input_indices is {type(input_indices)} "
                f"but len(inputs)={len(inputs)}."
            )
        else:
            self.input_indices = input_indices
        if len(self.input_indices) != len(self.inputs):
            raise RuntimeError(
                f"length mismatch len(self.input_indices)="
                f"{len(self.input_indices)} != len(self.inputs)="
                f"{len(self.inputs)}."
            )
        if self.onnx_op is None:
            if not isinstance(self, (Input, Cst)):
                raise RuntimeError(f"This case is not allowed: {self!r}.")
        self.set = Var._setter(self)
        self.current_var_ = None

    @property
    def annotation(self):
        """Returns a type if known for the Var itself."""
        if self._annotation is None:
            if "dtype" in self.onnx_op_kwargs:
                dtype = self.onnx_op_kwargs["dtype"]
                if isinstance(dtype, DType):
                    return TensorType[dtype]
        return self._annotation

    @property
    def self_var(self):
        """
        Returns itself or the variable corresponding to its
        state after a call to `__setitem__`.
        """
        if not hasattr(self, "current_var_"):
            raise AttributeError(
                f"Class {type(self)} is missing attribute 'current_var_'."
            )
        return self if self.current_var_ is None else self.current_var_

    def __call__(self):
        return self.self_var

    def replace_inputs(
        self, new_inputs: List["Var"], input_indices: Optional[List[int]] = None
    ) -> "Var":
        """
        Replaces inputs by new ones. It creates a copy.
        It is needed when inlining functions.
        """
        new_var = Var(
            *new_inputs,
            op=self.onnx_op,
            dtype=self.dtype,
            inline=self.inline,
            input_indices=input_indices,
            n_var_outputs=self.n_var_outputs,
            **self.onnx_op_kwargs,
        )
        new_var._prefix = self._prefix
        return new_var

    def __repr__(self) -> str:
        "usual"
        args = []
        for inp in self.inputs:
            n = inp.__class__.__name__
            args.append(f"{n[0]}.")
        if self.onnx_op is not None:
            args.append(f"op={self.onnx_op!r}")
        if self.n_var_outputs != 1:
            args.append(f"n_var_outputs={self.n_var_outputs!r}")
        if max(self.input_indices) != 0:
            args.append(f"input_indices={self.input_indices!r}")
        for k, v in sorted(self.onnx_op_kwargs.items()):
            args.append(f"{k}={v!r}")
        res = f"{self.__class__.__name__}({', '.join(args)})"
        return res

    def set_onnx_name(self, prefix: str):
        """
        Forces this variable to get this name during

        :param prefix: prefix
        """
        self._prefix = prefix

    def _get_vars(self):
        vs = []
        stack = [self.self_var]
        replacement = {}
        replacement_cst = {}
        deleted = []
        while len(stack) > 0:
            var = stack.pop()
            key = id(var)
            if key in replacement:
                while key in replacement:
                    var = replacement[key]
                    key = id(var)
            if var.onnx_op is not None and var.onnx_op[0] is None and var.inline:
                fct = var.onnx_op[1]
                applied = fct(*var.inputs, **var.onnx_op_kwargs)
                if isinstance(applied, (ManyIdentity, Var)):
                    stack.append(applied)
                    replacement[id(var)] = applied
                    deleted.append(var)
                    continue
                raise TypeError(
                    f"Unexpected type {type(applied)} as output of function {fct}."
                )
            vs.append(var)
            for i in reversed(var.inputs):
                if isinstance(i, Var):
                    stack.insert(0, i)
                    continue
                if isinstance(i, np.ndarray):
                    cst = Var.get_cst_var()[0]
                    replacement_cst[id(i)] = cst(i)
                    continue
                if isinstance(i, (int, float, bool)):
                    cst = Var.get_cst_var()[0]
                    replacement_cst[id(i)] = cst(np.array(i))
                    continue
                if isinstance(i, tuple):
                    if all(isinstance(x, int) for x in i):
                        cst = Var.get_cst_var()[0]
                        replacement_cst[id(i)] = cst(np.array(list(i), dtype=np.int64))
                        continue
                    if any(isinstance(t, Var) for t in i):
                        raise TypeError(
                            f"Unexpected types in tuple "
                            f"({[type(t) for t in i]}), "
                            f"function {self.f} from module {self.f.__module__!r}."
                        )
                    raise TypeError(
                        f"Unsupported tuple {i!r}, "
                        f"function {self.f} from module {self.f.__module__!r}."
                    )
                if i is None:
                    continue
                raise TypeError(
                    f"Unexpected type {type(i)} for an input of node {var}."
                )
        res = list(reversed(vs))

        # replacement: a node calling a function can either
        # remains as a call to a local function or the code
        # of the function can replace the call inline.
        # replacement keeps a map of function call to replace
        # by the return itself to avoid calling the same function
        # twice.
        new_res = []
        for r in res:
            new_inputs = []
            new_indices = []
            repl = False
            for v, ind in zip(r.inputs, r.input_indices):
                key = id(v)
                if key in replacement:
                    while key in replacement:
                        var = replacement[key]
                        key = id(var)
                    new_inputs.append(var)
                    new_indices.append(ind)
                    repl = True
                else:
                    new_inputs.append(v)
                    new_indices.append(ind)
            if repl:
                new_r = r.replace_inputs(new_inputs, input_indices=new_indices)
                replacement[id(r)] = new_r
                new_res.append(new_r)
            else:
                new_res.append(r)

        # check the graph is consistent
        known = {}
        for r in new_res:
            known[id(r)] = r
            if isinstance(r, (Cst, Input)):
                continue
            for ind, i in enumerate(r.inputs):
                if i is None:
                    # optional input
                    continue
                if id(i) in replacement_cst:
                    # constant to replace
                    continue
                if id(i) not in known:
                    raise RuntimeError(
                        f"An input {ind} ({id(i)}, type={type(i)}) from "
                        f"{id(r)}-{r} is not known, it is not produced by a "
                        f"previous var (scheduled for replacement: "
                        f"{id(i) in replacement}). This also happens if "
                        f"a constant is not wrapped by 'cst(.)'."
                    )
        return new_res

    @property
    def is_function(self):
        """
        Tells if this variable encapsulate a function.
        """
        return self.onnx_op is not None and self.onnx_op[0] is None

    def to_onnx(
        self,
        target_opsets: Optional[Dict[str, int]] = None,
        as_function: bool = False,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        constraints: Optional[Dict[Any, TensorType]] = None,
        ir_version: Optional[int] = None,
    ) -> Union[ModelProto, FunctionProto, List[Any]]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :param attributes: function attributes if any
        :param constraints: specifies a precise type for the type
            constraints when a function allows more than one type,
            this works if there is only one variable to be converted
        :return: ModelProto, FunctionProto
        """
        from .npx_graph_builder import _GraphBuilder

        # Var.to_onnx
        if target_opsets is None:
            target_opsets = DEFAULT_OPSETS

        vs = self._get_vars()

        g = _GraphBuilder(
            target_opsets,
            as_function=as_function,
            name=name,
            domain=domain,
            attributes=attributes,
            constraints=constraints,
            ir_version=ir_version,
        )

        for var in vs:
            g.append(var)
        onx = g.to_onnx()
        if as_function and len(g.functions_) > 0:
            return [g.functions_, onx]
        return onx

    # Operators

    def __iter__(self):
        """
        The :epkg:`Array API` does not define this function (2022/12).
        This method raises an exception with a better error message.
        """
        raise ArrayApiError(
            f"Iterators are not implemented in the generic case. "
            f"Every function using them cannot be converted into ONNX "
            f"(Var - {type(self)})."
        )

    def _binary_op(self, ov: "Var", op_name: str, **kwargs) -> "Var":
        var = Var.get_cst_var()[1]
        if isinstance(ov, (int, float, bool, np.ndarray, Cst)):
            return var(self.self_var, var(ov, self.self_var, op="CastLike"), op=op_name)
        return var(self.self_var, ov, op=op_name, **kwargs)

    def _binary_op_right(self, ov: "Var", op_name: str, **kwargs) -> "Var":
        var = Var.get_cst_var()[1]
        if isinstance(ov, (int, float, bool, np.ndarray, Cst)):
            return var(var(ov, self.self_var, op="CastLike"), self.self_var, op=op_name)
        return var(ov, self.self_var, op=op_name, **kwargs)

    def __neg__(self) -> "Var":
        """
        Automatically adds operator `Neg` to the graph.
        It does not cast automatically.
        """
        var = Var.get_cst_var()[1]

        return var(self.self_var, op="Neg")

    def __invert__(self) -> "Var":
        """
        Automatically adds operator `BitwiseNot` to the graph.
        It does not cast automatically.
        """
        var = Var.get_cst_var()[1]

        return var(self.self_var, op="BitwiseNot")

    def __add__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Add` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Add")

    def __radd__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Add` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "Add")

    def __sub__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Sub` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Sub")

    def __rsub__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Sub` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "Sub")

    def __mul__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Mul` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Mul")

    def __rmul__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Mul` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "Mul")

    def __matmul__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `MatMul` to the graph.
        It does not cast automatically.
        `__rmatmul__` would not be called as a numpy array
        overwrites `__matmul__` on its side.
        """
        return self._binary_op(ov, "MatMul")

    def __truediv__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Div` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Div")

    def __rtruediv__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Div` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "Div")

    def __mod__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Mod` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Mod")

    def __rmod__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Mod` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "Mod")

    def __pow__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Pow` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Pow")

    def __rpow__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Pow` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "Pow")

    def __lt__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Less` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Less")

    def __le__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `LessOrEqual` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "LessOrEqual")

    def __gt__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Greater` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Greater")

    def __ge__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `GreaterOrEqual` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "GreaterOrEqual")

    def __eq__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Equal` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "Equal")

    def __ne__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `Not + Equal` to the graph.
        It does not cast automatically.
        """
        var = Var.get_cst_var()[1]

        return var(self._binary_op(ov, "Equal"), op="Not")

    def __lshift__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitShift` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "BitShift", direction="LEFT")

    def __rshift__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitShift` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "BitShift", direction="RIGHT")

    def __and__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitwiseAnd` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "BitwiseAnd")

    def __rand__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitwiseAnd` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "BitwiseAnd")

    def __or__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitwiseOr` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "BitwiseOr")

    def __ror__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitwiseOr` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "BitwiseOr")

    def __xor__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitwiseXor` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, "BitwiseXor")

    def __rxor__(self, ov: "Var") -> "Var":
        """
        Automatically adds operator `BitwiseXor` to the graph.
        It does not cast automatically.
        """
        return self._binary_op_right(ov, "BitwiseXor")

    @property
    def T(self) -> "Var":
        "Transpose."
        var = Var.get_cst_var()[1]
        return var(self.self_var, op="Transpose", perm=[1, 0])

    def astype(self, dtype) -> "Var":
        "Cast"
        var = Var.get_cst_var()[1]

        if isinstance(dtype, Var):
            return var(self.self_var, dtype, op="CastLike")
        if not isinstance(dtype, DType):
            raise TypeError(f"dtype cannot be {type(dtype)}.")
        return var(self.self_var, op="Cast", to=dtype)

    @property
    def shape(self) -> "Var":
        "Shape"
        var = Var.get_cst_var()[1]

        return var(self.self_var, op="Shape")

    def reshape(self, shape: "Var") -> "Var":
        "Reshape"
        cst, var = Var.get_cst_var()

        if isinstance(shape, (tuple, list)):
            shape = np.array(shape, dtype=np.int64)
        else:
            shape = var(shape, cst(np.array([-1], dtype=np.int64)), op="Reshape")
        return var(self.self_var, shape, op="Reshape")

    def reduce_function(
        self,
        reduce_op,
        axis: OptTensorType[ElemType.int64, "I"] = None,
        keepdims: ParType[int] = 0,
    ) -> "Var":
        "See :func:`numpy.sum` or any other reduce function."
        var = Var.get_cst_var()[1]

        if axis is None:
            return var(self.self_var, op=reduce_op, keepdims=keepdims)
        if isinstance(axis, int):
            axis = [axis]
        if isinstance(axis, (tuple, list)):
            cst = Var.get_cst_var()[0]

            axis = cst(np.array(axis, dtype=np.int64))
        return var(self.self_var, axis, op=reduce_op, keepdims=keepdims)

    def sum(
        self, axis: TensorType[ElemType.int64, "I"] = None, keepdims: ParType[int] = 0
    ) -> "Var":
        "See :func:`numpy.sum`."
        return self.reduce_function("ReduceSum", axis=axis, keepdims=keepdims)

    def mean(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "Var":
        "See :func:`numpy.mean`."
        return self.reduce_function("ReduceMean", axis=axis, keepdims=keepdims)

    def min(
        self, axis: TensorType[ElemType.int64, "I"] = None, keepdims: ParType[int] = 0
    ) -> "Var":
        "See :func:`numpy.min`."
        return self.reduce_function("ReduceMin", axis=axis, keepdims=keepdims)

    def max(
        self, axis: TensorType[ElemType.int64, "I"] = None, keepdims: ParType[int] = 0
    ) -> "Var":
        "See :func:`numpy.max`."
        return self.reduce_function("ReduceMax", axis=axis, keepdims=keepdims)

    def prod(
        self, axis: TensorType[ElemType.int64, "I"] = None, keepdims: ParType[int] = 0
    ) -> "Var":
        "See :func:`numpy.prod`."
        return self.reduce_function("ReduceProd", axis=axis, keepdims=keepdims)

    def copy(self) -> "Var":
        """
        Returns a copy of self (use of Identity node).
        """
        var = Var.get_cst_var()[1]

        return var(self.self_var, op="Identity")

    def flatten(self) -> "Var":
        """
        Flattens a matrix (see :meth:`numpy.ndarray.flatten`).

        :param axis: only flatten from axis to the end.
        :return: :class:`Var <onnx_array_api.npx.npx_var.Var>`
        """
        cst, var = Var.get_cst_var()

        return var(
            var(self.self_var, op="Flatten", axis=0),
            cst(np.array([0], dtype=np.int64)),
            op="Squeeze",
        )

    def get(self, index: int) -> "Var":
        """
        If an operator or a function returns more than one output,
        this takes only one.

        :param index: index of the output to select
        :return: Var
        """
        if index < 0 or index >= self.n_var_outputs:
            raise ValueError(
                f"index={index} must be positive and < {self.n_var_outputs} "
                f"for var={self!r}."
            )
        return Var(self.self_var, input_indices=[index], op="Identity")

    def __getitem__(self, index: Any) -> "Var":
        """
        Deals with multiple scenarios.

        * *index* is an integer and the object produces multiple
          outputs and this returns one of them (**scenario 0**)
        * *index* is an integer or a slice, a tuple of integers and slices,
          example: `[0, 1]`, `[:5, :6]`, `[::2]` (**scenario 1**)
        * *index* is an *ONNX* object (more precisely an instance of
          :class:`Var <onnx_array_api.npx.npx_var.Var>`),
          then the method assumes it is an array of
          boolean to select a subset of the tensor along the first axis,
          example: `mat[mat == 0]` (**scenario 2**)
        """
        cst, var = Var.get_cst_var()

        if self.n_var_outputs != 1:
            # Multioutput
            if not isinstance(index, int):
                raise TypeError(
                    f"Only indices are allowed when selecting an output, "
                    f"not {type(index)})."
                )
            return self.get(index)

        if isinstance(index, Var):
            # scenario 2
            # we rely on the annotation if it exists
            if index.annotation is None:
                dtype_bool = True
            elif issubclass(index.annotation, TensorType):
                if index.annotation.supports_dtype(
                    DType(TensorProto.INT64)
                ) or index.annotation.supports_dtype(DType(TensorProto.INT32)):
                    dtype_bool = False
                elif index.annotation.supports_dtype(DType(TensorProto.BOOL)):
                    dtype_bool = True
                else:
                    raise TypeError(
                        f"Unexpected dtype for annotation={index.annotation!r} "
                        f"for index={index!r}."
                    )
            else:
                raise TypeError(
                    f"Unexpected annotation={index.annotation!r} "
                    f"for index={index!r}."
                )

            if dtype_bool:
                # TODO: fix this when index is an integer and the annotation unknown
                # it needs to support subgraph and tests
                new_shape = cst(np.array([-1], dtype=np.int64))
                new_self = self.reshape(new_shape)
                new_index = index.reshape(new_shape)
                return var(new_self, new_index, op="Compress")

            # dtype is int
            return var(self, index, axis=0, op="Gather")

        if isinstance(index, int):
            # Use Gather instead.
            return var(self, cst(np.array(index, dtype=np.int64)), axis=0, op="Gather")

        if not isinstance(index, tuple):
            index = (index,)
        elif not index:
            # The array contains a scalar and it needs to be returned.
            return var(self, op="Identity")

        # only one integer?
        ni = None
        ax = None
        for i, a in enumerate(index):
            if isinstance(a, int):
                if ni is None:
                    ni = i
                    ax = a
                else:
                    ax = None
                    ni = None
                    break
            if (
                isinstance(a, slice)
                and a.start is None
                and a.stop is None
                and a.step is None
            ):
                continue
            ax = None
            ni = None
            break

        if ni is not None and ax is not None:
            # Use Gather instead.
            return var(self, cst(np.array(ni, dtype=np.int64)), axis=ax, op="Gather")

        # scenario 1
        starts = []
        ends = []
        axes = []
        steps = []
        axis_squeeze = []
        needs_shape = []
        for i, ind in enumerate(index):
            if isinstance(ind, int):
                starts.append(ind)
                ends.append(ind + 1)
                axes.append(i)
                steps.append(1)
                axis_squeeze.append(i)
                continue
            if isinstance(ind, slice):
                if ind.start is None and ind.stop is None and ind.step is None:
                    continue
                start = 0 if ind.start is None else ind.start
                end = (None, i) if ind.stop is None else ind.stop
                step = 1 if ind.step is None else ind.step
                starts.append(start)
                ends.append(end)
                axes.append(i)
                steps.append(step)
                if isinstance(end, tuple):
                    needs_shape.append(len(ends) - 1)
                elif isinstance(end, Var):
                    needs_shape.append(end)
                continue
            raise NotImplementedError(f"Not implemented for type {type(ind)!r}.")

        if max(steps) == min(steps) == 1:
            steps = None
        else:
            steps = np.array(steps, dtype=np.int64)

        starts = np.array(starts, dtype=np.int64)
        axes = np.array(axes, dtype=np.int64)

        if needs_shape:
            shape = self.shape
            conc = []
            for e in ends:
                if isinstance(e, tuple):
                    conc.append(
                        var(shape, cst(np.array([e[1]], np.int64)), op="Gather")
                    )
                elif isinstance(e, Var):
                    conc.append(e.reshape(np.array([-1], dtype=np.int64)))
                else:
                    conc.append(np.array([e], dtype=np.int64))
            if len(conc) > 1:
                conc_cst = [v if isinstance(v, Var) else cst(v) for v in conc]
                ends = var(*conc_cst, op="Concat", axis=0)
            else:
                ends = conc[0]
        else:
            ends = np.array(ends, dtype=np.int64)

        sliced_args = [starts, ends, axes]
        if steps is not None:
            sliced_args.append(steps)
        sliced_args_cst = [v if isinstance(v, Var) else cst(v) for v in sliced_args]
        sliced = var(self.self_var, *sliced_args_cst, op="Slice")
        if axis_squeeze:
            return var(
                sliced,
                cst(np.array(axis_squeeze, dtype=np.int64)),
                op="Squeeze",
            )
        return sliced

    def __setitem__(self, index, values):
        new_op = self.set[index](values)
        self.current_var_ = new_op
        self.input_indices = None


class Input(Var):
    """
    Defines an input, a placeholder.

    :param name: input name or None if undefined
    :param annotation: annotation if any is available
    """

    def __init__(self, name: Optional[str] = None, annotation: Optional[type] = None):
        Var.__init__(self)
        self.name = name
        self._prefix = name or "I"
        self._annotation = annotation

    def __repr__(self):
        if self.annotation is None:
            return f"{self.__class__.__name__}({self.name!r})"
        return (
            f"{self.__class__.__name__}({self.name!r}, "
            f"{self._annotation.__name__!r})"
        )

    @property
    def annotation(self):
        return self._annotation


class Cst(Var):
    """
    Defines a constant.
    """

    def __init__(self, cst: Any):
        if isinstance(cst, np.ndarray):
            Var.__init__(self, cst, op="Identity")
        elif isinstance(cst, bool):
            Var.__init__(self, np.array(cst, dtype=np.bool_), op="Identity")
        elif isinstance(cst, int):
            Var.__init__(self, np.array(cst, dtype=np.int64), op="Identity")
        elif isinstance(cst, float):
            Var.__init__(self, np.array(cst, dtype=np.float64), op="Identity")
        elif isinstance(cst, complex):
            Var.__init__(self, np.array(cst, dtype=np.complex128), op="Identity")
        elif isinstance(cst, list):
            if all(isinstance(t, bool) for t in cst):
                Var.__init__(self, np.array(cst, dtype=np.bool_), op="Identity")
            elif all(isinstance(t, (int, bool)) for t in cst):
                Var.__init__(self, np.array(cst, dtype=np.int64), op="Identity")
            elif all(isinstance(t, (float, int, bool)) for t in cst):
                Var.__init__(self, np.array(cst, dtype=np.float64), op="Identity")
            elif all(isinstance(t, (float, int, bool, complex)) for t in cst):
                Var.__init__(self, np.array(cst, dtype=np.complex128), op="Identity")
            else:
                raise ValueError(
                    f"Unable to convert cst (type={type(cst)}), value={cst}."
                )
        else:
            raise NotImplementedError(
                f"Constant of type {type(cst)} are not implemented yet. "
                f"You should not use 'float32(x)' but 'array(x, dtype=float32)'."
            )
        self._prefix = "cst"
