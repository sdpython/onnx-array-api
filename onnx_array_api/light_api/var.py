from typing import Any, Dict, List, Optional, Union
import numpy as np
from onnx import TensorProto
from .annotations import (
    elem_type_int,
    make_shape,
    ELEMENT_TYPE,
    ELEMENT_TYPE_NAME,
    GRAPH_PROTO,
    SHAPE_TYPE,
    VAR_CONSTANT_TYPE,
)
from .model import OnnxGraph
from ._op_var import OpsVar
from ._op_vars import OpsVars


class BaseVar:
    """
    Represents an input, an initializer, a node, an output,
    multiple variables.

    :param parent: the graph containing the Variable
    """

    def __init__(
        self,
        parent: OnnxGraph,
    ):
        self.parent = parent

    def make_node(
        self,
        op_type: str,
        *inputs: List[VAR_CONSTANT_TYPE],
        domain: str = "",
        n_outputs: int = 1,
        output_names: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> Union["Var", "Vars"]:
        """
        Creates a node with this Var as the first input.

        :param op_type: operator type
        :param inputs: others inputs
        :param domain: domain
        :param n_outputs: number of outputs
        :param output_names: output names, if not specified, outputs are given
            unique names
        :param kwargs: node attributes
        :return: instance of :class:`onnx_array_api.light_api.Var` or
            :class:`onnx_array_api.light_api.Vars`
        """
        node_proto = self.parent.make_node(
            op_type,
            *inputs,
            domain=domain,
            n_outputs=n_outputs,
            output_names=output_names,
            **kwargs,
        )
        names = node_proto.output
        if len(names) == 1:
            return Var(self.parent, names[0])
        return Vars(*map(lambda v: Var(self.parent, v), names))

    def vin(
        self,
        name: str,
        elem_type: ELEMENT_TYPE = TensorProto.FLOAT,
        shape: Optional[SHAPE_TYPE] = None,
    ) -> "Var":
        """
        Declares a new input to the graph.

        :param name: input name
        :param elem_type: element_type
        :param shape: shape
        :return: instance of :class:`onnx_array_api.light_api.Var`
        """
        return self.parent.vin(name, elem_type=elem_type, shape=shape)

    def cst(self, value: np.ndarray, name: Optional[str] = None) -> "Var":
        """
        Adds an initializer

        :param value: constant tensor
        :param name: input name
        :return: instance of :class:`onnx_array_api.light_api.Var`
        """
        c = self.parent.make_constant(value, name=name)
        return Var(self.parent, c.name, elem_type=c.data_type, shape=tuple(c.dims))

    def vout(
        self,
        elem_type: ELEMENT_TYPE = TensorProto.FLOAT,
        shape: Optional[SHAPE_TYPE] = None,
    ) -> "Var":
        """
        Declares a new output to the graph.

        :param elem_type: element_type
        :param shape: shape
        :return: instance of :class:`onnx_array_api.light_api.Var`
        """
        output = self.parent.make_output(self.name, elem_type=elem_type, shape=shape)
        return Var(
            self.parent,
            output,
            elem_type=output.type.tensor_type.elem_type,
            shape=make_shape(output.type.tensor_type.shape),
        )

    def v(self, name: str) -> "Var":
        """
        Retrieves another variable than this one.

        :param name: name of the variable
        :return: instance of :class:`onnx_array_api.light_api.Var`
        """
        return self.parent.get_var(name)

    def bring(self, *vars: List[Union[str, "Var"]]) -> "Vars":
        """
        Creates a set of variable as an instance of
        :class:`onnx_array_api.light_api.Vars`.
        """
        return Vars(self.parent, *vars)

    def left_bring(self, *vars: List[Union[str, "Var"]]) -> "Vars":
        """
        Creates a set of variables as an instance of
        :class:`onnx_array_api.light_api.Vars`.
        `*vars` is added to the left, `self` is added to the right.
        """
        vs = [*vars, self]
        return Vars(self.parent, *vs)

    def right_bring(self, *vars: List[Union[str, "Var"]]) -> "Vars":
        """
        Creates a set of variables as an instance of
        :class:`onnx_array_api.light_api.Vars`.
        `*vars` is added to the right, `self` is added to the left.
        """
        vs = [self, *vars]
        return Vars(self.parent, *vs)

    def to_onnx(self) -> GRAPH_PROTO:
        "Creates the onnx graph."
        return self.parent.to_onnx()


class Var(BaseVar, OpsVar):
    """
    Represents an input, an initializer, a node, an output.

    :param parent: graph the variable belongs to
    :param name: input name
    :param elem_type: element_type
    :param shape: shape
    """

    def __init__(
        self,
        parent: OnnxGraph,
        name: str,
        elem_type: Optional[ELEMENT_TYPE] = 1,
        shape: Optional[SHAPE_TYPE] = None,
    ):
        BaseVar.__init__(self, parent)
        self.name_ = name
        self.elem_type = elem_type
        self.shape = shape

    @property
    def name(self):
        "Returns the name of the variable or the new name if it was renamed."
        return self.parent.true_name(self.name_)

    def __str__(self) -> str:
        "usual"
        s = f"{self.name}"
        if self.elem_type is None:
            return s
        s = f"{s}:{ELEMENT_TYPE_NAME[self.elem_type]}"
        if self.shape is None:
            return s
        return f"{s}:[{''.join(map(str, self.shape))}]"

    def rename(self, new_name: str) -> "Var":
        "Renames a variable."
        self.parent.rename(self.name, new_name)
        return self

    def to(self, to: ELEMENT_TYPE) -> "Var":
        "Casts a tensor into another element type."
        return self.Cast(to=elem_type_int(to))

    def astype(self, to: ELEMENT_TYPE) -> "Var":
        "Casts a tensor into another element type."
        return self.Cast(to=elem_type_int(to))

    def reshape(self, new_shape: VAR_CONSTANT_TYPE) -> "Var":
        "Reshapes a variable."
        if isinstance(new_shape, tuple):
            cst = self.cst(np.array(new_shape, dtype=np.int64))
            return self.bring(self, cst).Reshape()
        return self.bring(self, new_shape).Reshape()

    def __add__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Add()

    def __eq__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Equal()

    def __float__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Cast(to=TensorProto.FLOAT)

    def __gt__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Greater()

    def __ge__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).GreaterOrEqual()

    def __int__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Cast(to=TensorProto.INT64)

    def __lt__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Less()

    def __le__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).LessOrEqual()

    def __matmul__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).MatMul()

    def __mod__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Mod()

    def __mul__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Mul()

    def __ne__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Equal().Not()

    def __neg__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.Neg()

    def __pow__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Pow()

    def __sub__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Sub()

    def __truediv__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        "Intuitive."
        return self.bring(self, var).Div()


class Vars(BaseVar, OpsVars):
    """
    Represents multiple Var.

    :param parent: graph the variable belongs to
    :param vars: list of names or variables
    """

    def __init__(self, parent, *vars: List[Union[str, Var]]):
        BaseVar.__init__(self, parent)
        self.vars_ = []
        for v in vars:
            if isinstance(v, str):
                var = self.parent.get_var(v)
            else:
                var = v
            self.vars_.append(var)

    def __len__(self):
        "Returns the number of variables."
        return len(self.vars_)

    def _check_nin(self, n_inputs):
        if len(self) != n_inputs:
            raise RuntimeError(f"Expecting {n_inputs} inputs not {len(self)}.")
        return self

    def rename(self, new_name: str) -> "Var":
        "Renames variables."
        raise NotImplementedError("Not yet implemented.")
