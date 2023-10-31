from typing import Any, Dict, List, Optional, Tuple, Union
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


class Var:
    """
    Represents an input, an initializer, a node, an output.
    """

    def __init__(
        self,
        parent: OnnxGraph,
        name: str,
        elem_type: Optional[ELEMENT_TYPE] = 1,
        shape: Optional[SHAPE_TYPE] = None,
    ):
        self.name_ = name
        self.parent = parent
        self.elem_type = elem_type
        self.shape = shape

    @property
    def name(self):
        return self.parent.true_name(self.name_)

    def __str__(self) -> str:
        s = f"{self.name}"
        if self.elem_type is None:
            return s
        s = f"{s}:{ELEMENT_TYPE_NAME[self.elem_type]}"
        if self.shape is None:
            return s
        return f"{s}:[{''.join(map(str, self.shape))}]"

    # main function
    def make_node(
        self,
        op_type: str,
        *inputs: List[VAR_CONSTANT_TYPE],
        domain: str = "",
        n_outputs: int = 1,
        output_names: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> Union["Var", Tuple["Var"]]:
        """
        Creates a node with this Var as the first input.

        :param op_type: operator type
        :param inputs: others inputs
        :param domain: domain
        :param n_outputs: number of outputs
        :param output_names: output names, if not specified, outputs are given
            unique names
        :param kwargs: node attributes
        :return: Var or Tuple
        """
        node_proto = self.parent.make_node(
            op_type,
            self,
            *inputs,
            domain=domain,
            n_outputs=n_outputs,
            output_names=output_names,
            **kwargs,
        )
        names = node_proto.output
        if len(names) == 1:
            return Var(self.parent, names[0])
        return tuple(map(lambda v: Var(self.parent, v), names))

    def vout(self) -> "Var":
        """
        Creates an output.
        """
        output = self.parent.make_output(self.name)
        return Var(
            self.parent,
            output,
            elem_type=output.type.tensor_type.elem_type,
            shape=make_shape(output.type.tensor_type.shape),
        )

    def to_onnx(self) -> GRAPH_PROTO:
        "Creates the model."
        return self.parent.to_onnx()

    # shortcuts
    def rename(self, new_name: str) -> "Var":
        self.parent.rename(self.name, new_name)
        return self

    def to(self, to: ELEMENT_TYPE) -> "Var":
        "Casts a tensor into another element type."
        return self.Cast(to=elem_type_int(to))

    def astype(self, to: ELEMENT_TYPE) -> "Var":
        "Casts a tensor into another element type."
        return self.Cast(to=elem_type_int(to))

    def __add__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        return self.Add(var)

    def __sub__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        return self.Sub(var)

    def __mul__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        return self.Mul(var)

    def __div__(self, var: VAR_CONSTANT_TYPE) -> "Var":
        return self.Div(var)

    def __neg__(self) -> "Var":
        return self.Neg()

    # operators

    def Neg(self):
        return self.make_node("Neg")


def _complete_class_methods():
    pass


_complete_class_methods()
