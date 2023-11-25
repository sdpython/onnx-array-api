from typing import Any, Dict, List, Optional, Union
from enum import IntEnum
import numpy as np
from onnx import NodeProto, SparseTensorProto, TensorProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
    make_tensor_type_proto,
)
from onnx.numpy_helper import from_array
from ..ext_test_case import is_azure, is_windows
from .annotations import (
    elem_type_int,
    make_shape,
    GRAPH_PROTO,
    ELEMENT_TYPE,
    SHAPE_TYPE,
    VAR_CONSTANT_TYPE,
)


class ProtoType(IntEnum):
    """
    The same code can be used to output a GraphProto, a FunctionProto or a ModelProto.
    This class specifies the output type at the beginning of the code.
    """

    FUNCTION = 1
    GRAPH = 2
    MODEL = 3


class OnnxGraph:
    """
    Contains every piece needed to create an onnx model in a single instructions.
    This API is meant to be light and allows the description of a graph.

    :param opset: main opset version
    :param opsets: other opsets as a dictionary
    :param is_function: a :class:`onnx.ModelProto` or a :class:`onnx.FunctionProto`
    """

    def __init__(
        self,
        opset: Optional[int] = None,
        opsets: Optional[Dict[str, int]] = None,
        proto_type: ProtoType = ProtoType.MODEL,
    ):
        if opsets is not None and "" in opsets:
            if opset is None:
                opset = opsets[""]
            elif opset != opsets[""]:
                raise ValueError(
                    "The main opset can be specified twice with different values."
                )
        if proto_type == ProtoType.FUNCTION:
            raise NotImplementedError(
                "The first version of this API does not support functions."
            )
        self.proto_type = proto_type
        self.opsets = opsets
        self.opset = opset
        self.nodes: List[Union[NodeProto, TensorProto]] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[TensorProto] = []
        self.unique_names_: Dict[str, Any] = {}
        self.renames_: Dict[str, str] = {}

    @property
    def is_function(self) -> bool:
        return self.proto_type == ProtoType.FUNCTION

    def __repr__(self) -> str:
        "usual"
        sts = [f"{self.__class__.__name__}("]
        els = [
            repr(getattr(self, o))
            for o in ["opset", "opsets"]
            if getattr(self, o) is not None
        ]
        if self.is_function:
            els.append("is_function=True")
        sts.append(", ".join(els))
        sts.append(")")
        return "".join(sts)

    @property
    def input_names(self) -> List[str]:
        "Returns the input names"
        return [v.name for v in self.inputs]

    @property
    def output_names(self) -> List[str]:
        "Returns the output names"
        return [v.name for v in self.outputs]

    def has_name(self, name: str) -> bool:
        "Tells if a name is already used."
        return name in self.unique_names_

    def unique_name(self, prefix="r", value: Optional[Any] = None) -> str:
        """
        Returns a unique name.

        :param prefix: prefix
        :param value: this name is mapped to this value
        :return: unique name
        """
        if isinstance(value, int):
            raise TypeError(f"Unexpected type {type(value)}, prefix={prefix!r}.")
        name = prefix
        i = len(self.unique_names_)
        while name in self.unique_names_:
            name = f"prefix{i}"
            i += 1
        self.unique_names_[name] = value
        return name

    def make_input(
        self,
        name: str,
        elem_type: ELEMENT_TYPE = TensorProto.FLOAT,
        shape: Optional[SHAPE_TYPE] = None,
    ) -> ValueInfoProto:
        """
        Adds an input to the graph.

        :param name: input name
        :param elem_type: element type (the input is assumed to be a tensor)
        :param shape: shape
        :return: an instance of ValueInfoProto
        """
        if self.has_name(name):
            raise ValueError(f"Name {name!r} is already taken.")
        var = make_tensor_value_info(name, elem_type, shape)
        self.inputs.append(var)
        self.unique_names_[name] = var
        return var

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
        from .var import Var

        proto = self.make_input(name, elem_type=elem_type_int(elem_type), shape=shape)
        return Var(
            self,
            proto.name,
            elem_type=proto.type.tensor_type.elem_type,
            shape=make_shape(proto.type.tensor_type.shape),
        )

    def make_output(
        self,
        name: str,
        elem_type: ELEMENT_TYPE = TensorProto.FLOAT,
        shape: Optional[SHAPE_TYPE] = None,
    ) -> ValueInfoProto:
        """
        Adds an output to the graph.

        :param name: input name
        :param elem_type: element type (the input is assumed to be a tensor)
        :param shape: shape
        :return: an instance of ValueInfoProto
        """
        if not self.has_name(name):
            raise ValueError(f"Name {name!r} does not exist.")
        var = make_tensor_value_info(name, elem_type_int(elem_type), shape)
        self.outputs.append(var)
        self.unique_names_[name] = var
        return var

    def make_constant(
        self, value: np.ndarray, name: Optional[str] = None
    ) -> TensorProto:
        "Adds an initializer to the graph."
        if self.is_function:
            raise NotImplementedError(
                "Adding a constant to a FunctionProto is not supported yet."
            )
        if isinstance(value, np.ndarray):
            if name is None:
                name = self.unique_name()
            elif self.has_name(name):
                raise RuntimeError(f"Name {name!r} already exists.")
            tensor = from_array(value, name=name)
            self.unique_names_[name] = tensor
            self.initializers.append(tensor)
            return tensor
        raise TypeError(f"Unexpected type {type(value)} for constant {name!r}.")

    def make_node(
        self,
        op_type: str,
        *inputs: List[VAR_CONSTANT_TYPE],
        domain: str = "",
        n_outputs: int = 1,
        output_names: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> NodeProto:
        """
        Creates a node.

        :param op_type: operator type
        :param inputs: others inputs
        :param domain: domain
        :param n_outputs: number of outputs
        :param output_names: output names, if not specified, outputs are given
            unique names
        :param kwargs: node attributes
        :return: NodeProto
        """
        if output_names is None:
            output_names = [
                self.unique_name(prefix=f"r{len(self.nodes)}_{i}")
                for i in range(n_outputs)
            ]
        elif n_outputs != len(output_names):
            raise ValueError(
                f"Expecting {n_outputs} outputs but received {output_names}."
            )
        input_names = []
        for i in inputs:
            if hasattr(i, "name"):
                input_names.append(i.name)
            elif isinstance(i, np.ndarray):
                input_names.append(self.make_constant(i))
            else:
                raise TypeError(f"Unexpected type {type(i)} for one input.")

        node = make_node(op_type, input_names, output_names, domain=domain, **kwargs)
        self.nodes.append(node)
        if domain != "":
            if not self.opsets or domain not in self.opsets:
                raise RuntimeError(f"No opset value was given for domain {domain!r}.")
        return node

    def cst(self, value: np.ndarray, name: Optional[str] = None) -> "Var":
        """
        Adds an initializer

        :param value: constant tensor
        :param name: input name
        :return: instance of :class:`onnx_array_api.light_api.Var`
        """
        from .var import Var

        c = self.make_constant(value, name=name)
        return Var(self, c.name, elem_type=c.data_type, shape=tuple(c.dims))

    def true_name(self, name: str) -> str:
        """
        Some names were renamed. If name is one of them, the function
        returns the new name.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Unexpected type {type(name)}, rename must be placed before vout."
            )
        while name in self.renames_:
            name = self.renames_[name]
        return name

    def get_var(self, name: str) -> "Var":
        from .var import Var

        tr = self.true_name(name)
        proto = self.unique_names_[tr]
        if proto is None:
            return Var(self, name)
        if isinstance(proto, ValueInfoProto):
            return Var(
                self,
                proto.name,
                elem_type=proto.type.tensor_type.elem_type,
                shape=make_shape(proto.type.tensor_type.shape),
            )
        if isinstance(proto, TensorProto):
            return Var(
                self, proto.name, elem_type=proto.data_type, shape=tuple(proto.dims)
            )
        raise TypeError(f"Unexpected type {type(proto)} for name {name!r}.")

    def rename(self, old_name: str, new_name: str):
        """
        Renames a variable. The renaming does not
        change anything but is stored in a container.

        :param old_name: old name
        :param new_name: new name
        """
        if not self.has_name(old_name):
            raise RuntimeError(f"Name {old_name!r} does not exist.")
        if self.has_name(new_name):
            raise RuntimeError(f"Name {old_name!r} already exist.")
        value = self.unique_names_[old_name]
        if isinstance(value, int):
            raise TypeError(
                f"Unexpected type {type(value)} for value {old_name!r} renamed into {new_name!r}."
            )
        self.unique_names_[new_name] = value
        self.renames_[old_name] = new_name

    def _fix_name_tensor(
        self, obj: Union[TensorProto, SparseTensorProto, ValueInfoProto]
    ) -> Union[TensorProto, SparseTensorProto, ValueInfoProto]:
        true_name = self.true_name(obj.name)
        if true_name != obj.name:
            obj.name = true_name
        return obj

    def _fix_name_tensor_input(
        self, obj: Union[TensorProto, SparseTensorProto, ValueInfoProto]
    ) -> Union[TensorProto, SparseTensorProto, ValueInfoProto]:
        obj = self._fix_name_tensor(obj)
        shape = make_shape(obj.type.tensor_type.shape)
        if shape is None:
            tensor_type_proto = make_tensor_type_proto(
                obj.type.tensor_type.elem_type, []
            )
            obj.type.CopyFrom(tensor_type_proto)
        return obj

    def _fix_name_tensor_output(
        self, obj: Union[TensorProto, SparseTensorProto, ValueInfoProto]
    ) -> Union[TensorProto, SparseTensorProto, ValueInfoProto]:
        obj = self._fix_name_tensor(obj)
        shape = make_shape(obj.type.tensor_type.shape)
        if shape is None:
            tensor_type_proto = make_tensor_type_proto(
                obj.type.tensor_type.elem_type, []
            )
            obj.type.CopyFrom(tensor_type_proto)
        return obj

    def _fix_name_node(self, obj: NodeProto) -> NodeProto:
        new_inputs = [self.true_name(i) for i in obj.input]
        if new_inputs != obj.input:
            del obj.input[:]
            obj.input.extend(new_inputs)
        new_outputs = [self.true_name(o) for o in obj.output]
        if new_outputs != obj.output:
            del obj.output[:]
            obj.output.extend(new_outputs)
        return obj

    def _check_input(self, i):
        "Checks one input is fully specified."
        if i.type.tensor_type.elem_type <= 0:
            raise ValueError(f"Input {i.name!r} has no element type.")
        return i

    def to_onnx(self) -> GRAPH_PROTO:
        """
        Converts the graph into an ONNX graph.
        """
        if self.is_function:
            raise NotImplementedError("Unable to convert a graph input ")
        dense = [
            self._fix_name_tensor(i)
            for i in self.initializers
            if isinstance(i, TensorProto)
        ]
        sparse = [
            self._fix_name_tensor(i)
            for i in self.initializers
            if isinstance(i, SparseTensorProto)
        ]
        graph = make_graph(
            [self._fix_name_node(n) for n in self.nodes],
            "light_api",
            [self._check_input(self._fix_name_tensor_input(i)) for i in self.inputs],
            [self._fix_name_tensor_output(o) for o in self.outputs],
            dense,
            sparse,
        )
        opsets = [make_opsetid("", self.opset or onnx_opset_version() - 1)]
        if self.opsets:
            for k, v in self.opsets.items():
                opsets.append(make_opsetid(k, v))
        if self.proto_type == ProtoType.GRAPH:
            # If no opsets, it a subgraph, not a model.
            return graph
        model = make_model(graph, opset_imports=opsets)
        if not is_windows() or not is_azure():
            # check_model fails sometimes on Windows
            check_model(model)
        return model
