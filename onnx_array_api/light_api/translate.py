from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto
from onnx.numpy_helper import to_array
from ..reference import to_array_extended
from .base_emitter import EventType
from .light_emitter import LightEmitter


class Translater:
    """
    Translates an ONNX graph into a code following the light API.
    """

    def __init__(
        self,
        proto: Union[ModelProto, FunctionProto, GraphProto],
        emitter: Optional[LightEmitter] = None,
    ):
        self.proto_ = proto
        self.emitter = emitter or LightEmitter()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<{type(self.proto_)})"

    def export(self, as_str, single_line: bool = False) -> Union[str, List[str]]:
        """
        Exports into a code.

        :param as_str: as a single string or by rows
        :param single_line: tries to compress the output into a single line
        :return: list of instructions
        """
        rows = []
        last_event = None
        if isinstance(self.proto_, ModelProto):
            opsets = {d.domain: d.version for d in self.proto_.opset_import}
            rows.extend(self.emitter(EventType.START, opsets=opsets))
            inputs = self.proto_.graph.input
            outputs = self.proto_.graph.output
            nodes = self.proto_.graph.node
            initializers = self.proto_.graph.initializer
            sparse_initializers = self.proto_.graph.sparse_initializer
            attributes = []
            last_event = EventType.TO_ONNX_MODEL
            is_function = False
        elif isinstance(self.proto_, (FunctionProto, GraphProto)):
            inputs = self.proto_.input
            outputs = self.proto_.output
            nodes = self.proto_.node
            if isinstance(self.proto_, GraphProto):
                initializers = self.proto_.initializer
                sparse_initializers = self.proto_.sparse_initializer
            else:
                initializers = []
                sparse_initializers = []
            attributes = (
                self.proto_.attribute if hasattr(self.proto_, "attribute") else []
            )
            is_function = isinstance(self.proto_, FunctionProto)
            last_event = (
                EventType.TO_ONNX_FUNCTION if is_function else EventType.TO_ONNX_MODEL
            )
        else:
            raise ValueError(f"Unexpected type {type(self.proto_)} for proto.")

        if sparse_initializers:
            raise NotImplementedError("Sparse initializer not supported yet.")

        if is_function:
            rows.extend(
                self.emitter(
                    EventType.BEGIN_FUNCTION,
                    name=self.proto_.name,
                    domain=self.proto_.domain,
                )
            )
        else:
            rows.extend(self.emitter(EventType.BEGIN_GRAPH))

        for i in initializers:
            rows.extend(
                self.emitter(
                    EventType.INITIALIZER,
                    name=i.name,
                    init=i,
                    value=to_array_extended(i),
                )
            )

        for i in inputs:
            if is_function:
                rows.extend(self.emitter(EventType.FUNCTION_INPUT, name=i))
            else:
                rows.extend(
                    self.emitter(
                        EventType.INPUT,
                        name=i.name,
                        elem_type=i.type.tensor_type.elem_type,
                        shape=tuple(
                            d.dim_value or d.dim_param
                            for d in i.type.tensor_type.shape.dim
                        ),
                    )
                )

        if is_function and attributes:
            rows.extend(
                self.emitter(EventType.FUNCTION_ATTRIBUTES, attributes=list(attributes))
            )

        for node in nodes:
            atts = self.extract_attributes(node)
            rows.extend(
                self.emitter(
                    EventType.NODE,
                    op_type=node.op_type,
                    inputs=node.input,
                    outputs=node.output,
                    domain=node.domain,
                    atts=atts,
                )
            )

        for o in outputs:
            if is_function:
                rows.extend(self.emitter(EventType.FUNCTION_OUTPUT, name=o))
            else:
                rows.extend(
                    self.emitter(
                        EventType.OUTPUT,
                        name=o.name,
                        elem_type=o.type.tensor_type.elem_type,
                        shape=tuple(
                            d.dim_value or d.dim_param
                            for d in o.type.tensor_type.shape.dim
                        ),
                    )
                )
        if isinstance(self.proto_, (GraphProto, FunctionProto)):
            name = self.proto_.name
        else:
            name = self.proto_.graph.name

        rows.extend(
            self.emitter(
                EventType.END_FUNCTION if is_function else EventType.END_GRAPH,
                name=name,
            )
        )

        if isinstance(self.proto_, ModelProto) and len(self.proto_.functions) > 0:
            for fu in self.proto_.functions:
                cl = self.__class__(fu, self.emitter)
                text = cl.export(False, single_line=False)
                rows.extend(text)

        rows.extend(self.emitter(last_event))
        if as_str:
            return self.emitter.join(rows, single_line=single_line)
        return rows

    def extract_attributes(
        self, node: NodeProto
    ) -> Dict[str, Tuple[AttributeProto, Any]]:
        """
        Extracts all atributes of a node.

        :param node: node proto
        :return: dictionary
        """
        atts: Dict[str, Tuple[AttributeProto, Any]] = {}
        for att in node.attribute:
            if hasattr(att, "ref_attr_name") and att.ref_attr_name:
                atts[att.name] = (att, None)
                continue
            if att.type == AttributeProto.INT:
                atts[att.name] = (att, att.i)
                continue
            if att.type == AttributeProto.FLOAT:
                atts[att.name] = (att, att.f)
                continue
            if att.type == AttributeProto.INTS:
                atts[att.name] = (att, np.array(att.ints))
                continue
            if att.type == AttributeProto.FLOATS:
                atts[att.name] = (att, np.array(att.floats, dtype=np.float32))
                continue
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                atts[att.name] = (att, None)
                continue
            if att.type == AttributeProto.SPARSE_TENSORS:
                atts[att.name] = (att, to_array(att.sparse_tensor))
                continue
            if att.type == AttributeProto.TENSOR:
                atts[att.name] = (att, to_array(att.t))
                continue
            if att.type == AttributeProto.TENSORS:
                atts[att.name] = (att, [to_array(t) for t in att.tensors])
                continue
            if att.type == AttributeProto.SPARSE_TENSORS:
                atts[att.name] = (att, [to_array(t) for t in att.sparse_tensors])
                continue
            if att.type == AttributeProto.STRING:
                atts[att.name] = (att, att.s.decode("utf-8"))
                continue
            if att.type == AttributeProto.STRINGS:
                atts[att.name] = (
                    att,
                    np.array([s.decode("utf-8") for s in att.strings]),
                )
                continue
            raise ValueError(
                f"Attribute {att.name!r} with type {att.type} cannot be extracted yet."
            )
        return atts
