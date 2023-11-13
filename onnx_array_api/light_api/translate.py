from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto
from onnx.numpy_helper import to_array
from .emitter import EventType, Emitter


class Translater:
    """
    Translates an ONNX graph into a code following the light API.
    """

    def __init__(
        self,
        proto: Union[ModelProto, FunctionProto, GraphProto],
        emitter: Optional[Emitter] = None,
    ):
        self.proto_ = proto
        self.emitter = emitter or Emitter()

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
        if isinstance(self.proto_, ModelProto):
            opsets = {d.domain: d.version for d in self.proto_.opset_import}
            rows.extend(self.emitter(EventType.START, opsets=opsets))
            inputs = self.proto_.graph.input
            outputs = self.proto_.graph.output
            nodes = self.proto_.graph.node
            initializers = self.proto_.graph.initializer
            sparse_initializers = self.proto_.graph.sparse_initializer
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
        else:
            raise ValueError(f"Unexpected type {type(self.proto_)} for proto.")

        if sparse_initializers:
            raise NotImplementedError("Sparse initializer not supported yet.")

        rows.extend(
            self.emitter(
                EventType.BEGIN_FUNCTION
                if isinstance(self.proto_, FunctionProto)
                else EventType.BEGIN_GRAPH
            )
        )

        for i in initializers:
            rows.extend(
                self.emitter(
                    EventType.INITIALIZER, name=i.name, init=i, value=to_array(i)
                )
            )

        for i in inputs:
            if isinstance(i, str):
                rows.extend(self.emitter(EventType.INPUT, name=i))
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
            if isinstance(o, str):
                rows.extend(self.emitter(EventType.INPUT, name=o))
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
                EventType.END_FUNCTION
                if isinstance(self.proto_, FunctionProto)
                else EventType.END_GRAPH,
                name=name,
            )
        )

        if isinstance(self.proto_, ModelProto) and len(self.proto_.functions) > 0:
            raise NotImplementedError("Local functions are not yet implemented.")

        rows.extend(self.emitter(EventType.TO_ONNX))
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
