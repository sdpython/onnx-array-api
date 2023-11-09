from typing import Any, Dict, List, Optional, Tuple, Union
from enum import IntEnum
import numpy as np
from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto
from onnx.numpy_helper import to_array
from .annotations import ELEMENT_TYPE_NAME


class EventType(IntEnum):
    START = 0
    INPUT = 1
    OUTPUT = 2
    NODE = 3
    TO_ONNX = 4


class Emitter:
    """
    Converts event into proper code.
    """

    def __call__(self, event: EventType, **kwargs: Dict[str, Any]) -> List[str]:
        """
        Converts an event into an instruction.

        :param event: event kind
        :param kwargs: event parameters
        :return: list of instructions
        """
        if event == EventType.START:
            opsets = kwargs.get("opsets", {})
            opset = opsets.get("", None)
            if opset is not None:
                del opsets[""]
            args = []
            if opset:
                args.append(f"opset={opset}")
            if opsets:
                args.append(f"opsets={opsets}")
            return [f"start({', '.join(args)})"]

        if event == EventType.TO_ONNX:
            return ["to_onnx()"]

        if event == EventType.INPUT:
            name = kwargs["name"]
            elem_type = kwargs.get("elem_type", None)
            shape = kwargs.get("shape", None)
            if elem_type and shape:
                return [
                    f"vin({name!r}, elem_type=TensorProto.{ELEMENT_TYPE_NAME[elem_type]}, shape={shape!r})"
                ]
            if elem_type:
                return [
                    f"vin({name!r}, elem_type=TensorProto.{ELEMENT_TYPE_NAME[elem_type]})"
                ]
            return [f"vin({name!r})"]

        if event == EventType.OUTPUT:
            inst = []
            if "name" in kwargs:
                name = kwargs["name"]
                inst.append(f"bring({name!r})")
            elem_type = kwargs.get("elem_type", None)
            shape = kwargs.get("shape", None)
            if elem_type and shape:
                inst.append(
                    f"vout(elem_type=TensorProto.{ELEMENT_TYPE_NAME[elem_type]}, shape={shape!r})"
                )
            elif elem_type:
                inst.append(
                    f"vout(elem_type=TensorProto.{ELEMENT_TYPE_NAME[elem_type]})"
                )
            else:
                inst.append("vout()")
            return inst

        if event == EventType.NODE:
            op_type = kwargs["op_type"]
            inputs = kwargs["inputs"]
            outputs = kwargs["outputs"]
            if kwargs.get("domain", "") != "":
                domain = kwargs["domain"]
                raise NotImplementedError(f"domain={domain!r} not supported yet.")
            atts = kwargs.get("atts", {})
            args = []
            for k, v in atts.items():
                args.append(f"{k}={self.render_attribute_value(v)}")

            str_inputs = ", ".join([f"{i!r}" for i in inputs])
            inst = [f"bring({str_inputs})", f"{op_type}({', '.join(args)})"]
            if len(outputs) == 1:
                inst.append(f"rename({outputs[0]!r})")
            else:
                str_outputs = ", ".join([f"{o!r}" for o in outputs])
                inst.append(f"rename({str_outputs})")
            return inst

        raise ValueError(f"Unexpected EventType {event}.")

    def render_attribute_value(self, value: Any) -> str:
        """
        Renders an attribute value into a string.
        """
        v = value[-1]
        if isinstance(v, (int, float, list)):
            return str(v)
        if isinstance(v, np.ndarray):
            if len(v.shape) == 0:
                return str(v)
            if len(v.shape) == 1:
                return str(v.tolist())
        raise ValueError(f"Unable to render an attribute {value}.")


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
        self.emit = emitter or Emitter()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<{type(self.proto_)})"

    def export(self) -> List[str]:
        """
        Exports into a code.

        :return: list of instructions
        """
        rows = []
        if isinstance(self.proto_, ModelProto):
            opsets = {d.domain: d.version for d in self.proto_.opset_import}
            rows.extend(self.emit(EventType.START, opsets=opsets))
            inputs = self.proto_.graph.input
            outputs = self.proto_.graph.output
            nodes = self.proto_.graph.node
        elif isinstance(self.proto_, (FunctionProto, GraphProto)):
            inputs = self.proto_.input
            outputs = self.proto_.output
            nodes = self.proto_.node
        else:
            raise ValueError(f"Unexpected type {type(self.proto_)} for proto.")

        for i in inputs:
            if isinstance(i, str):
                rows.extend(self.emit(EventType.INPUT, name=i))
            else:
                rows.extend(
                    self.emit(
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
                self.emit(
                    EventType.NODE,
                    op_type=node.op_type,
                    inputs=node.input,
                    outputs=node.output,
                    domain=node.domain,
                    atts=atts,
                )
            )

        for o in outputs:
            if isinstance(i, str):
                rows.extend(self.emit(EventType.INPUT, name=o))
            else:
                rows.extend(
                    self.emit(
                        EventType.OUTPUT,
                        name=o.name,
                        elem_type=o.type.tensor_type.elem_type,
                        shape=tuple(
                            d.dim_value or d.dim_param
                            for d in o.type.tensor_type.shape.dim
                        ),
                    )
                )

        if isinstance(self.proto_, ModelProto) and len(self.proto_.functions) > 0:
            raise NotImplementedError("Local functions are not yet implemented.")

        rows.extend(self.emit(EventType.TO_ONNX))
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
