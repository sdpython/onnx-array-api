from typing import Any, Dict, List, Tuple
from enum import IntEnum
import numpy as np
from .annotations import ELEMENT_TYPE_NAME


class EventType(IntEnum):
    START = 0
    INPUT = 1
    OUTPUT = 2
    NODE = 3
    TO_ONNX = 4
    BEGIN_GRAPH = 5
    END_GRAPH = 6
    BEGIN_FUNCTION = 7
    END_FUNCTION = 8
    INITIALIZER = 9


class Emitter:
    """
    Converts event into proper code.
    """

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Join the rows"
        if single_line:
            return ".".join(rows)
        return "".join(["(\n    ", "\n    .".join(rows), "\n)"])

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

        if event == EventType.BEGIN_GRAPH:
            return []

        if event == EventType.END_GRAPH:
            return []

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
                before, vatt = self.render_attribute_value(v)
                if before:
                    raise NotImplementedError("Graph attribute not supported yet.")
                args.append(f"{k}={vatt}")

            str_inputs = ", ".join([f"{i!r}" for i in inputs])
            inst = [f"bring({str_inputs})", f"{op_type}({', '.join(args)})"]
            if len(outputs) == 1:
                inst.append(f"rename({outputs[0]!r})")
            else:
                str_outputs = ", ".join([f"{o!r}" for o in outputs])
                inst.append(f"rename({str_outputs})")
            return inst

        raise ValueError(f"Unexpected EventType {event}.")

    def render_attribute_value(self, value: Any) -> Tuple[List[str], str]:
        """
        Renders an attribute value into a string.

        :param value: value to converter
        :return: rows to append before, actual value
        """
        v = value[-1]
        if isinstance(v, (int, float, list)):
            return [], str(v)
        if isinstance(v, np.ndarray):
            if len(v.shape) == 0:
                return [], str(v)
            if len(v.shape) == 1:
                return [], str(v.tolist())
        raise ValueError(f"Unable to render an attribute {value}.")
