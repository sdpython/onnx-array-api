import inspect
from typing import Any, Dict, List, Tuple
from enum import IntEnum
import numpy as np
from onnx import AttributeProto
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
    SPARSE_INITIALIZER = 10

    @classmethod
    def to_str(cls, self) -> str:
        for k, v in EventType.__dict__.items():
            if self == v:
                return f"{cls.__name__}.{k}"


class BaseEmitter:
    def __call__(self, event: EventType, **kwargs: Dict[str, Any]) -> List[str]:
        """
        Converts an event into an instruction.

        :param event: event kind
        :param kwargs: event parameters
        :return: list of instructions
        """

        if event == EventType.NODE:
            return self._emit_node(**kwargs)

        if event == EventType.INITIALIZER:
            return self._emit_initializer(**kwargs)

        if event == EventType.SPARSE_INITIALIZER:
            return self._emit_sparse_initializer(**kwargs)

        if event == EventType.INPUT:
            return self._emit_input(**kwargs)

        if event == EventType.OUTPUT:
            return self._emit_output(**kwargs)

        if event == EventType.START:
            return self._emit_start(**kwargs)

        if event == EventType.TO_ONNX:
            return self._emit_to_onnx(**kwargs)

        if event == EventType.BEGIN_GRAPH:
            return self._emit_begin_graph(**kwargs)

        if event == EventType.END_GRAPH:
            return self._emit_end_graph(**kwargs)

        raise ValueError(f"Unexpected event {EventType.to_str(event)}.")

    def render_attribute_value(self, value: Any) -> Tuple[List[str], str]:
        """
        Renders an attribute value into a string.

        :param value: value to converter
        :return: rows to append before, actual value
        """
        v = value[-1]
        if value[0].type == AttributeProto.TENSOR:
            repl = {"bool": "bool_", "object": "object_", "str": "str_"}
            sdtype = repl.get(str(v.dtype), str(str(v.dtype)))
            return [], (
                f"from_array(np.array({v.tolist()}, dtype=np.{sdtype}), "
                f"name={value[0].name!r})"
            )
        if isinstance(v, (int, float, list)):
            return [], str(v)
        if isinstance(v, str):
            return [], f"{v!r}"
        if isinstance(v, np.ndarray):
            if not v.shape:
                return [], str(v)
            if len(v.shape) == 1:
                if value[0].type in (
                    AttributeProto.INTS,
                    AttributeProto.FLOATS,
                    AttributeProto.STRINGS,
                ):
                    return [], str(v.tolist())

        if value[0].type == AttributeProto.GRAPH:
            from .translate import Translater

            tr = Translater(value[0].g, emitter=self)
            rows = tr.export(as_str=False, single_line=False)
            # last instruction is to_onnx, let's drop it.
            srows = ".".join(rows[:-1])
            return [], f"g().{srows}"

        raise ValueError(
            f"Unable to render an attribute {type(v)}, "
            f"attribute type={value[0].type}, "
            f"dtype={getattr(v, 'dtype', '-')}, "
            f"shape={getattr(v, 'shape', '-')}, {value}."
        )

    def join(self, rows: List[str], single_line: bool = False) -> str:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_to_onnx(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_sparse_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )


class Emitter(BaseEmitter):
    """
    Converts event into proper code.
    """

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Join the rows"
        if single_line:
            return ".".join(rows)
        return "".join(["(\n    ", "\n    .".join(rows), "\n)"])

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
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

    def _emit_to_onnx(self, **kwargs: Dict[str, Any]) -> List[str]:
        return ["to_onnx()"]

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        value = kwargs["value"]
        repl = {"bool": "bool_", "object": "object_", "str": "str_"}
        sdtype = repl.get(str(value.dtype), str(str(value.dtype)))
        return [
            f"cst(np.array({value.tolist()}, dtype=np.{sdtype}))",
            f"rename({name!r})",
        ]

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
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

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
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
            inst.append(f"vout(elem_type=TensorProto.{ELEMENT_TYPE_NAME[elem_type]})")
        else:
            inst.append("vout()")
        return inst

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        op_type = kwargs["op_type"]
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        if kwargs.get("domain", "") != "":
            domain = kwargs["domain"]
            op_type = f"{domain}.{op_type}"
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
