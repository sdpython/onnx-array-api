import inspect
from typing import Any, Dict, List, Optional, Tuple
from enum import IntEnum
import numpy as np
from onnx import AttributeProto


class EventType(IntEnum):
    START = 0
    INPUT = 1
    OUTPUT = 2
    NODE = 3
    TO_ONNX_MODEL = 4
    BEGIN_GRAPH = 5
    END_GRAPH = 6
    BEGIN_FUNCTION = 7
    END_FUNCTION = 8
    INITIALIZER = 9
    SPARSE_INITIALIZER = 10
    FUNCTION_INPUT = 11
    FUNCTION_OUTPUT = 12
    FUNCTION_ATTRIBUTES = 13
    TO_ONNX_FUNCTION = 14

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

        if event == EventType.TO_ONNX_MODEL:
            return self._emit_to_onnx_model(**kwargs)

        if event == EventType.TO_ONNX_FUNCTION:
            return self._emit_to_onnx_function(**kwargs)

        if event == EventType.BEGIN_GRAPH:
            return self._emit_begin_graph(**kwargs)

        if event == EventType.END_GRAPH:
            return self._emit_end_graph(**kwargs)

        if event == EventType.BEGIN_FUNCTION:
            return self._emit_begin_function(**kwargs)

        if event == EventType.END_FUNCTION:
            return self._emit_end_function(**kwargs)

        if event == EventType.FUNCTION_INPUT:
            return self._emit_function_input(**kwargs)

        if event == EventType.FUNCTION_OUTPUT:
            return self._emit_function_output(**kwargs)

        if event == EventType.FUNCTION_ATTRIBUTES:
            return self._emit_function_attributes(**kwargs)

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

        if isinstance(value, tuple) and len(value) == 2 and value[1] is None:
            # in a function, an attribute receiving a value from an attribute
            v = value[0]
            name = v.name
            ref = v.ref_attr_name
            dt = v.type
            return [], self._make_attribute(name=name, ref_attr_name=ref, attr_type=dt)

        raise ValueError(
            f"Unable to render an attribute {type(v)}, "
            f"attribute type={value[0].type}, "
            f"dtype={getattr(v, 'dtype', '-')}, "
            f"shape={getattr(v, 'shape', '-')}, type(value)={type(value)}, "
            f"value={value!r}."
        )

    def _make_attribute(
        self, name: str, attr_type: int, ref_attr_name: Optional[str] = None
    ) -> str:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def join(self, rows: List[str], single_line: bool = False) -> str:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
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

    def _emit_begin_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_function_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_function_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )

    def _emit_function_attributes(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError(
            f"Method {inspect.currentframe().f_code.co_name!r} was not overloaded."
        )
