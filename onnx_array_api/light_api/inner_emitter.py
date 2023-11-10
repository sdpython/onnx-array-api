from typing import Any, Dict, List, Tuple
from onnx import AttributeProto
from .annotations import ELEMENT_TYPE_NAME
from .translate import Emitter, EventType, Translater


class InnerEmitter(Emitter):
    """
    Converts event into proper code.
    """

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Returns the separators. `single_line` is unused."
        return "\n".join(rows)

    def __call__(self, event: EventType, **kwargs: Dict[str, Any]) -> List[str]:
        """
        Converts an event into an instruction.

        :param event: event kind
        :param kwargs: event parameters
        :return: list of instructions
        """
        if event == EventType.START:
            lines = ["opset_imports = ["]
            opsets = kwargs.get("opsets", {})
            for k, v in opsets.items():
                lines.append(f"    make_opsetid({k!r}, {v!r}),")
            lines.append("]")
            return lines

        if event == EventType.TO_ONNX:
            lines = [
                "model = make_model(",
                "    graph,",
                "    functions=functions,",
                "    opset_imports=opset_imports",
                ")",
            ]
            return lines

        if event == EventType.BEGIN_GRAPH:
            lines = [
                "inputs = []",
                "outputs = []",
                "nodes = []",
                "initializers = []",
                "sparse_initializers = []",
                "functions = []",
            ]
            return lines

        if event == EventType.END_GRAPH:
            lines = [
                "graph = make_graph(",
                "    nodes,",
                "    'noname',",
                "    inputs,",
                "    outputs,",
                "    initializers,",
                "    sparse_initializer=sparse_initializers,",
                ")",
            ]
            return lines

        if event in (EventType.INPUT, EventType.OUTPUT):
            container = "inputs" if event == EventType.INPUT else "outputs"
            name = kwargs["name"]
            elem_type = kwargs.get("elem_type", None)
            shape = kwargs.get("shape", None)
            if elem_type and shape:
                return [
                    f"{container}.append(make_tensor_value_info({name!r}, TensorProto.{ELEMENT_TYPE_NAME[elem_type]}, shape={shape!r}))"
                ]
            if elem_type:
                return [
                    f"{container}.append(make_tensor_value_info({name!r}, TensorProto.{ELEMENT_TYPE_NAME[elem_type]}, shape=[]))"
                ]
            return [f"{container}.append(make_tensor_value_info({name!r}))"]
        if event == EventType.NODE:
            op_type = kwargs["op_type"]
            inputs = kwargs["inputs"]
            outputs = kwargs["outputs"]
            if kwargs.get("domain", "") != "":
                domain = kwargs["domain"]
                raise NotImplementedError(f"domain={domain!r} not supported yet.")

            before_lines = []
            lines = [
                "nodes.append(",
                "    make_node(",
                f"        {op_type!r},",
                f"        {inputs},",
                f"        {outputs},",
            ]
            domain = kwargs.get("domain", "")
            if domain:
                lines.append(f"        domain={domain!r},")
            atts = kwargs.get("atts", {})
            for k, v in atts.items():
                before, value = self.render_attribute_value(v)
                before_lines.extend(before)
                lines.append(f"        {k}={value},")
            lines[-1] = lines[-1][:-1]
            lines.extend(["    )", ")"])
            return before_lines + lines

        raise ValueError(f"Unexpected EventType {event}.")

    def render_attribute_value(self, value: Any) -> Tuple[List[str], str]:
        """
        Renders an attribute value into a string.

        :param value: value to converter
        :return: rows to append before, actual value
        """
        if value[0].type == AttributeProto.GRAPH:
            tr = Translater(value[0].g, emitter=self)
            rows = tr.export(as_str=False, single_line=False)
            new_rows = [f"def _make_local_graph_{value[0].name}():"]
            for line in rows:
                if "make_model" in line:
                    break
                new_rows.append("    " + line)
            new_rows.append("    return graph")
            new_rows.append(f"{value[0].name} = _make_local_graph_{value[0].name}()")
            return new_rows, value[0].name

        return super().render_attribute_value(value)
