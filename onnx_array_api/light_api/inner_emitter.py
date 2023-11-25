from typing import Any, Dict, List, Tuple
from onnx import AttributeProto
from .annotations import ELEMENT_TYPE_NAME
from .emitter import BaseEmitter
from .translate import Translater


class InnerEmitter(BaseEmitter):
    """
    Converts event into proper code.
    """

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

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Returns the separators. `single_line` is unused."
        return "\n".join(rows)

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = ["opset_imports = ["]
        opsets = kwargs.get("opsets", {})
        for k, v in opsets.items():
            lines.append(f"    make_opsetid({k!r}, {v!r}),")
        lines.append("]")
        return lines

    def _emit_to_onnx(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "model = make_model(",
            "    graph,",
            "    functions=functions,",
            "    opset_imports=opset_imports",
            ")",
        ]
        return lines

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "inputs = []",
            "outputs = []",
            "nodes = []",
            "initializers = []",
            "sparse_initializers = []",
            "functions = []",
        ]
        return lines

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs.get("name", "noname")
        lines = [
            "graph = make_graph(",
            "    nodes,",
            f"    {name!r},",
            "    inputs,",
            "    outputs,",
            "    initializers,",
            "    sparse_initializer=sparse_initializers,",
            ")",
        ]
        return lines

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        value = kwargs["value"]
        repl = {"bool": "bool_", "object": "object_", "str": "str_"}
        sdtype = repl.get(str(value.dtype), str(str(value.dtype)))
        return [
            "initializers.append(",
            "    from_array(",
            f"        np.array({value.tolist()}, dtype=np.{sdtype}),",
            f"        name={name!r}",
            "    )",
            ")",
        ]

    def _emit_io(self, container: str, **kwargs: Dict[str, Any]) -> List[str]:
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
        return [
            f"{container}.append(make_tensor_value_info({name!r}, TensorProto.UNDEFINED, []))"
        ]

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        return self._emit_io("inputs", **kwargs)

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        return self._emit_io("outputs", **kwargs)

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        op_type = kwargs["op_type"]
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        if kwargs.get("domain", "") != "":
            domain = kwargs["domain"]

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
