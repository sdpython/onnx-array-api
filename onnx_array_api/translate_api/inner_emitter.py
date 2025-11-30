from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from onnx import AttributeProto
from ..annotations import ELEMENT_TYPE_NAME
from .base_emitter import BaseEmitter
from .translate import Translater


class InnerEmitter(BaseEmitter):
    """Converts event into proper code."""

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
                if "oh.make_model" in line:
                    break
                new_rows.append("    " + line)
            new_rows.append("    return graph")
            new_rows.append(f"{value[0].name} = _make_local_graph_{value[0].name}()")
            return new_rows, value[0].name

        return super().render_attribute_value(value)

    def _make_attribute(
        self, name: str, attr_type: int, ref_attr_name: Optional[str] = None
    ) -> str:
        assert (
            ref_attr_name is not None
        ), f"Cannot create attribute with name={name!r}, attr_type={attr_type}."
        return (
            f"make_ref_attribute(key={name!r}, attr_type={attr_type}, "
            f"ref_attr_name={ref_attr_name!r})"
        )

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Returns the separators. `single_line` is unused."
        return "\n".join(rows)

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = ["opset_imports = ["]
        opsets = kwargs.get("opsets", {})
        for k, v in opsets.items():
            lines.append(f"    oh.make_opsetid({k!r}, {v!r}),")
        lines.append("]")
        return lines

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "model = oh.make_model(",
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
            "graph = oh.make_graph(",
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
        fra = "onh.from_array"
        sdtype = repl.get(str(value.dtype), str(value.dtype))
        sdtype = f"np.{sdtype}" if hasattr(np, sdtype) else f"ml_dtypes.{sdtype}"

        return [
            "initializers.append(",
            f"    {fra}(",
            f"        np.array({value.tolist()}, dtype={sdtype}),",
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
                f"{container}.append(oh.make_tensor_value_info({name!r}, "
                f"onnx.TensorProto.{ELEMENT_TYPE_NAME[elem_type]}, shape={shape!r}))"
            ]
        if elem_type:
            return [
                f"{container}.append(oh.make_tensor_value_info({name!r}, "
                f"onnx.TensorProto.{ELEMENT_TYPE_NAME[elem_type]}, shape=[]))"
            ]
        return [
            f"{container}.append(oh.make_tensor_value_info({name!r}, "
            f"onnx.TensorProto.UNDEFINED, []))"
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
            "    make_node_extended(",
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

    def _emit_begin_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "",
            f"name_f = {kwargs['name']!r}",
            f"domain_f = {kwargs['domain']!r}",
            "nodes = []",
            "inputs = []",
            "outputs = []",
            "atts = []",
        ]
        return lines

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_function_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        return [f"inputs.append({kwargs['name']!r})"]

    def _emit_function_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        return [f"outputs.append({kwargs['name']!r})"]

    def _emit_function_attributes(self, **kwargs: Dict[str, Any]) -> List[str]:
        atts = kwargs["attributes"]
        if isinstance(atts, list) and all(isinstance(t, str) for t in atts):
            return [f"atts.extend({atts!r})"]
        raise NotImplementedError(f"Unable to process function attributes {atts!r}.")

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "functions.append(",
            "    oh.make_function(",
            "        domain_f, ",
            "        name_f, ",
            "        inputs, ",
            "        outputs, ",
            "        nodes, ",
            "        attributes=atts, ",
            "        opset_imports=opset_imports,",
            "   )",
            ")",
        ]
        return lines


class InnerEmitterShortInitializer(InnerEmitter):
    """
    Converts event into proper code.
    Initializer are replaced by random values if too big.
    """

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        value = kwargs["value"]
        repl = {"bool": "bool_", "object": "object_", "str": "str_"}
        fra = "onh.from_array"
        sdtype = repl.get(str(value.dtype), str(value.dtype))
        sdtype = f"np.{sdtype}" if hasattr(np, sdtype) else f"ml_dtypes.{sdtype}"
        if value.size <= 16:
            return [
                "initializers.append(",
                f"    {fra}(",
                f"        np.array({value.tolist()}, dtype={sdtype}),",
                f"        name={name!r}",
                "    )",
                ")",
            ]
        if "int" in sdtype:
            return [
                f"value = np.random.randint(0, 10, size={value.shape})"
                f".astype({sdtype})",
                "initializers.append(",
                f"    {fra}(",
                f"        np.array(value, dtype={sdtype}),",
                f"        name={name!r}",
                "    )",
                ")",
            ]
        return [
            f"value = np.random.randn({', '.join(map(str,value.shape))})"
            f".astype({sdtype})",
            "initializers.append(",
            f"    {fra}(",
            f"        np.array(value, dtype={sdtype}),",
            f"        name={name!r}",
            "    )",
            ")",
        ]
