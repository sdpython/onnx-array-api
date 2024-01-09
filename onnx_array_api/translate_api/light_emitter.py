from typing import Any, Dict, List
from ..annotations import ELEMENT_TYPE_NAME
from .base_emitter import BaseEmitter


class LightEmitter(BaseEmitter):
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

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        return ["to_onnx()"]

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

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
