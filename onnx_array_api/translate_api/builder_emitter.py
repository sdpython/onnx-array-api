from typing import Any, Dict, List
from onnx import TensorProto
from onnx.numpy_helper import to_array
from .base_emitter import BaseEmitter

_types = {
    TensorProto.DOUBLE: "DOUBLE",
    TensorProto.FLOAT: "FLOAT",
    TensorProto.FLOAT16: "FLOAT16",
    TensorProto.INT64: "INT64",
    TensorProto.INT32: "INT32",
    TensorProto.INT16: "INT16",
    TensorProto.UINT64: "UINT64",
    TensorProto.UINT32: "UINT32",
    TensorProto.UINT16: "UINT16",
    TensorProto.STRING: "STRING",
    TensorProto.BOOL: "BOOL",
}


def _itype_to_string(itype: int) -> str:
    return _types[itype]


class BuilderEmitter(BaseEmitter):
    """
    Converts event into proper code.
    """

    def __init__(self, make_model_function: str = ""):
        super().__init__()
        self.make_model_function = make_model_function

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Join the rows"
        assert (
            not single_line
        ), f"The emitter {type(self)} does not work with single_line=True."
        return "\n".join(rows)

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.opsets = kwargs.get("opsets", {})
        self.ir_version = kwargs.get("ir_version", None)
        self.function_calls = []
        return []

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        inps = ", ".join(["g.op", *[f'"{i}"' for i in self.inputs]])
        inputs = []
        for inp, stype, shape in self.inputs_full_:
            inputs.append(f'g.make_tensor_input("{inp}", TensorProto.{stype}, {shape})')
        outputs = []
        for inp, stype, shape in self.outputs_full_:
            outputs.append(
                f'g.make_tensor_output("{inp}", TensorProto.{stype}, '
                f"{shape}, is_dimension=False, indexed=False)"
            )
        rows = [
            "",
            (
                f"g = GraphBuilder({self.opsets}, ir_version={self.ir_version})"
                if self.ir_version
                else f"GraphBuilder({self.opsets})"
            ),
            *inputs,
            f"{self.name}({inps})",
            *outputs,
            *self.function_calls,
            "model = g.to_onnx()",
        ]
        if self.make_model_function:
            rows = [
                "",
                "",
                f'def {self.make_model_function}() -> "ModelProto":',
                *["    " + _ for _ in rows[1:]],
                "    return model",
                "",
                "",
                f"model = {self.make_model_function}()",
            ]
        return rows

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.inputs = []
        self.inputs_full = []
        self.outputs = []
        self.inits = []
        self.inputs_full_ = []
        self.outputs_full_ = []
        self.name = kwargs.get("name", "make_graph")
        return []

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        init = kwargs["init"]
        if isinstance(init, TensorProto):
            assert (
                kwargs["name"] == init.name
            ), f"Name mismatch init.name={init.name!r}, name={kwargs['name']!r}"
            self.inits.append(init)
            return []
        raise AssertionError(f"Unsupported type for an initializer {type(init)}")

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        itype = kwargs.get("elem_type", 0)
        shape = kwargs.get("shape", None)
        name = self._clean_result_name(name)
        if itype == 0:
            inp = name or "X"
        else:
            if shape is None:
                inp = f'{name}: "{_itype_to_string(itype)}"'
            else:
                inp = (
                    f'{name}: "{_itype_to_string(itype)}[{", ".join(map(str, shape))}]"'
                )
        self.inputs_full.append(inp)
        self.inputs.append(name)
        self.inputs_full_.append((name, _itype_to_string(itype), shape))
        return []

    def _emit_begin_signature(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_signature(self, **kwargs: Dict[str, Any]) -> List[str]:
        rows = ["", f"def {self.name}(", '    op: "GraphBuilder",']
        for i in self.inputs_full:
            rows.append(f"    {i},")
        rows.append("):")
        for init in self.inits:
            val = to_array(init)
            stype = str(val.dtype).split(".")[-1]
            name = self._clean_result_name(init.name)
            rows.append(f"    {name} = np.array({val.tolist()}, dtype=np.{stype})")
        return rows

    def _emit_begin_return(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_return(self, **kwargs: Dict[str, Any]) -> List[str]:
        outs = ", ".join(self.outputs)
        return [f"    return {outs}"]

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        name = self._clean_result_name(name)
        itype = kwargs.get("elem_type", 0)
        shape = kwargs.get("shape", None)
        self.outputs.append(name)
        self.outputs_full_.append((name, _itype_to_string(itype), shape))
        return [f'    op.Identity({name}, outputs=["{name}"])']

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        op_type = kwargs["op_type"]
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        domain = kwargs.get("domain", "")
        atts = kwargs.get("atts", {})
        args = []
        for k, v in atts.items():
            before, vatt = self.render_attribute_value(v)
            if before:
                raise NotImplementedError("Graph attribute not supported yet.")
            args.append(f"{k}={vatt}")

        outs = ", ".join(map(self._clean_result_name, outputs))
        inps = ", ".join(map(self._clean_result_name, inputs))
        op_type = self._emit_node_type(op_type, domain)
        sdomain = "" if not domain else f", domain={domain!r}"
        if args:
            sargs = ", ".join(args)
            if inps:
                row = f"    {outs} = op.{op_type}({inps}, {sargs}{sdomain})"
            else:
                row = f"    {outs} = op.{op_type}({sargs}{sdomain})"
        else:
            row = f"    {outs} = op.{op_type}({inps}{sdomain})"
        return [row]

    def _clean_result_name(self, name):
        return name

    def _emit_node_type(self, op_type, domain):
        return op_type

    def _emit_begin_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.f_inputs = []
        self.f_outputs = []
        self.f_inits = []
        self.f_name = kwargs["name"]
        self.f_domain = kwargs["domain"]
        self.f_attributes = []
        self.f_opsets = kwargs["opsets"]
        return []

    def _emit_begin_function_signature(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_function_signature(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.f_call_name = f"make_{self.f_domain}_{self.f_name}"
        return [
            "",
            "",
            f'def {self.f_call_name}(g: "GraphBuilder"):',
            f"    gr = GraphBuilder({self.f_opsets}, as_function=True)",
            *[f"    {name} = gr.make_tensor_input({name!r})" for name in self.f_inputs],
            "    op = gr.op",
        ]

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        return ["    return gr"]

    def _emit_function_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.f_inputs.append(kwargs["name"])
        return []

    def _emit_function_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.f_outputs.append(kwargs["name"])
        return []

    def _emit_function_attributes(self, **kwargs: Dict[str, Any]) -> List[str]:
        raise NotImplementedError("Function attribute are not implemented yet.")

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.function_calls.append(f"{self.f_call_name}(g)")
        return [
            *[f"    gr.make_tensor_output({name})" for name in self.f_outputs],
            "    g.add_function(builder=gr)",
        ]

    def _emit_begin_function_return(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_function_return(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []
