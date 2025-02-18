from typing import Any, Dict, List
from onnx import TensorProto
from onnx.numpy_helper import to_array
from .base_emitter import BaseEmitter

_types = {
    TensorProto.FLOAT: "FLOAT",
    TensorProto.FLOAT16: "FLOAT16",
    TensorProto.INT64: "INT64",
    TensorProto.INT32: "INT32",
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
        return []

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        inps = ", ".join(["g.op", *[f'"{i}"' for i in self.inputs]])
        inputs = []
        for inp, stype, shape in self.inputs_full_:
            inputs.append(f'g.make_tensor_input("{inp}", TensorProto.{stype}, {shape})')
        outputs = []
        for inp, stype, shape in self.outputs_full_:
            outputs.append(
                f'g.make_tensor_output("{inp}", TensorProto.{stype}, {shape})'
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
            rows.append(f"    {init.name} = np.array({val.tolist()}, dtype=np.{stype})")
        return rows

    def _emit_begin_return(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_return(self, **kwargs: Dict[str, Any]) -> List[str]:
        outs = ", ".join(self.outputs)
        return [f"    return {outs}"]

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        itype = kwargs.get("elem_type", 0)
        shape = kwargs.get("shape", None)
        self.outputs.append(name)
        self.outputs_full_.append((name, _itype_to_string(itype), shape))
        return [f'    op.Identity({name}, outputs=["{name}"])']

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

        outs = ", ".join(outputs)
        inps = ", ".join(inputs)
        if args:
            sargs = ", ".join(args)
            row = f"    {outs} = op.{op_type}({inps}, {sargs})"
        else:
            row = f"    {outs} = op.{op_type}({inps})"
        return [row]
