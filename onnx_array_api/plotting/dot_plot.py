import re
from typing import Dict, Optional, Tuple

from onnx import GraphProto, ModelProto
from onnx.helper import tensor_dtype_to_string

from ..reference import to_array_extended as to_array
from ._helper import Graph, _get_shape, attributes_as_dict


def _type_to_string(dtype):
    """
    Converts a type into a readable string.
    """
    if dtype.HasField("tensor_type"):
        ttype = dtype.tensor_type
        return tensor_dtype_to_string(ttype.elem_type)
    if dtype.HasField("sequence_type"):
        stype = dtype.sequence_type
        return f"Sequence[{type(stype.elem_type)}]"
    raise ValueError(f"Unable to convert {dtype} into a string.")


def to_dot(
    proto: ModelProto,
    recursive: bool = False,
    prefix: str = "",
    use_onnx: bool = False,
    add_functions: bool = True,
    rt_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    **params,
) -> str:
    """
    Produces a :epkg:`DOT` language string for the graph.

    :param params: additional params to draw the graph
    :param recursive: also show subgraphs inside operator like `Scan`
    :param prefix: prefix for every node name
    :param use_onnx: use :epkg:`onnx` dot format instead of this one
    :param add_functions: add functions to the graph
    :param rt_shapes: indicates shapes obtained from the execution or inference
    :return: string

    Default options for the graph are:

    ::

        options = {
            'orientation': 'portrait',
            'ranksep': '0.25',
            'nodesep': '0.05',
            'width': '0.5',
            'height': '0.1',
            'size': '7',
        }

    One example:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning, FutureWarning
        :process:

        import numpy as np  # B
        from onnx_array_api.npx import absolute, jit_onnx
        from onnx_array_api.plotting.dot_plot import to_dot

        def l1_loss(x, y):
            return absolute(x - y).sum()


        def l2_loss(x, y):
            return ((x - y) ** 2).sum()


        def myloss(x, y):
            return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])


        jitted_myloss = jit_onnx(myloss)

        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)
        res = jitted_myloss(x, y)
        print(res)

    .. gdot::
        :script: DOT-SECTION
        :process:

        # to_dot
        import numpy as np
        from onnx_array_api.npx import absolute, jit_onnx
        from onnx_array_api.plotting.dot_plot import to_dot

        def l1_loss(x, y):
            return absolute(x - y).sum()


        def l2_loss(x, y):
            return ((x - y) ** 2).sum()


        def myloss(x, y):
            return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])


        jitted_myloss = jit_onnx(myloss)

        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)
        res = jitted_myloss(x, y)
        print(to_dot(jitted_myloss.get_onnx()))
    """
    clean_label_reg1 = re.compile("\\\\x\\{[0-9A-F]{1,6}\\}")
    clean_label_reg2 = re.compile("\\\\p\\{[0-9P]{1,6}\\}")

    def dot_name(text):
        return text.replace("/", "_").replace(":", "__").replace(".", "_")

    def dot_label(text):
        if text is None:
            return ""
        for reg in [clean_label_reg1, clean_label_reg2]:
            fall = reg.findall(text)
            for f in fall:
                text = text.replace(f, "_")
        return text

    options = {
        "orientation": "portrait",
        "ranksep": "0.25",
        "nodesep": "0.05",
        "width": "0.5",
        "height": "0.1",
        "size": "7",
    }
    options.update({k: v for k, v in params.items() if v is not None})

    if use_onnx:
        from onnx.tools.net_drawer import GetOpNodeProducer, GetPydotGraph

        pydot_graph = GetPydotGraph(
            proto.graph,
            name=proto.graph.name,
            rankdir=params.get("rankdir", "TB"),
            node_producer=GetOpNodeProducer(
                "docstring", fillcolor="orange", style="filled", shape="box"
            ),
        )
        return pydot_graph.to_string()

    inter_vars = {}
    exp = ["digraph{"]
    for opt in {"orientation", "pad", "nodesep", "ranksep", "size"}:
        if opt in options:
            exp.append(f"  {opt}={options[opt]};")
    fontsize = 10

    shapes = {}
    if rt_shapes:
        for name, shape in rt_shapes.items():
            va = str(shape.shape)
            shapes[name] = va

    # inputs
    exp.append("")
    graph = proto.graph if isinstance(proto, ModelProto) else proto
    for obj in graph.input:
        if isinstance(obj, str):
            exp.append(
                '  {2}{0} [shape=box color=red label="{0}" fontsize={1}];'
                "".format(obj, fontsize, prefix)
            )
            inter_vars[obj] = obj
        else:
            sh = _get_shape(obj)
            if sh:
                sh = f"\\nshape={sh}"
            exp.append(
                '  {3}{0} [shape=box color=red label="{0}\\n{1}{4}" fontsize={2}];'
                "".format(
                    obj.name, _type_to_string(obj.type), fontsize, prefix, dot_label(sh)
                )
            )
            inter_vars[obj.name] = obj

    # outputs
    exp.append("")
    for obj in graph.output:
        if isinstance(obj, str):
            exp.append(
                '  {2}{0} [shape=box color=green label="{0}" fontsize={1}];'.format(
                    obj, fontsize, prefix
                )
            )
            inter_vars[obj] = obj
        else:
            sh = _get_shape(obj)
            if sh:
                sh = f"\\nshape={sh}"
            exp.append(
                f"  {prefix}{obj.name} [shape=box color=green "
                f'label="{obj.name}\\n{_type_to_string(obj.type)}'
                f'{dot_label(sh)}" fontsize={fontsize}];'
            )
            inter_vars[obj.name] = obj

    # initializer
    exp.append("")
    if hasattr(proto, "graph"):
        inits = list(proto.graph.initializer) + list(proto.graph.sparse_initializer)
        for obj in inits:
            val = to_array(obj)
            flat = val.flatten()
            if flat.shape[0] < 9:
                st = str(val)
            else:
                st = str(val)
                if len(st) > 50:
                    st = st[:50] + "..."
            st = st.replace("\n", "\\n")
            kind = ""
            exp.append(
                f"  {prefix}{dot_name(obj.name)} "
                f'[shape=box label="{dot_name(obj.name)}'
                f"\\n{kind}{val.dtype}({val.shape})"
                f'\\n{dot_label(st)}" fontsize={fontsize}];'
            )
            inter_vars[obj.name] = obj

    # nodes
    fill_names = {}
    if hasattr(proto, "graph"):
        static_inputs = [n.name for n in proto.graph.input]
        static_inputs.extend(n.name for n in proto.graph.initializer)
        static_inputs.extend(n.name for n in proto.graph.sparse_initializer)
        nodes = list(proto.graph.node)
    else:
        static_inputs = list(proto.input)
        nodes = proto.node
    for node in nodes:
        exp.append("")
        for out in node.output:
            if out and out not in inter_vars:
                inter_vars[out] = out
                sh = shapes.get(out, "")
                if sh:
                    sh = f"\\nshape={sh}"
                exp.append(
                    '  {2}{0} [shape=box label="{0}{3}" fontsize={1}];'.format(
                        dot_name(out), fontsize, dot_name(prefix), dot_label(sh)
                    )
                )
            static_inputs.append(out)

        if node.name.strip() == "" or node.name in fill_names:
            name = node.op_type
            iname = 1
            while name in fill_names:
                name = "%s%d" % (name, iname)
                iname += 1
            node.name = name
            fill_names[name] = node

        atts = []
        node_attributes = attributes_as_dict(node)
        for k, v in sorted(node_attributes.items()):
            if isinstance(v, (GraphProto, Graph)):
                continue
            val = str(v).replace("\n", "\\n").replace('"', "'")
            sl = max(30 - len(k), 10)
            if len(val) > sl:
                val = val[:sl] + "..."
            if val is not None:
                atts.append(f"{k}={val}")
        satts = "" if len(atts) == 0 else ("\\n" + "\\n".join(atts))

        connects = []
        if recursive and node.op_type in {"Scan", "Loop", "If"}:
            fields = (
                ["then_branch", "else_branch"] if node.op_type == "If" else ["body"]
            )
            for field in fields:
                if field not in node_attributes:
                    continue

                # creates the subgraph
                body = node_attributes[field]
                subprefix = prefix + "B_"
                subdot = to_dot(
                    body, recursive=recursive, prefix=subprefix, rt_shapes=rt_shapes
                )
                lines = subdot.split("\n")
                start = 0
                for i, line in enumerate(lines):
                    if "[" in line:
                        start = i
                        break
                subgraph = "\n".join(lines[start:])

                # connecting the subgraph
                cluster = f"cluster_{node.op_type}{id(node)}_{id(field)}"
                exp.append(f"  subgraph {cluster} {{")
                exp.append(f'    label="{node.op_type}\\n({dot_name(field)}){satts}";')
                exp.append(f"    fontsize={fontsize};")
                exp.append("    color=black;")
                exp.append("\n".join(map(lambda s: "  " + s, subgraph.split("\n"))))

                node0 = body.node[0]
                connects.append(
                    (f"{dot_name(subprefix)}{dot_name(node0.name)}", cluster)
                )

                for inp1, inp2 in zip(node.input, body.input):
                    exp.append(
                        f"  {dot_name(prefix)}{dot_name(inp1)} -> "
                        f"{dot_name(subprefix)}{dot_name(inp2.name)};"
                    )
                for out1, out2 in zip(body.output, node.output):
                    if not out2:
                        # Empty output, it cannot be used.
                        continue
                    exp.append(
                        f"  {dot_name(subprefix)}{dot_name(out1.name)} -> "
                        f"{dot_name(prefix)}{dot_name(out2)};"
                    )
        else:
            exp.append(
                f"  {dot_name(prefix)}{dot_name(node.name)} "
                f'[shape=box style="filled,rounded" color=orange '
                f'label="{node.op_type}{satts}" '
                f"fontsize={fontsize}];"
            )

        if connects is not None and len(connects) > 0:
            for name, cluster in connects:
                exp.append(
                    f"  {dot_name(prefix)}{dot_name(node.name)} -> "
                    f"{name} [lhead={cluster}];"
                )

        for inp in node.input:
            exp.append(
                f"  {dot_name(prefix)}{dot_name(inp)} -> "
                f"{dot_name(prefix)}{dot_name(node.name)};"
            )
        for out in node.output:
            if not out:
                # Empty output, it cannot be used.
                continue
            exp.append(
                f"  {dot_name(prefix)}{dot_name(node.name)} -> "
                f"{dot_name(prefix)}{dot_name(out)};"
            )

    functions = getattr(proto, "function", [])
    if add_functions and len(functions) > 0:
        for f in functions:
            dot = to_dot(
                f,
                recursive=recursive,
                prefix=prefix + f.name,
                use_onnx=use_onnx,
                add_functions=False,
                rt_shapes=rt_shapes,
                **params,
            )
            spl = dot.split("\n")[1:]
            exp.append("")
            exp.append("  subgraph cluster_%d {" % i)
            exp.append(f'    label="{v.obj.name}";')
            exp.append("    color=blue;")
            # exp.append('    style=filled;')
            exp.extend(("  " + line) for line in spl)

    exp.append("}")
    return "\n".join(exp)
