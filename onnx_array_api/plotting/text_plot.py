import pprint
from collections import OrderedDict
import numpy
from onnx import AttributeProto
from ..reference import to_array_extended as to_array
from ._helper import _get_shape, _get_type, attributes_as_dict


def _rule(r):
    if r == "BRANCH_LEQ":
        return "<="
    if r == "BRANCH_LT":
        return "<"
    if r == "BRANCH_GEQ":
        return ">="
    if r == "BRANCH_GT":
        return ">"
    if r == "BRANCH_EQ":
        return "=="
    if r == "BRANCH_NEQ":
        return "!="
    raise ValueError(f"Unexpected rule {r!r}.")


def _number2str(i):
    if isinstance(i, int):
        return str(i)
    if int(i) == i:
        return str(int(i))
    return f"{i:1.2f}"


def onnx_text_plot_tree(node):
    """
    Gives a textual representation of a tree ensemble.

    :param node: `TreeEnsemble*`
    :return: text

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning, FutureWarning

        import numpy
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeRegressor
        from skl2onnx import to_onnx
        from onnx_array_api.plotting.text_plot import onnx_text_plot_tree

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeRegressor(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        res = onnx_text_plot_tree(onx.graph.node[0])
        print(res)
    """

    class Node:
        "Node representation."

        def __init__(self, i, atts):
            self.nodes_hitrates = None
            self.nodes_missing_value_tracks_true = None
            for k, v in atts.items():
                if k.startswith("nodes"):
                    setattr(self, k, v[i])
            self.depth = 0
            self.true_false = ""
            self.targets = []

        def append_target(self, tid, weight):
            self.targets.append(dict(target_id=tid, weight=weight))

        def process_node(self):
            "node to string"
            if self.nodes_modes == "LEAF":
                if not self.targets:
                    text = f"{self.true_false}f"
                elif len(self.targets) == 1:
                    t = self.targets[0]
                    text = (
                        f"{self.true_false}f "
                        f"{t['target_id']}:{_number2str(t['weight'])}"
                    )
                else:
                    ts = " ".join(
                        f"{t['target_id']}:{_number2str(t['weight'])}"
                        for t in self.targets
                    )
                    text = f"{self.true_false}f {ts}"
            else:
                text = "%sn X%d %s %r" % (
                    self.true_false,
                    self.nodes_featureids,
                    _rule(self.nodes_modes),
                    self.nodes_values,
                )
                if self.nodes_hitrates and self.nodes_hitrates != 1:
                    text += f" hi={self.nodes_hitrates!r}"
                if self.nodes_missing_value_tracks_true:
                    text += f" miss={self.nodes_missing_value_tracks_true!r}"
            return f"{'   ' * self.depth}{text}"

    def process_tree(atts, treeid):
        "tree to string"
        rows = [f"treeid={treeid!r}"]
        if "base_values" in atts:
            if treeid < len(atts["base_values"]):
                rows.append(f"base_value={atts['base_values'][treeid]!r}")

        short = {}
        for prefix in ["nodes", "target", "class"]:
            if (f"{prefix}_treeids") not in atts:
                continue
            idx = [
                i
                for i in range(len(atts[f"{prefix}_treeids"]))
                if atts[f"{prefix}_treeids"][i] == treeid
            ]
            for k, v in atts.items():
                if k.startswith(prefix):
                    if "classlabels" in k:
                        short[k] = list(v)
                    else:
                        short[k] = [v[i] for i in idx]

        nodes = OrderedDict()
        for i in range(len(short["nodes_treeids"])):
            nodes[i] = Node(i, short)
        prefix = "target" if "target_treeids" in short else "class"
        for i in range(len(short[f"{prefix}_treeids"])):
            idn = short[f"{prefix}_nodeids"][i]
            node = nodes[idn]
            node.append_target(
                tid=short[f"{prefix}_ids"][i], weight=short[f"{prefix}_weights"][i]
            )

        def iterate(nodes, node, depth=0, true_false=""):
            node.depth = depth
            node.true_false = true_false
            yield node
            if node.nodes_falsenodeids > 0:
                for n in iterate(
                    nodes,
                    nodes[node.nodes_falsenodeids],
                    depth=depth + 1,
                    true_false="-",
                ):
                    yield n
                for n in iterate(
                    nodes,
                    nodes[node.nodes_truenodeids],
                    depth=depth + 1,
                    true_false="+",
                ):
                    yield n

        for node in iterate(nodes, nodes[0]):
            rows.append(node.process_node())
        return rows

    if node.op_type in ("TreeEnsembleRegressor", "TreeEnsembleClassifier"):
        d = attributes_as_dict(node)
        atts = {}
        for k, v in d.items():
            atts[k] = v if isinstance(v, int) else list(v)
        trees = list(sorted(set(atts["nodes_treeids"])))
        if "n_targets" in atts:
            rows = [f"n_targets={atts['n_targets']!r}"]
        else:
            rows = [
                "n_classes=%r"
                % len(
                    atts.get("classlabels_int64s", atts.get("classlabels_strings", []))
                )
            ]
        rows.append(f"n_trees={len(trees)!r}")
        for tree in trees:
            r = process_tree(atts, tree)
            rows.append("----")
            rows.extend(r)
        return "\n".join(rows)

    raise NotImplementedError(f"Type {node.op_type!r} cannot be displayed.")


def _append_succ_pred(
    subgraphs,
    successors,
    predecessors,
    node_map,
    node,
    prefix="",
    parent_node_name=None,
):
    node_name = prefix + node.name + "#" + "|".join(node.output)
    node_map[node_name] = node
    successors[node_name] = []
    predecessors[node_name] = []
    for name in node.input:
        predecessors[node_name].append(name)
        if name not in successors:
            successors[name] = []
        successors[name].append(node_name)
    for name in node.output:
        successors[node_name].append(name)
        predecessors[name] = [node_name]
    if node.op_type in {"If", "Scan", "Loop", "Expression"}:
        for att in node.attribute:
            if (
                att.type != AttributeProto.GRAPH
                or not hasattr(att, "g")
                or att.g is None
            ):
                continue
            subgraphs.append((node, att.name, att.g))
            _append_succ_pred_s(
                subgraphs,
                successors,
                predecessors,
                node_map,
                att.g.node,
                prefix=node_name + ":/:",
                parent_node_name=node_name,
                parent_graph=att.g,
            )


def _append_succ_pred_s(
    subgraphs,
    successors,
    predecessors,
    node_map,
    nodes,
    prefix="",
    parent_node_name=None,
    parent_graph=None,
):
    for node in nodes:
        _append_succ_pred(
            subgraphs,
            successors,
            predecessors,
            node_map,
            node,
            prefix=prefix,
            parent_node_name=parent_node_name,
        )
    if parent_node_name is not None:
        unknown = set()
        known = {}
        for i in parent_graph.initializer:
            known[i.name] = None
        for i in parent_graph.input:
            known[i.name] = None
        for n in parent_graph.node:
            for i in n.input:
                if i not in known:
                    unknown.add(i)
            for i in n.output:
                known[i] = n
        if unknown:
            # These inputs are coming from the graph below.
            for name in unknown:
                successors[name].append(parent_node_name)
                predecessors[parent_node_name].append(name)


def graph_predecessors_and_successors(graph):
    """
    Returns the successors and the predecessors within on ONNX graph.
    """
    node_map = {}
    successors = {}
    predecessors = {}
    subgraphs = []
    _append_succ_pred_s(subgraphs, successors, predecessors, node_map, graph.node)
    return subgraphs, predecessors, successors, node_map


def get_hidden_inputs(nodes):
    """
    Returns the list of hidden inputs used by subgraphs.

    :param nodes: list of nodes
    :return: list of names
    """
    inputs = set()
    outputs = set()
    for node in nodes:
        inputs |= set(node.input)
        outputs |= set(node.output)
        for att in node.attribute:
            if (
                att.type != AttributeProto.GRAPH
                or not hasattr(att, "g")
                or att.g is None
            ):
                continue
            hidden = get_hidden_inputs(att.g.node)
            inits = set(i.name for i in att.g.initializer)
            inits |= set(i.name for i in att.g.sparse_initializer)
            inputs |= hidden - (inits & hidden)
    return inputs - (outputs & inputs)


def reorder_nodes_for_display(nodes, verbose=False):
    """
    Reorders the node with breadth first seach (BFS).

    :param nodes: list of ONNX nodes
    :param verbose: dislay intermediate informations
    :return: reordered list of nodes
    """

    class temp:
        "Fake GraphProto."

        def __init__(self, nodes):
            self.node = nodes

    _, predecessors, successors, dnodes = graph_predecessors_and_successors(temp(nodes))
    local_variables = get_hidden_inputs(nodes)

    all_outputs = set()
    all_inputs = set(local_variables)
    for node in nodes:
        all_outputs |= set(node.output)
        all_inputs |= set(node.input)
    common = all_outputs & all_inputs

    successors = {k: set(v) for k, v in successors.items()}
    predecessors = {k: set(v) for k, v in predecessors.items()}
    if verbose:
        pprint.pprint(
            [
                "[reorder_nodes_for_display]",
                "predecessors",
                predecessors,
                "successors",
                successors,
            ]
        )

    known = all_inputs - common
    new_nodes = []
    done = set()

    def _find_sequence(node_name, known, done):
        inputs = dnodes[node_name].input
        if any((i not in known) for i in inputs):
            return []

        res = [node_name]
        while res[-1] in successors:
            next_names = successors[res[-1]]
            if res[-1] not in dnodes:
                next_names = set(v for v in next_names if v not in known)
                if len(next_names) == 1:
                    next_name = next_names.pop()
                    inputs = dnodes[next_name].input
                    if any((i not in known) for i in inputs):
                        break
                    res.extend(next_name)
                else:
                    break
            else:
                next_names = set(v for v in next_names if v not in done)
                if len(next_names) == 1:
                    next_name = next_names.pop()
                    res.append(next_name)
                else:
                    break

        return [r for r in res if r in dnodes and r not in done]

    while len(done) < len(nodes):
        # possible
        possibles = OrderedDict()
        for k, v in dnodes.items():
            if k in done:
                continue
            if ":/:" in k:
                # node part of a sub graph (assuming :/: is never used in a node name)
                continue
            if predecessors[k] <= known:
                possibles[k] = v

        sequences = OrderedDict()
        for k, _v in possibles.items():
            if k in done:
                continue
            sequences[k] = _find_sequence(k, known, done)
            if verbose:
                print(
                    "[reorder_nodes_for_display] * sequence(%s)=%s - %r"
                    % (k, ",".join(sequences[k]), list(sequences))
                )

        if not sequences:
            raise RuntimeError(
                "Unexpected empty sequence (len(possibles)=%d, "
                "len(done)=%d, len(nodes)=%d). This is usually due to "
                "a name used both as result name and node node. "
                "known=%r." % (len(possibles), len(done), len(nodes), known)
            )

        # find the best sequence
        best = None
        for k, v in sequences.items():
            if best is None or len(v) > len(sequences[best]):
                # if the sequence of successors is longer
                best = k
            elif len(v) == len(sequences[best]):
                if new_nodes:
                    # then choose the next successor sharing input with
                    # previous output
                    so = set(new_nodes[-1].output)
                    first1 = dnodes[sequences[best][0]]
                    first2 = dnodes[v[0]]
                    if len(set(first1.input) & so) < len(set(first2.input) & so):
                        best = k
                else:
                    first1 = dnodes[sequences[best][0]]
                    first2 = dnodes[v[0]]
                    if first1.op_type > first2.op_type:
                        best = k
                    elif first1.op_type == first2.op_type and first1.name > first2.name:
                        best = k

        if best is None:
            raise RuntimeError(
                f"Wrong implementation (len(sequence)={len(sequences)})."
            )
        if verbose:
            print(
                "[reorder_nodes_for_display] BEST: sequence(%s)=%s"
                % (best, ",".join(sequences[best]))
            )

        # process the sequence
        for k in sequences[best]:
            v = dnodes[k]
            new_nodes.append(v)
            if verbose:
                print(f"[reorder_nodes_for_display] + {v.name!r} ({v.op_type!r})")
            done.add(k)
            known |= set(v.output)

    if len(new_nodes) != len(nodes):
        raise RuntimeError(
            "The returned new nodes are different. "
            "len(nodes=%d) != %d=len(new_nodes). done=\n%r"
            "\n%s\n----------\n%s"
            % (
                len(nodes),
                len(new_nodes),
                done,
                "\n".join(
                    "%d - %s - %s - %s"
                    % (
                        (n.name + "".join(n.output)) in done,
                        n.op_type,
                        n.name,
                        n.name + "".join(n.output),
                    )
                    for n in nodes
                ),
                "\n".join(
                    "%d - %s - %s - %s"
                    % (
                        (n.name + "".join(n.output)) in done,
                        n.op_type,
                        n.name,
                        n.name + "".join(n.output),
                    )
                    for n in new_nodes
                ),
            )
        )
    n0s = set(n.name for n in nodes)
    n1s = set(n.name for n in new_nodes)
    if n0s != n1s:
        raise RuntimeError(
            "The returned new nodes are different.\n"
            "%r !=\n%r\ndone=\n%r"
            "\n----------\n%s\n----------\n%s"
            % (
                n0s,
                n1s,
                done,
                "\n".join(
                    "%d - %s - %s - %s"
                    % (
                        (n.name + "".join(n.output)) in done,
                        n.op_type,
                        n.name,
                        n.name + "".join(n.output),
                    )
                    for n in nodes
                ),
                "\n".join(
                    "%d - %s - %s - %s"
                    % (
                        (n.name + "".join(n.output)) in done,
                        n.op_type,
                        n.name,
                        n.name + "".join(n.output),
                    )
                    for n in new_nodes
                ),
            )
        )
    return new_nodes


def onnx_simple_text_plot(
    model,
    verbose=False,
    att_display=None,
    add_links=False,
    recursive=False,
    functions=True,
    raise_exc=True,
    sub_graphs_names=None,
    level=1,
    indent=True,
):
    """
    Displays an ONNX graph into text.

    :param model: ONNX graph
    :param verbose: display debugging information
    :param att_display: list of attributes to display, if None,
        a default list if used
    :param add_links: displays links of the right side
    :param recursive: display subgraphs as well
    :param functions: display functions as well
    :param raise_exc: raises an exception if the model is not valid,
        otherwise tries to continue
    :param sub_graphs_names: list of sub-graphs names
    :param level: sub-graph level
    :param indent: use indentation or not
    :return: str

    An ONNX graph is printed the following way:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning, FutureWarning

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False)
        print(text)

    The same graphs with links.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning, FutureWarning

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False, add_links=True)
        print(text)

    Visually, it looks like the following:

    .. gdot::
        :script: DOT-SECTION

        # onnx_simple_text_plot
        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from onnx_array_api.plotting.dot_plot import to_dot

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        model_onnx = to_onnx(model, x.astype(numpy.float32),
                             target_opset=15)
        print("DOT-SECTION", to_dot(model_onnx))
    """
    use_indentation = indent
    if att_display is None:
        att_display = [
            "activations",
            "align_corners",
            "allowzero",
            "alpha",
            "auto_pad",
            "axis",
            "axes",
            "batch_axis",
            "batch_dims",
            "beta",
            "bias",
            "blocksize",
            "case_change_action",
            "ceil_mode",
            "center_point_box",
            "clip",
            "coordinate_transformation_mode",
            "count_include_pad",
            "cubic_coeff_a",
            "decay_factor",
            "detect_negative",
            "detect_positive",
            "dilation",
            "dilations",
            "direction",
            "dtype",
            "end",
            "epsilon",
            "equation",
            "exclusive",
            "exclude_outside",
            "extrapolation_value",
            "fmod",
            "gamma",
            "group",
            "hidden_size",
            "high",
            "ignore_index",
            "input_forget",
            "is_case_sensitive",
            "k",
            "keepdims",
            "kernel_shape",
            "lambd",
            "largest",
            "layout",
            "linear_before_reset",
            "locale",
            "low",
            "max_gram_length",
            "max_skip_count",
            "mean",
            "min_gram_length",
            "mode",
            "momentum",
            "nearest_mode",
            "ngram_counts",
            "ngram_indexes",
            "noop_with_empty_axes",
            "norm_coefficient",
            "norm_coefficient_post",
            "num_scan_inputs",
            "output_height",
            "output_padding",
            "output_shape",
            "output_width",
            "p",
            "padding_mode",
            "pads",
            "perm",
            "pooled_shape",
            "reduction",
            "reverse",
            "sample_size",
            "sampling_ratio",
            "scale",
            "scan_input_axes",
            "scan_input_directions",
            "scan_output_axes",
            "scan_output_directions",
            "seed",
            "select_last_index",
            "size",
            "sorted",
            "spatial_scale",
            "start",
            "storage_order",
            "strides",
            "time_axis",
            "to",
            "training_mode",
            "transA",
            "transB",
            "type",
            "upper",
            "xs",
            "y",
            "zs",
        ]

    if sub_graphs_names is None:
        sub_graphs_names = {}

    def _get_subgraph_name(idg):
        if idg in sub_graphs_names:
            return sub_graphs_names[idg]
        g = "G%d" % (len(sub_graphs_names) + 1)
        sub_graphs_names[idg] = g
        return g

    def str_node(indent, node):
        atts = []
        if hasattr(node, "attribute"):
            for att in node.attribute:
                done = True
                if hasattr(att, "ref_attr_name") and att.ref_attr_name:
                    atts.append(f"{att.name}=${att.ref_attr_name}")
                    continue
                if att.name in att_display:
                    if att.type == AttributeProto.INT:
                        atts.append("%s=%d" % (att.name, att.i))
                    elif att.type == AttributeProto.FLOAT:
                        atts.append(f"{att.name}={att.f:1.2f}")
                    elif att.type == AttributeProto.INTS:
                        atts.append(
                            "%s=%s" % (att.name, str(list(att.ints)).replace(" ", ""))
                        )
                    else:
                        done = False
                elif (
                    att.type == AttributeProto.GRAPH
                    and hasattr(att, "g")
                    and att.g is not None
                ):
                    atts.append(f"{att.name}={_get_subgraph_name(id(att.g))}")
                else:
                    done = False
                if done:
                    continue
                if att.type in (
                    AttributeProto.TENSOR,
                    AttributeProto.TENSORS,
                    AttributeProto.SPARSE_TENSOR,
                    AttributeProto.SPARSE_TENSORS,
                ):
                    try:
                        val = str(to_array(att.t).tolist())
                    except TypeError as e:
                        raise TypeError(
                            "Unable to display tensor type %r.\n%s"
                            % (att.type, str(att))
                        ) from e
                    if "\n" in val:
                        val = val.split("\n", maxsplit=1) + "..."
                    if len(val) > 10:
                        val = val[:10] + "..."
                elif att.type == AttributeProto.STRING:
                    val = str(att.s)
                    if len(val) > 50:
                        val = val[:40] + "..." + val[-10:]
                elif att.type == AttributeProto.STRINGS:
                    n_val = list(att.strings)
                    if len(n_val) < 5:
                        val = ",".join(map(str, n_val))
                    else:
                        val = "%d:[%s...%s]" % (
                            len(n_val),
                            ",".join(map(str, n_val[:2])),
                            ",".join(map(str, n_val[-2:])),
                        )
                elif att.type == AttributeProto.INT:
                    val = str(att.i)
                elif att.type == AttributeProto.FLOAT:
                    val = str(att.f)
                elif att.type == AttributeProto.INTS:
                    n_val = list(att.ints)
                    if len(n_val) < 6:
                        val = f"[{','.join(map(str, n_val))}]"
                    else:
                        val = "%d:[%s...%s]" % (
                            len(n_val),
                            ",".join(map(str, n_val[:3])),
                            ",".join(map(str, n_val[-3:])),
                        )
                elif att.type == AttributeProto.FLOATS:
                    n_val = list(att.floats)
                    if len(n_val) < 5:
                        val = f"[{','.join(map(str, n_val))}]"
                    else:
                        val = "%d:[%s...%s]" % (
                            len(n_val),
                            ",".join(map(str, n_val[:2])),
                            ",".join(map(str, n_val[-2:])),
                        )
                else:
                    val = ".%d" % att.type
                atts.append(f"{att.name}={val}")
        inputs = list(node.input)
        if atts:
            inputs.extend(atts)
        if node.domain in ("", "ai.onnx.ml"):
            domain = ""
        else:
            domain = f"[{node.domain}]"
        return "%s%s%s(%s) -> %s" % (
            "  " * indent,
            node.op_type,
            domain,
            ", ".join(inputs),
            ", ".join(node.output),
        )

    rows = []
    if hasattr(model, "opset_import"):
        for opset in model.opset_import:
            rows.append(f"opset: domain={opset.domain!r} version={opset.version!r}")
    if hasattr(model, "graph"):
        if model.doc_string:
            if len(model.doc_string) < 55:
                rows.append(f"doc_string: {model.doc_string}")
            else:
                rows.append(f"doc_string: {model.doc_string[:55]}...")
        main_model = model
        model = model.graph
    else:
        main_model = None

    # inputs
    line_name_new = {}
    line_name_in = {}
    if level == 0:
        rows.append("----- input ----")
    for inp in model.input:
        if isinstance(inp, str):
            rows.append(f"input: {inp!r}")
        else:
            line_name_new[inp.name] = len(rows)
            rows.append(
                "input: name=%r type=%r shape=%r"
                % (inp.name, _get_type(inp), _get_shape(inp))
            )
    if hasattr(model, "attribute"):
        for att in model.attribute:
            if isinstance(att, str):
                rows.append(f"attribute: {att!r}")
            else:
                raise NotImplementedError("Not yet introduced in onnx.")

    # initializer
    if hasattr(model, "initializer"):
        if len(model.initializer) and level == 0:
            rows.append("----- initializer ----")
        for init in model.initializer:
            if numpy.prod(_get_shape(init)) < 5:
                content = f" -- {to_array(init).ravel()!r}"
            else:
                content = ""
            line_name_new[init.name] = len(rows)
            if init.doc_string:
                t = (
                    f"init: name={init.name!r} type={_get_type(init)} "
                    f"shape={_get_shape(init)}{content}"
                )
                rows.append(f"{t}{' ' * max(0, 70 - len(t))}-- {init.doc_string}")
                continue
            rows.append(
                f"init: name={init.name!r} type={_get_type(init)} "
                f"shape={_get_shape(init)}{content}"
            )
    if level == 0:
        rows.append("----- main graph ----")

    # successors, predecessors, it needs to support subgraphs
    subgraphs = graph_predecessors_and_successors(model)[0]

    # walk through nodes
    init_names = set()
    indents = {}
    for inp in model.input:
        if isinstance(inp, str):
            indents[inp] = 0
            init_names.add(inp)
        else:
            indents[inp.name] = 0
            init_names.add(inp.name)
    if hasattr(model, "initializer"):
        for init in model.initializer:
            indents[init.name] = 0
            init_names.add(init.name)

    try:
        nodes = reorder_nodes_for_display(model.node, verbose=verbose)
    except RuntimeError as e:
        if raise_exc:
            raise e
        else:
            rows.append(f"ERROR: {e}")
        nodes = model.node

    previous_indent = None
    previous_out = None
    previous_in = None
    for node in nodes:
        add_break = False
        name = node.name + "#" + "|".join(node.output)
        if name in indents:
            indent = indents[name]
            if previous_indent is not None and indent < previous_indent:
                if verbose:
                    print(f"[onnx_simple_text_plot] break1 {node.op_type}")
                add_break = True
        elif previous_in is not None and set(node.input) == previous_in:
            indent = previous_indent
        else:
            inds = [indents.get(i, 0) for i in node.input if i not in init_names]
            if not inds:
                indent = 0
            else:
                mi = min(inds)
                indent = mi
                if previous_indent is not None and indent < previous_indent:
                    if verbose:
                        print(f"[onnx_simple_text_plot] break2 {node.op_type}")
                    add_break = True
            if not add_break and previous_out is not None:
                if not (set(node.input) & previous_out):
                    if verbose:
                        print(f"[onnx_simple_text_plot] break3 {node.op_type}")
                    add_break = True
                    indent = 0

        if add_break and verbose:
            print("[onnx_simple_text_plot] add break")
        for n in node.input:
            if n in line_name_in:
                line_name_in[n].append(len(rows))
            else:
                line_name_in[n] = [len(rows)]
        for n in node.output:
            line_name_new[n] = len(rows)
        rows.append(str_node(indent if use_indentation else 0, node))
        indents[name] = indent

        for _i, o in enumerate(node.output):
            indents[o] = indent + 1

        previous_indent = indents[name]
        previous_out = set(node.output)
        previous_in = set(node.input)

    # outputs
    if level == 0:
        rows.append("----- output ----")
    for out in model.output:
        if isinstance(out, str):
            if out in line_name_in:
                line_name_in[out].append(len(rows))
            else:
                line_name_in[out] = [len(rows)]
            rows.append(f"output: name={out!r} type={'?'} shape={'?'}")
        else:
            if out.name in line_name_in:
                line_name_in[out.name].append(len(rows))
            else:
                line_name_in[out.name] = [len(rows)]
            rows.append(
                "output: name=%r type=%r shape=%r"
                % (out.name, _get_type(out), _get_shape(out))
            )

    if add_links:

        def _mark_link(rows, lengths, r1, r2, d):
            maxl = max(lengths[r1], lengths[r2]) + d * 2
            maxl = max(maxl, max(len(rows[r]) for r in range(r1, r2 + 1))) + 2

            if rows[r1][-1] == "|":
                p1, p2 = rows[r1][: lengths[r1] + 2], rows[r1][lengths[r1] + 2 :]
                rows[r1] = p1 + p2.replace(" ", "-")
            rows[r1] += ("-" * (maxl - len(rows[r1]) - 1)) + "+"

            if rows[r2][-1] == " ":
                rows[r2] += "<"
            elif rows[r2][-1] == "|":
                if "<" not in rows[r2]:
                    p = lengths[r2]
                    rows[r2] = rows[r2][:p] + "<" + rows[r2][p + 1 :]
                p1, p2 = rows[r2][: lengths[r2] + 2], rows[r2][lengths[r2] + 2 :]
                rows[r2] = p1 + p2.replace(" ", "-")
            rows[r2] += ("-" * (maxl - len(rows[r2]) - 1)) + "+"

            for r in range(r1 + 1, r2):
                if len(rows[r]) < maxl:
                    rows[r] += " " * (maxl - len(rows[r]) - 1)
                rows[r] += "|"

        diffs = []
        for n, r1 in line_name_new.items():
            if n not in line_name_in:
                continue
            r2s = line_name_in[n]
            for r2 in r2s:
                if r1 >= r2:
                    continue
                diffs.append((r2 - r1, (n, r1, r2)))
        diffs.sort()
        for i in range(len(rows)):
            rows[i] += "  "
        lengths = [len(r) for r in rows]

        for d, (n, r1, r2) in diffs:
            if d == 1 and len(line_name_in[n]) == 1:
                # no line for link to the next node
                continue
            _mark_link(rows, lengths, r1, r2, d)

    # subgraphs
    if recursive:
        for node, name, g in subgraphs:
            rows.append(
                "----- subgraph ---- %s - %s - att.%s=%s -- level=%d -- %s -> %s"
                % (
                    node.op_type,
                    node.name,
                    name,
                    _get_subgraph_name(id(g)),
                    level,
                    ",".join(i.name for i in g.input),
                    ",".join(i.name for i in g.output),
                )
            )
            res = onnx_simple_text_plot(
                g,
                verbose=verbose,
                att_display=att_display,
                add_links=add_links,
                recursive=recursive,
                sub_graphs_names=sub_graphs_names,
                level=level + 1,
                raise_exc=raise_exc,
            )
            rows.append(res)

    # functions
    if functions and main_model is not None:
        for fct in main_model.functions:
            rows.append(f"----- function name={fct.name} domain={fct.domain}")
            if fct.doc_string:
                if len(fct.doc_string) < 55:
                    rows.append(f"----- doc_string: {fct.doc_string}")
                else:
                    rows.append(f"----- doc_string: {fct.doc_string[:55]}...")
            res = onnx_simple_text_plot(
                fct,
                verbose=verbose,
                att_display=att_display,
                add_links=add_links,
                recursive=recursive,
                functions=False,
                sub_graphs_names=sub_graphs_names,
                level=1,
            )
            rows.append(res)

    return "\n".join(rows)


def onnx_text_plot_io(model, verbose=False, att_display=None):
    """
    Displays information about input and output types.

    :param model: ONNX graph
    :param verbose: display debugging information
    :return: str

    An ONNX graph is printed the following way:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning, FutureWarning

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from onnx_array_api.plotting.text_plot import onnx_text_plot_io

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_text_plot_io(onx, verbose=False)
        print(text)
    """
    rows = []
    if hasattr(model, "opset_import"):
        for opset in model.opset_import:
            rows.append(f"opset: domain={opset.domain!r} version={opset.version!r}")
    if hasattr(model, "graph"):
        model = model.graph

    # inputs
    for inp in model.input:
        rows.append(
            "input: name=%r type=%r shape=%r"
            % (inp.name, _get_type(inp), _get_shape(inp))
        )
    # initializer
    for init in model.initializer:

        if init.doc_string:
            t = (
                f"init: name={init.name!r} type={_get_type(init)} "
                f"shape={_get_shape(init)}"
            )
            rows.append(f"{t}{' ' * max(0, 70 - len(t))}-- {init.doc_string}")
            continue
        rows.append(
            f"init: name={init.name!r} type={_get_type(init)} "
            f"shape={_get_shape(init)}"
        )

    # outputs
    for out in model.output:
        rows.append(
            "output: name=%r type=%r shape=%r"
            % (out.name, _get_type(out), _get_shape(out))
        )
    return "\n".join(rows)
