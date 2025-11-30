import numpy as np
from onnx import FunctionProto, ModelProto, GraphProto, AttributeProto
from onnx.helper import (
    make_model,
    set_model_props,
    make_graph,
    make_node,
    make_attribute,
    make_function,
    tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import from_array


def replace_initializer_by_constant_of_shape(
    onx, threshold=128, op_type="ConstantOfShape", domain=""
):
    """
    Replaces initializers by nodes *ConstantOfShape* to reduce
    the size and still write a unit test.

    :param onx: ModelProto
    :param threshold: every initializer under this threshold is not impacted
    :param op_type: replace by this node
    :param domain: replace by this domain
    :return: onx, modified ModelProto
    """
    if isinstance(onx, FunctionProto):
        modified = False
        new_nodes = []
        for node in onx.node:
            if node.op_type == "Constant":
                from onnx_array_api.reference import ExtendedReferenceEvaluator

                ref = ExtendedReferenceEvaluator(node)
                cst = ref.run(None, {})[0]

                size = np.prod(cst.shape)
                if size <= threshold:
                    new_nodes.append(node)
                    continue

                new_name = f"{node.output[0]}__SHAPE"
                new_nodes.append(
                    make_node(
                        "Constant",
                        [],
                        [new_name],
                        value=from_array(
                            np.array(cst.shape, dtype=np.int64), name=new_name
                        ),
                    )
                )
                dtype = cst.dtype
                new_nodes.append(
                    make_node(
                        op_type,
                        [new_name],
                        node.output,
                        value=from_array(np.array([0.5], dtype=dtype)),
                        domain=domain,
                    )
                )
                modified = True
                continue

            new_nodes.append(node)

        if not modified:
            return onx

        onxf = make_function(
            domain=onx.domain,
            fname=onx.name,
            inputs=onx.input,
            outputs=onx.output,
            nodes=new_nodes,
            doc_string=onx.doc_string,
            overload=onx.overload,
            opset_imports=[],
        )
        if onx.opset_import:
            onxf.opset_import.extend(onx.opset_import)
        if onx.value_info:
            onxf.value_info.extend(onx.value_info)
        if onx.attribute:
            onxf.attribute.extend(onx.attribute)
        if onx.attribute_proto:
            onxf.attribute_proto.extend(onx.attribute_proto)
        return onxf

    if isinstance(onx, ModelProto):
        new_graph = replace_initializer_by_constant_of_shape(
            onx.graph, threshold=threshold, op_type=op_type, domain=domain
        )
        new_functions = [
            replace_initializer_by_constant_of_shape(
                f, threshold=threshold, op_type=op_type, domain=domain
            )
            for f in onx.functions
        ]
        model = make_model(
            new_graph,
            functions=new_functions,
            producer_name=onx.producer_name,
            producer_version=onx.producer_version,
            ir_version=onx.ir_version,
            doc_string=onx.doc_string,
            domain=onx.domain,
            model_version=onx.model_version,
        )
        if len(onx.metadata_props) > 0:  # pragma: no cover
            values = {p.key: p.value for p in onx.metadata_props}
            set_model_props(model, values)

        del model.opset_import[:]  # pylint: disable=E1101
        for oimp in onx.opset_import:
            op_set = model.opset_import.add()  # pylint: disable=E1101
            if oimp.domain == "" and oimp.version < 9:
                raise RuntimeError(
                    f"ConstantOfShape was introduced in "
                    f"opset 9 but opset is {oimp.version}."
                )
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return model

    if not isinstance(onx, GraphProto):
        raise TypeError(f"onx should be a GraphProto as this stage not {type(onx)}.")

    new_nodes = []
    removed = set()
    additional_inputs = []

    new_inits = []
    for init in onx.initializer:
        dims = tuple(init.dims)
        size = np.prod(dims)
        if size <= threshold:
            new_inits.append(init)
            continue
        new_name = f"{init.name}__SHAPE"
        new_inits.append(
            from_array(np.array(list(dims), dtype=np.int64), name=new_name)
        )
        dtype = tensor_dtype_to_np_dtype(init.data_type)
        node = make_node(
            op_type,
            [new_name],
            [init.name],
            value=from_array(np.array([0.5], dtype=dtype)),
            domain=domain,
        )
        new_nodes.append(node)
        removed.add(init.name)

    new_sparse_inits = []
    for init in onx.sparse_initializer:
        dims = tuple(init.dims)
        size = np.prod(dims)
        if size <= threshold:
            new_sparse_inits.append(init)
            continue
        raise NotImplementedError(
            f"This feature is not yet implemented for sparse initializer"
            f"(name={init.name!r})."
        )

    for node in onx.node:
        if node.op_type == "Constant":
            from onnx_array_api.reference import ExtendedReferenceEvaluator

            ref = ExtendedReferenceEvaluator(node)
            cst = ref.run(None, {})[0]

            size = np.prod(cst.shape)
            if size <= threshold:
                new_nodes.append(node)
                continue

            new_name = f"{node.output[0]}__SHAPE"
            new_inits.append(
                from_array(np.array(cst.shape, dtype=np.int64), name=new_name)
            )
            dtype = cst.dtype
            new_nodes.append(
                make_node(
                    op_type,
                    [new_name],
                    node.output,
                    value=from_array(np.array([0.5], dtype=dtype)),
                    domain=domain,
                )
            )
            continue

        modified = False
        atts = []
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                modified = True
                g = replace_initializer_by_constant_of_shape(
                    att.g, threshold=threshold, op_type=op_type, domain=domain
                )
                att = make_attribute(att.name, g)
            atts.append(att)
        if modified:
            new_node = make_node(node.op_type, node.input, node.output)
            new_node.attribute.extend(atts)
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    graph = make_graph(
        new_nodes,
        onx.name,
        [i for i in onx.input if i.name not in removed] + additional_inputs,
        onx.output,
        initializer=new_inits,
        sparse_initializer=new_sparse_inits,
    )
    return graph
