from typing import Union
import numpy
from onnx import (
    AttributeProto,
    GraphProto,
    FunctionProto,
    ModelProto,
    NodeProto,
    TensorProto,
)
from onnx.helper import (
    make_attribute,
    make_function,
    make_graph,
    make_model,
    make_node,
    set_model_props,
)
from ..reference import from_array_extended as from_array, to_array_extended as to_array


def randomize_proto(
    onx: Union[ModelProto, GraphProto, FunctionProto, NodeProto, TensorProto]
) -> Union[ModelProto, GraphProto, FunctionProto, NodeProto, TensorProto]:
    """
    Randomizes float initializers or constant nodes.

    :param onx: onnx model or proto
    :return: same object
    """
    if isinstance(onx, TensorProto):
        t = to_array(onx)
        mini, maxi = t.min(), t.max()
        new_t = numpy.clip(
            numpy.random.random(t.shape) * (maxi - mini) + mini, mini, maxi
        )
        return from_array(new_t.astype(t.dtype), name=onx.name)

    if isinstance(onx, ModelProto):
        new_graph = randomize_proto(onx.graph)
        new_functions = [randomize_proto(f) for f in onx.functions]

        onnx_model = make_model(
            new_graph,
            functions=new_functions,
            ir_version=onx.ir_version,
            producer_name=onx.producer_name,
            domain=onx.domain,
            doc_string=onx.doc_string,
            opset_imports=list(onx.opset_import),
        )
        if onx.metadata_props:
            values = {p.key: p.value for p in onx.metadata_props}
            set_model_props(onnx_model, values)
        return onnx_model

    if isinstance(onx, (GraphProto, FunctionProto)):
        nodes = []
        for node in onx.node:
            if node.op_type in "Constant":
                nodes.append(randomize_proto(node))
                continue
            changed = False
            atts = []
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    new_g = randomize_proto(att.g)
                    att = make_attribute(att.name, new_g)
                    changed = True
                atts.append(att)
            if changed:
                new_node = make_node(
                    node.op_type, node.input, node.output, domain=node.domain
                )
                new_node.attribute.extend(node.attribute)
                nodes.append(new_node)
                continue
            nodes.append(node)

        if isinstance(onx, FunctionProto):
            new_onx = make_function(
                onx.domain,
                onx.name,
                onx.input,
                onx.output,
                nodes,
                opset_imports=onx.opset_import,
            )
            return new_onx

        inits = [randomize_proto(init) for init in onx.initializer]
        sp_inits = [randomize_proto(init) for init in onx.sparse_initializer]

        graph = make_graph(
            nodes,
            onx.name,
            onx.input,
            onx.output,
            initializer=inits,
            sparse_initializer=sp_inits,
        )
        return graph

    raise TypeError(f"Unexpected type for onx {type(onx)}.")
