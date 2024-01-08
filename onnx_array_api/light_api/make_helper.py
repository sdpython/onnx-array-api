from typing import Any, Optional, Sequence
from onnx import AttributeProto, NodeProto
from onnx.helper import make_attribute


def make_ref_attribute(
    key: str, attr_type: int, ref_attr_name: Optional[str] = None
) -> AttributeProto:
    """
    Creates an attribute.

    :param key: atttribute name
    :param attr_type: attribute type
    :param ref_attr_name: if not None, link this attribute
        to a function attribute
    :return: attribute
    """
    att = AttributeProto()
    att.name = key
    att.type = attr_type
    att.ref_attr_name = ref_attr_name
    return att


def make_node_extended(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    domain: Optional[str] = None,
    **kwargs: Any,
) -> NodeProto:
    """
    Constructs a NodeProto.

    :param op_type: The name of the operator to construct
    :param inputs: list of input names
    :param outputs: list of output names
    :param name: optional unique identifier for NodeProto
    :param doc_string: optional documentation string for NodeProto
    :param domain: optional domain for NodeProto.
        If it's None, we will just use default domain (which is empty)
    :param kwargs: the attributes of the node.
    :return: node proto
    """
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if kwargs:
        for key, value in sorted(kwargs.items()):
            if value is None:
                continue
            if isinstance(value, AttributeProto):
                node.attribute.append(value)
            else:
                node.attribute.append(make_attribute(key, value))
    return node
