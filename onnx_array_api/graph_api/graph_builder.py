import sys
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from onnx.defs import onnx_opset_version
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
)
from onnx.reference import ReferenceEvaluator

T = "TENSOR"


class OptimizationOptions:
    def __init__(
        self,
        remove_unused: bool = True,
        constant_folding: bool = False,
        constant_size: int = 1024,
    ):
        self.remove_unused = remove_unused
        self.constant_folding = constant_folding
        self.constant_size = constant_size


class NodePattern:
    """
    Class defining a matching pattern able to find nodes in a set of nodes.
    """

    def __init__(
        self,
        index: Optional[int] = None,
        op_type: Optional[str] = None,
        name: Optional[None] = None,
    ):
        self.index = index
        self.op_type = op_type
        self.name = name

    def __repr__(self):
        "usual"
        args = ["index", "op_type", "name"]
        sargs = []
        for a in args:
            if a:
                sargs.append(f"{a}={getattr(self, a)!r}")
        return f"{self.__class__.__name__}({', '.join(sargs)})"

    def find(self, graph: "GraphBuilder") -> Iterator:
        """
        Iterates on nodes matching the pattern.
        """
        for index, node in enumerate(graph.nodes):
            if self.match(index, node):
                yield node

    def match(self, index, node: NodeProto) -> bool:
        """
        Tells if a node is matching this pattern.
        """
        if self.index is not None and self.index != index:
            return False
        if self.op_type is not None and self.op_type != node.op_type:
            return False
        if self.name is not None and self.name != node.name:
            return False
        return True


class Opset:
    # defined for opset >= 18
    # name: number of expected outputs
    _implemented = {
        "Add": 1,
        "And": 1,
        "Cast": 1,
        "Concat": 1,
        "Constant": 1,
        "Div": 1,
        "Exp": 1,
        "Expand": 1,
        "GatherElements": 1,
        "Gemm": 1,
        "Identity": 1,
        "MatMul": 1,
        "MaxPool": 2,
        "Mul": 1,
        "Log": 1,
        "Or": 1,
        "Pow": 1,
        "Relu": 1,
        "ReduceSum": 1,
        "Reshape": 1,
        "Shape": 1,
        "Slice": 1,
        "Squeeze": 1,
        "Sub": 1,
        "Transpose": 1,
        "Unsqueeze": 1,
    }

    def __init__(self, builder: "GraphBuilder", opset: int):
        self.opset = opset
        self.builder = builder

    def __getattr__(self, name):
        if name in self._implemented:
            return partial(self.make_node, name)
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            raise AttributeError(f"Unable to access attribute {name!r}.") from e

    def make_node(
        self,
        op_type: str,
        *inputs: Optional[Union[str, List[str]]],
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        **kwargs,
    ):
        if outputs is None:
            outputs = self._implemented[op_type]
        if inputs is None:
            inputs = []
        new_inputs = []
        for i in inputs:
            if not isinstance(i, str):
                name = self.builder.unique_name("cst")
                self.builder.make_initializer(i, name=name, exists=True)
                new_inputs.append(name)
            else:
                new_inputs.append(i)

        return self.builder.make_node(
            op_type, new_inputs, outputs=outputs, domain=domain, **kwargs
        )


class GraphBuilder:
    def __init__(
        self,
        target_opset_or_existing_proto: Optional[
            Union[int, Dict[str, int], ModelProto, FunctionProto]
        ] = None,
        input_names: Optional[Sequence[str]] = None,
        as_function: bool = False,
        optimization_options: Optional[OptimizationOptions] = None,
        args: Optional[List[Any]] = None,
        verbose: int = 0,
    ):
        self.optimization_options = optimization_options or OptimizationOptions()
        self.as_function = as_function
        self.input_args = args
        self.verbose = verbose

        if target_opset_or_existing_proto is None:
            target_opset_or_existing_proto = onnx_opset_version() - 1
        if isinstance(target_opset_or_existing_proto, (int, dict)):
            self.opsets = (
                {"": target_opset_or_existing_proto}
                if isinstance(target_opset_or_existing_proto, int)
                else target_opset_or_existing_proto
            )
            self.nodes = []
            self.initializers_dict = {}
            self.inputs = []
            self.outputs = []
            self._unique_names = set()
            self.input_names = input_names or []
            self.current_input = 0
            self._known_shapes = {}
            self._known_types = {}
            self.constants_ = {}
        elif isinstance(target_opset_or_existing_proto, ModelProto):
            assert (
                not input_names
            ), "input_names must be empty if the input is an existing model."
            proto = target_opset_or_existing_proto
            self.opsets = {d.domain: d.version for d in proto.opset_import}
            self.nodes = list(proto.graph.node)
            self.initializers_dict = {i.name: i for i in proto.graph.initializer}
            self.initializers_dict.update(
                {i.name: i for i in proto.graph.sparse_initializer}
            )
            self.inputs = list(proto.graph.input)
            self.outputs = list(proto.graph.output)
            self.input_names = [i.name for i in proto.graph.input]
            self.current_input = len(self.inputs)
            # This should be improve.
            self._known_shapes = {}
            self._known_types = {}
            self.constants_ = {}
            for k, v in self.initializers_dict.items():
                self.constants_[k] = None
                self.set_shape(k, self._get_tensor_shape(v))
                self.set_type(k, self._get_tensor_type(v))
            for node in self.nodes:
                if node.op_type == "Constant":
                    self.constants_[node.output[0]] = node
                    self.set_shape(node.output[0], self._get_tensor_shape(node))
                    self.set_type(node.output[0], self._get_tensor_type(node))
        else:
            raise NotImplementedError(
                f"{type(target_opset_or_existing_proto)} is not supported."
            )

        self.op = Opset(self, self.opsets[""]) if "" in self.opsets else None
        self._cache_array = []

    def _get_tensor_shape(
        self, proto: Union[NodeProto, TensorProto]
    ) -> Tuple[int, ...]:
        if isinstance(proto, TensorProto):
            return tuple(proto.dims)
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name == "value_float":
                    return tuple()
                if att.name == "value_int":
                    return tuple()
                if att.name == "value_floats":
                    return tuple(att.floats)
                if att.name == "value_ints":
                    return (len(att.ints),)
                if att.name == "value":
                    t = onh.to_array(att.t)
                    return t.shape
        raise TypeError(
            f"Unexpected or unsupported scenario type {type(proto)}: {proto}."
        )

    def _get_tensor_type(self, proto: Union[NodeProto, TensorProto]) -> int:
        if isinstance(proto, TensorProto):
            return proto.data_type
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name == "value_float":
                    return TensorProto.FLOAT
                if att.name == "value_int":
                    return TensorProto.INT64
                if att.name == "value_floats":
                    return TensorProto.FLOAT
                if att.name == "value_ints":
                    return TensorProto.INT64
                if att.name == "value":
                    t = onh.to_array(att.t)
                    return oh.np_dtype_to_tensor_dtype(t.dtype)
        raise ValueError(f"Unexpected type or value {type(proto)}: {proto}.")

    def is_constant(self, name: str) -> bool:
        """Tells if a result is a constant."""
        return name in self.constants_

    def get_constant(self, name: str) -> np.ndarray:
        assert self.is_constant(name), f"Result {name!r} is not a constant."
        assert (
            name in self.initializers_dict
        ), f"Result {name!r} was never evaluated within method 'constant_folding'."
        value = self.initializers_dict[name]
        if isinstance(value, np.ndarray):
            return value

        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def set_shape(self, name: str, shape: Tuple[int, ...]):
        assert isinstance(
            name, str
        ), f"Unexpected type {type(name)} for name, it should be a string."
        if name in self._known_shapes:
            assert shape == self._known_shapes[name], (
                f"Name {name!r} already exists and it is different "
                f"{self._known_shapes[name]} != {shape}"
            )
            return
        assert isinstance(
            shape, tuple
        ), f"Unexpected shape type {type(shape)}, it should be a tuple."
        self._known_shapes[name] = shape

    def set_type(self, name: str, dtype: int):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        int_type = dtype if isinstance(dtype, int) else self._get_type(dtype)
        if name in self._known_types:
            assert int_type == self._known_types[name], (
                f"Name {name!r} already exists and it is different "
                f"{self._known_types[name]} != {int_type}."
            )
        self._known_types[name] = int_type

    def rank(self, name: str) -> int:
        return len(self.get_shape(name))

    def has_shape(self, name: str) -> bool:
        return name in self._known_shapes

    def get_shape(self, name: str) -> int:
        assert name in self._known_shapes, (
            f"Shape is unknown for result {name!r}, "
            f"known_shapes={self._known_shapes}."
        )
        return self._known_shapes[name]

    def has_type(self, name: str) -> bool:
        return name in self._known_types

    def get_type(self, name: str) -> int:
        assert name in self._known_types, (
            f"Type is unknown for result {name!r}, " f"known_types={self._known_types}."
        )
        return self._known_types[name]

    def unique_name(self, prefix: str) -> str:
        if prefix in self._unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug in self._unique_names:
                i += 1
                sug = f"{prefix}{i}"
            self._unique_names.add(sug)
            return sug
        self._unique_names.add(prefix)
        return prefix

    def _prepare_inputs(self, schema: Optional[Any], *inputs: List[Any]) -> List[str]:
        input_names = []
        for i in inputs:
            self.make_input(i.name, i.dtype, i.shape)
            input_names.append(i.name)
        return input_names

    def _get_type(self, elem_type: Any, exc: bool = True) -> int:
        if not isinstance(elem_type, int):
            st = str(elem_type)
            if "float32" in st:
                elem_type = TensorProto.FLOAT
            elif "int64" in st:
                elem_type = TensorProto.INT64
            elif elem_type is None:
                elem_type = TensorProto.UNDEFINED
            elif exc:
                raise ValueError(f"Unable to interpret elem_type {elem_type!r}.")
        return elem_type

    def make_initializer(
        self, value: Any, name: str = "", external: bool = False, exists: bool = False
    ) -> str:
        if external:
            raise NotImplementedError("External initializers are not implemented yet.")
        if name == "":
            if exists:
                raise ValueError("Undefined name cannot exist.")
            name = self.unique_name("cst")
        elif not exists:
            if name in self._unique_names:
                raise ValueError(f"{name!r} is already assigned.")
            self._unique_names.add(name)
        self.set_shape(name, value.shape)
        self.set_type(name, self._get_type(value.dtype))
        self.initializers_dict[name] = value
        self.constants_[name] = None
        if self.verbose and np.prod(value.shape) > 100:
            print(
                f"[GraphBuilder] make_initializer:{name}[{value.dtype}:{value.shape}]"
            )
        return name

    def make_tensor_input(
        self, name: str, elem_type: Any, shape: Tuple[int, ...]
    ) -> str:
        if self.current_input < len(self.input_names):
            # The input needs to be renamed, an identity node is added.
            input_name = self.input_names[self.current_input]
            self.make_node("Identity", [input_name], [name])
        else:
            self.input_names.append(name)
            input_name = name
            if name in self._unique_names:
                raise ValueError(f"{name!r} is already assigned.")
            self._unique_names.add(name)
        self.current_input += 1
        elem_type = self._get_type(elem_type)
        self.inputs.append(oh.make_tensor_value_info(input_name, elem_type, shape))
        if self.verbose:
            print(f"[GraphBuilder] make_tensor_input:{name}[{elem_type}:{shape}]")
        if shape:
            self.set_shape(name, shape)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Union[str, List[str]]:
        if isinstance(name, list):
            res = []
            for n in name:
                res.append(self.make_tensor_output(n, elem_type, shape))
            return res

        elem_type = self._get_type(elem_type, False)
        assert (
            self.as_function or elem_type != 0
        ), f"Undefined element type for {name!r}."
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, shape))
        if self.verbose:
            print(f"[GraphBuilder] make_tensor_output:{name}[{elem_type}:{shape}]")
        if shape:
            self.set_shape(name, shape)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        assert (
            not kwargs or not attributes
        ), f"Only attributes or kwargs can be filled for node {op_type!r}."
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if isinstance(outputs, int):
            assert outputs > 0, f"outputs={outputs} must be > 0."
            lower = op_type.lower()
            output_names = [
                self.unique_name(f"_onx_{lower}{i}") for i in range(outputs)
            ]
        elif isinstance(outputs, str):
            output_names = [outputs]
        else:
            output_names = outputs
        if isinstance(inputs, str):
            inputs = [inputs]

        # next
        try:
            node = oh.make_node(op_type, inputs, output_names, domain=domain, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"A node {op_type!r} cannot be created with "
                f"inputs={inputs} (types={[type(i) for i in inputs]}), "
                f"outputs={outputs} "
                f"(types={[type(o) for o in outputs] if isinstance(outputs, (tuple, list)) else outputs}), "
                f"domain={domain!r}, kwargs={kwargs}."
            ) from e
        if attributes:
            node.attribute.extend(attributes)

        # constant handling, shape, type
        if node.op_type == "Constant":
            size = len(node.SerializeToString())
            assert size < self.optimization_options.constant_size, (
                f"A node Constant holds a tensor bigger than "
                f"the constant: {size} >= {self.constant_size}."
            )
            k = node.output[0]
            self.constants_[k] = node
            shape = self._get_tensor_shape(node)
            dtype = self._get_tensor_type(node)
            self.set_shape(k, shape)
            self.set_type(k, dtype)
            if self.verbose and np.prod(shape) > 100:
                print(f"[GraphBuilder] make_constant:{k}[{dtype}:{shape}]")
        elif node.op_type == "Identity":
            if node.input[0] in self._known_shapes:
                self.set_shape(node.output[0], self._known_shapes[node.input[0]])
            if node.input[0] in self._known_types:
                self.set_type(node.output[0], self._known_types[node.input[0]])
            if self.is_constant(node.input[0]):
                self.constants_[node.output[0]] = node
        else:
            if all(map(self.is_constant, node.input)):
                for o in node.output:
                    self.constants_[o] = node

        # add the node
        self.nodes.append(node)
        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def make_nodes(
        self,
        builder: "GraphBuilder",
        input_names: List[str],
        output_names: List[str],
        prefix: str = "",
    ) -> Union[str, List[str]]:
        """
        Appends all nodes and initializers from another builder.
        Handles the renaming of results.
        The content stored in 'builder' is modified inplace to avoid copying.

        :param builder: other builder
        :param input_names: input names
        :param output_names: output names
        :param prefix: prefix all name from this builder
        :return: output names
        """
        renaming = {}
        for init, value in builder.initializers_dict.items():
            name = self.unique_name(f"{prefix}{init}")
            renaming[init] = name
            if isinstance(value, TensorProto):
                value.name = name
            self.initializers_dict[name] = value

            self.constants_[name] = None
            self.set_shape(name, builder._known_shapes[init])
            self.set_type(name, builder._known_types[init])

        assert len(input_names) == len(builder.inputs), (
            f"Inconsistency between input_names={input_names} "
            f"and the other builder inputs={builder.inputs}."
        )

        for name, inp in zip(input_names, builder.inputs):
            new_name = self.unique_name(f"{prefix}{inp.name}")
            renaming[inp.name] = new_name
            if builder.has_shape(inp.name):
                self.set_shape(new_name, builder.get_shape(inp.name))
            if builder.has_type(inp.name):
                self.set_type(new_name, builder.get_type(inp.name))
            self.make_node("Identity", [name], [new_name])

        for node in builder.nodes:
            new_inputs = [renaming[i] for i in node.input]
            new_outputs = [self.unique_name(f"{prefix}{o}") for o in node.output]
            for o, no in zip(node.output, new_outputs):
                renaming[o] = no
            self.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                domain=node.domain,
                attributes=node.attribute,
            )
            for o, no in zip(node.output, new_outputs):
                if builder.has_shape(o):
                    self.set_shape(no, builder.get_shape(o))
                if builder.has_type(o):
                    self.set_type(no, builder.get_type(o))

        assert len(output_names) == len(builder.outputs), (
            f"Inconsistency between output_names={output_names} and "
            f"outputs={builder.outputs}, renaming={renaming}."
        )
        for name, out in zip(output_names, builder.outputs):
            self.make_node("Identity", [renaming[out.name]], [name])

        # opsets and domains
        for o, v in builder.opsets.items():
            if o in self.opsets:
                assert self.opsets[o] == builder.opsets[o], (
                    f"Opset mismatch for domain {o!r}, "
                    f"{self.opsets[o]} != {builder.opsets[o]}."
                )
                continue
            self.opsets[o] = v

        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def from_array(self, arr: T, name: str = None) -> TensorProto:  # noqa: F821
        if isinstance(arr, np.ndarray):
            return self.from_np_array(arr, name)
        raise NotImplementedError(
            f"{type(arr)} is not supported yet but initializer {name or ''!r} is."
        )

    def from_np_array(self, arr: np.ndarray, name: str = None) -> TensorProto:
        arr_cpu = np.ascontiguousarray(arr) if not arr.flags["C_CONTIGUOUS"] else arr
        if arr_cpu.ctypes.data == arr.ctypes.data:
            if sys.byteorder == "big":
                arr_cpu = arr_cpu.copy()
                np.byteswap(
                    np.frombuffer(arr_cpu.ctypes.data, dtype=arr_cpu.dtype),
                    inplace=True,
                )
        else:
            if sys.byteorder == "big":
                np.byteswap(
                    np.frombuffer(arr_cpu.ctypes.data, dtype=arr_cpu.dtype),
                    inplace=True,
                )
        # let's the tensor until the builder is released
        # so the pointer does not disappear
        self._cache_array.append(arr_cpu)

        tensor = TensorProto()
        tensor.dims.extend(arr_cpu.shape)
        tensor.name = name
        tensor.data_type = self._get_type(arr_cpu.dtype)
        # this does not work...
        # tensor.raw_data = arr_cpu.ctypes.data
        tensor.raw_data = arr_cpu.tobytes()
        if self.verbose and np.prod(arr_cpu.shape) > 100:
            print(
                f"[GraphBuilder] from_array:{tensor.data_type}[{arr_cpu.shape}]:"
                f"{'swapped' if sys.byteorder == 'big' else ''}"
            )
        return tensor

    def _build_initializers(self) -> List[TensorProto]:
        res = []
        for k, v in sorted(self.initializers_dict.items()):
            if isinstance(v, np.ndarray):
                if np.prod(v.shape) > 100:
                    if self.verbose:
                        print(f"[GraphBuilder] from_array:{k}:{v.dtype}[{v.shape}]")
                    t = self.from_array(v, name=k)
                else:
                    t = onh.from_array(v, name=k)
                res.append(t)
                continue
            if isinstance(v, TensorProto):
                res.append(v)
                continue
            raise TypeError(
                f"Unable to convert initializer {k!r} with type "
                f"{type(v)} into a TensorProto."
            )
        return res

    def process(
        self,
        graph_module: Any,
        interpreter: "Interpreter",  # noqa: F821
    ):
        for node in graph_module.graph.nodes:
            interpreter.run_node(node)

    def to_onnx(
        self, as_function: bool = False, optimize: bool = True
    ) -> Union[FunctionProto, ModelProto]:
        if optimize:
            self.optimize()
        if as_function:
            raise NotImplementedError("Export as FunctionProto is not implemented yet.")
        dense = self._build_initializers()
        opsets = [oh.make_opsetid(*o) for o in self.opsets.items()]
        if as_function:
            return oh.make_function(
                self.nodes,
                self.name,
                [i.name for i in self.inputs],
                [o.name for o in self.outputs],
                domain=self.domain,
            )

        if self.verbose:
            print("[GraphBuilder] onh.make_graph")
        graph = oh.make_graph(
            self.nodes, "experiment", self.inputs, self.outputs, dense
        )
        if self.verbose:
            print("[GraphBuilder] onh.make_model")
        model = oh.make_model(graph, opset_imports=opsets)
        return model

    def _check_order_node(self, ind: int, node: NodeProto, existing: Set[str]):
        for i in node.input:
            assert i in existing, (
                f"Unknown input {i!r} from node {ind}:{node.op_type}:{node.name}. "
                f"Known: {existing}."
            )
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH and att.g:
                g_existing = existing.copy()
                for i in att.g.input:
                    g_existing.add(i.name)
                for ind2, node2 in enumerate(att.g.node):
                    self._check_order_node((ind, ind2), node2, g_existing)
                for o in att.g.output:
                    assert (
                        o.name in g_existing
                    ), f"Unknown output {o.name!r}. Known: {g_existing}."
        for o in node.output:
            existing.add(o)

    def check_order(self):
        existing = set(self.initializers_dict)
        for i in self.inputs:
            existing.add(i.name)
        for ind, node in enumerate(self.nodes):
            self._check_order_node(ind, node, existing)
        for o in self.outputs:
            assert o.name in existing, f"Unknown output {o.name!r}. Known: {existing}."

    def optimize(self, check_order: bool = False):
        if check_order:
            self.check_order()
        self.remove_identity_nodes()
        if check_order:
            self.check_order()
        if self.optimization_options.remove_unused:
            self.remove_unused()
            if check_order:
                self.check_order()
        if self.optimization_options.constant_folding:
            self.constant_folding()
            if check_order:
                self.check_order()
            if self.optimization_options.remove_unused:
                self.remove_unused()
                if check_order:
                    self.check_order()

    def hidden_inputs_graph(self, graph: GraphProto) -> Set[str]:
        hidden = set()
        memo = set(i.name for i in graph.initializer)
        memo |= set(i.name for i in graph.sparse_initializer)
        for node in graph.node:
            for i in node.input:
                if i not in memo:
                    hidden.add(i)
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH and att.g:
                    hid = self.hidden_inputs_graph(att.g)
                    less = set(h for h in hid if h not in memo)
                    hidden |= less
            memo |= set(node.output)
        return hidden

    def remove_unused(self):
        """
        Simple function to remove unused nodes.
        It does not look into subgraphs and assumes there is none.
        Everything is done in one pass.
        """

        # mark outputs
        marked = {o.name: set() for o in self.outputs}
        for node in reversed(self.nodes):
            used = False
            for o in node.output:
                if o in marked:
                    for i in node.input:
                        marked[o].add(i)
                        used = True
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH and att.g:
                    hidden_inputs = self.hidden_inputs_graph(att.g)
                    for i in hidden_inputs:
                        marked[i] = set()
            if used:
                for i in node.input:
                    marked[i] = set()

        # removed nodes
        removed = set()
        marked_set = set(marked)
        for ind, node in enumerate(self.nodes):
            if not (set(node.output) & marked_set):
                removed.add(ind)

        if self.verbose:
            for k, v in self.initializers_dict.items():
                if k not in marked:
                    v = self.initializers_dict[k]
                    print(f"[GraphBuilder] remove_initializer:{k}:{v.dtype}[{v.shape}]")
        self.initializers_dict = {
            k: v for k, v in self.initializers_dict.items() if k in marked
        }
        self.constants_ = {k: v for k, v in self.constants_.items() if k in marked}
        self.nodes = [node for i, node in enumerate(self.nodes) if i not in removed]

    def _apply_transpose(self, node: NodeProto, feeds: Dict[str, T]) -> T:  # noqa: F821
        perm = None
        for att in node.attribute:
            if att.name == "perm":
                perm = tuple(att.ints)
                break
        assert perm, f"perm not here in node {node}"
        return [np.transpose(feeds[node.input[0]], perm)]

    def constant_folding(self):
        """
        Folds all constants. Constants are marked during the creation of the graph.
        There is no need to propagate this information.
        """
        updates = {}
        node_to_remove = set()
        for k, v in self.constants_.items():
            if v is None:
                # this is an initiliazer
                continue
            # a node
            if all(map(self.is_constant, v.output)):
                node_to_remove.add(tuple(v.output))
                # node evaluation
                if v.op_type == "Transpose":
                    # bypassing onnx.numpy_helper.from_array, too slow
                    feeds = {i: self.initializers_dict[i] for i in v.input}
                    output = self._apply_transpose(v, feeds)
                else:
                    ref = ReferenceEvaluator(v)
                    feeds = {i: self.get_constant(i) for i in v.input}
                    output = ref.run(None, feeds)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    self.initializers_dict[name] = value
                    if self.verbose:
                        print(
                            f"[GraphBuilder] fold_constant:{v.op_type}:{name}[{value.dtype}:"
                            f"{value.shape}]:from:{','.join(sorted(feeds))}"
                        )

        self.constants_.update(updates)
        new_nodes = []
        for node in self.nodes:
            if tuple(node.output) in node_to_remove:
                continue
            new_nodes.append(node)
        self.nodes = new_nodes

    def remove_identity_nodes(self):
        """
        Removes identity nodes.
        """
        # first pass: detect replacements
        new_nodes = []
        input_names = set(i.name for i in self.inputs)
        output_names = set(i.name for i in self.outputs)
        replacements = {}
        replacements_rev = {}
        for node in self.nodes:
            if node.op_type != "Identity":
                new_nodes.append(node)
                continue

            if node.output[0] not in output_names:
                old_name, new_name = node.output[0], node.input[0]
            elif (
                node.input[0] not in input_names
                and node.input[0] not in output_names
                and node.input[0] not in replacements
            ):
                old_name, new_name = node.input[0], node.output[0]
            else:
                new_nodes.append(node)
                continue

            # the new name can be set for replacements as well
            if new_name in replacements:
                new_name = replacements[new_name]
                assert new_name not in replacements, (
                    f"Name {old_name!r} still in {replacements}, node.op_type={node.op_type!r}, "
                    f"node.input={node.input}, node.output={node.output}, "
                    f"input_names={input_names}, output_names={output_names}"
                )
            if old_name in replacements_rev:
                old_old_name = replacements_rev[old_name]
                replacements[old_old_name] = new_name
                replacements_rev[new_name] = old_old_name
            if old_name in replacements:
                replacements[replacements[old_name]] = new_name
            assert new_name not in replacements, (
                f"Name {old_name!r} still in {replacements}, node.op_type={node.op_type!r}, "
                f"node.input={node.input}, node.output={node.output}, "
                f"input_names={input_names}, output_names={output_names}"
            )
            replacements[old_name] = new_name
            replacements_rev[new_name] = old_name

            # verification
            for k, v in replacements.items():
                assert v not in replacements, (
                    f"replacement {k}->{v} is not possible because of "
                    f"{v}->{replacements[v]}, old_name={old_name!r}, new_name={new_name!r}"
                )

        # second pass: replacements in initializer
        for k, v in replacements.items():
            if k in self.initializers_dict:
                self.initializers_dict[v] = self.initializers_dict[k]
                del self.initializers_dict[k]
                assert self.constants_[v]
                self.constants_[v] = self.constants_[k]
                del self.constants_[k]

        # third pass: replacements in node
        self.nodes = []
        for node in new_nodes:
            repo = {o for o in node.output if o in replacements}
            repi = {o for o in node.input if o in replacements}
            if repi or repo:
                new_inputs = [replacements.get(i, i) for i in node.input]
                new_outputs = [replacements.get(i, i) for i in node.output]
                new_node = oh.make_node(
                    node.op_type,
                    new_inputs,
                    new_outputs,
                    domain=node.domain,
                    name=node.name,
                )
                new_node.attribute.extend(node.attribute)
                self.nodes.append(new_node)
            else:
                self.nodes.append(node)

    def np(
        self,
        index: Optional[int] = None,
        op_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> NodePattern:
        """
        Returns an instance of :class:`NodePattern
        <onnx_array_api.graph_api.graph_builder.NodePattern>`.
        """
        return NodePattern(index=index, op_type=op_type, name=name)

    def update_attribute(
        self,
        pat: NodePattern,
        recursive: bool = False,
        **kwargs: Dict[str, Any],
    ) -> int:
        """
        Udates attributes for nodes matching the

        :param pat: returned by method :meth:`GraphBuilder.np`
        :param recursive: walk through subgraph
        :param kwargs: attributes to modify
        :return: number of modified nodes
        """
        assert not recursive, "recursive=True is not implemented."
        modified = 0
        for node in pat.find(self):
            up = self.update_node(node, **kwargs)
            if up:
                modified += 1
        return modified

    DELETE = object()

    def update_node(self, node: NodeProto, **kwargs) -> bool:
        """
        Updates attributes of a node proto.
        Returns True if the node was updated.
        """
        processed = set()
        modified = True
        atts = []
        for att in node.attribute:
            if att.name in kwargs:
                processed.add(att.name)
                if kwargs[att.name] is GraphBuilder.DELETE:
                    continue
                new_att = oh.make_attribute(att.name, kwargs[att.name])
                assert new_att.type == att.type, (
                    f"Mismatch value for attribute {att.name!r} has type "
                    f"{att.type} but the new value leads to "
                    f"type={new_att.type}."
                )
                atts.append(new_att)
                modified = True
                continue
            atts.append(att)
        for k, v in kwargs.items():
            if k in processed or v is GraphBuilder.DELETE:
                continue
            modified = True
            new_att = oh.make_attribute(k, v)
            atts.append(new_att)

        if modified:
            del node.attribute[:]
            node.attribute.extend(atts)
        return modified
