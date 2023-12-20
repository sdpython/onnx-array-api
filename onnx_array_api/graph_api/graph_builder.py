from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import AttributeProto, FunctionProto, ModelProto, NodeProto, TensorProto
from onnx.reference import ReferenceEvaluator


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
        "Relu": 1,
        "Reshape": 2,
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
                self.builder.make_initializer(name, i)
                new_inputs.append(name)
            else:
                new_inputs.append(i)

        return self.builder.make_node(
            op_type, new_inputs, outputs=outputs, domain=domain, **kwargs
        )


class OptimizationOptions:
    def __init__(
        self,
        remove_unused: bool = False,
        constant_folding: bool = True,
        constant_size: int = 1024,
    ):
        self.remove_unused = remove_unused
        self.constant_folding = constant_folding
        self.constant_size = constant_size


class GraphBuilder:
    def __init__(
        self,
        target_opset_or_existing_proto: Union[
            int, Dict[str, int], ModelProto, FunctionProto
        ],
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
            if input_names:
                raise ValueError(
                    "input_names must be empty if the input is an existing model."
                )
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

        self.op = Opset(self, self.opsets[""])

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
                    return tuple(att.i)
                if att.name == "value_floats":
                    return tuple(att.floats)
                if att.name == "value_ints":
                    return (len(att.ints),)
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
        raise ValueError(f"Unexpected type or value {type(proto)}: {proto}.")

    def is_constant(self, name: str) -> bool:
        """Tells if a result is a constant."""
        return name in self.constants_

    def get_constant(self, name: str) -> np.ndarray:
        if not self.is_constant(name):
            raise ValueError(f"Result {name!r} is not a constant.")
        if name not in self.initializers_dict:
            raise ValueError(
                f"Result {name!r} was never evaluated within method 'constant_folding'."
            )
        value = self.initializers_dict[name]
        if isinstance(value, np.ndarray):
            return value

        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().numpy()
        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def set_shape(self, name: str, shape: Tuple[int, ...]):
        if not isinstance(name, str):
            raise TypeError(f"Unexpected type {type(name)} for name.")
        if name in self._known_shapes:
            if shape != self._known_shapes[name]:
                raise RuntimeError(
                    f"Name {name!r} already exists and it is different "
                    f"{self._known_shapes[name]} != {shape}"
                )
            return
        if not isinstance(shape, tuple):
            raise TypeError(f"Unexpected shape type {type(shape)}.")
        self._known_shapes[name] = shape

    def set_type(self, name: str, dtype: int):
        if not isinstance(name, str):
            raise TypeError(f"Unexpected type {type(name)} for name.")
        if isinstance(dtype, int):
            int_type = dtype
        else:
            int_type = self._get_type(dtype)
        if name in self._known_types:
            if int_type != self._known_types[name]:
                raise RuntimeError(
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

    def make_initializer(self, name: str, value: Any, external: bool = False) -> str:
        if external:
            raise NotImplementedError("External initializers are not implemented yet.")
        if name == "":
            name = self.unique_name("cst")
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
        if not self.as_function and elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
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
            if outputs < 1:
                raise ValueError(f"outputs={outputs} must be > 0.")
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
            iti = [type(i) for i in inputs]
            ito = (
                [type(o) for o in outputs]
                if isinstance(outputs, (tuple, list))
                else outputs
            )
            raise TypeError(
                f"A node {op_type!r} cannot be created with "
                f"inputs={inputs} (types={iti}), outputs={outputs} (types={ito}), "
                f"domain={domain!r}, kwargs={kwargs}."
            ) from e
        if attributes:
            node.attribute.extend(attributes)

        # constant handling, shape, type
        if node.op_type == "Constant":
            size = len(node.SerializeToString())
            if size >= self.optimization_options.constant_size:
                raise ValueError(
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

        assert len(input_names) == len(
            builder.inputs
        ), f"Inconsistency between input_names={input_names} and inputs={builder.inputs}."
        for name, inp in zip(input_names, builder.inputs):
            new_name = self.unique_name(f"{prefix}{inp.name}")
            self.set_shape(new_name, builder.get_shape(inp.name))
            self.set_type(new_name, builder.get_type(inp.name))
            renaming[inp.name] = new_name
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

    def from_array(
        self, arr: "torch.Tensor", name: str = None  # noqa: F821
    ) -> TensorProto:
        import sys
        import torch

        if not isinstance(arr, torch.Tensor):
            raise TypeError(f"Unexpected type {type(arr)}.")
        if arr.is_sparse:
            raise NotImplementedError(
                f"Sparse tensor is not supported yet but initializer {name!r} is."
            )

        arr_cont = arr.contiguous() if not arr.is_contiguous() else arr
        arr_cpu = arr_cont.cpu()
        if arr_cpu.data_ptr() == arr.data_ptr():
            copy = arr_cpu.clone().detach().requires_grad_(False)
            assert arr_cpu.data_ptr() != copy.data_ptr()
            np_arr = np.from_dlpack(copy)
        else:
            np_arr = np.from_dlpack(arr_cpu.detach())

        tensor = TensorProto()
        tensor.dims.extend(arr_cpu.shape)
        tensor.name = name
        tensor.data_type = self._get_type(arr_cpu.dtype)

        if self.verbose and np.prod(arr_cpu.shape) > 100:
            print(f"[GraphBuilder] from_array:{tensor.data_type}[{arr_cpu.shape}]")

        raw = np_arr.tobytes()
        tensor.raw_data = raw

        if sys.byteorder == "big":
            np_dtype = oh.tensor_dtype_to_np_dtype(tensor.data_type)
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)
        return tensor

    def _build_initializers(self) -> List[TensorProto]:
        import torch

        res = []
        for k, v in sorted(self.initializers_dict.items()):
            if isinstance(v, torch.Tensor):
                # no string tensor
                t = self.from_array(v, name=k)
                res.append(t)
                continue
            if isinstance(v, np.ndarray):
                if self.verbose and np.prod(v.shape) > 100:
                    print(f"[GraphBuilder] onh.from_array:{k}:{v.dtype}[{v.shape}]")
                t = onh.from_array(v, name=k)
                res.append(t)
                continue
            raise TypeError(
                f"Unable to convert initializer {k!r} with type "
                f"{type(v)} into a TensorProto."
            )
        return res

    def process(
        self,
        graph_module: "torch.f.GraphModule",  # noqa: F821
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

    def optimize(self):
        self.remove_identity_nodes()
        if self.optimization_options.remove_unused:
            self.remove_unused()
        if self.optimization_options.constant_folding:
            self.constant_folding()
            if self.optimization_options.remove_unused:
                self.remove_unused()

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

    def _apply_transpose(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        import torch

        perm = None
        for att in node.attribute:
            if att.name == "perm":
                perm = tuple(att.ints)
                break
        assert perm, f"perm not here in node {node}"
        assert len(perm) == 2, f"perm={perm} is not supported with torch"
        return [torch.transpose(feeds[node.input[0]], *perm)]

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
        # f<irst pass: detect replacements
        new_nodes = []
        input_names = set(i.name for i in self.inputs)
        output_names = set(i.name for i in self.outputs)
        replacements = {}
        for node in self.nodes:
            if node.op_type != "Identity":
                new_nodes.append(node)
                continue

            if node.output[0] not in output_names:
                old_name, new_name = node.output[0], node.input[0]
            elif node.input[0] not in input_names:
                old_name, new_name = node.input[0], node.output[0]
            else:
                new_nodes.append(node)
                continue

            # the new name can be set for replacements as well
            assert old_name not in replacements
            if new_name in replacements:
                new_name = replacements[new_name]
                assert new_name not in replacements
            replacements[old_name] = new_name

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
                new_node = oh.make_node(
                    node.op_type,
                    [replacements.get(i, i) for i in node.input],
                    [replacements.get(i, i) for i in node.output],
                    domain=node.domain,
                    name=node.name,
                )
                new_node.attribute.extend(node.attribute)
                self.nodes.append(new_node)
            else:
                self.nodes.append(node)
