from dataclasses import dataclass
from typing import Any, Dict, List, Iterator, Optional, Tuple
from enum import IntEnum
import numpy as np
from onnx import ModelProto, TensorProto, ValueInfoProto
from .evaluator import ExtendedReferenceEvaluator


def _align(res: str, limit: int) -> str:
    if len(res) == limit:
        return res
    if len(res) > limit:
        return res[:limit]
    return res + " " * (limit - len(res))


class ResultType(IntEnum):
    RESULT = 1
    INITIALIZER = 2
    SPARSE_INITIALIZER = 4
    INPUT = 8
    OUTPUT = 16

    def __repr__(self):
        return f"{self.__class__.__name__}.{self._name_}"


@dataclass
class ResultExecution:
    """
    The description of a result.
    """

    kind: ResultType
    dtype: object
    shape: tuple
    summary: str
    op_type: str
    name: str

    def __len__(self) -> int:
        return 6

    def __getitem__(self, i: int) -> Any:
        if i == 0:
            return self.kind
        if i == 1:
            return self.dtype
        if i == 2:
            return self.shape
        if i == 3:
            return self.summary
        if i == 4:
            return self.op_type
        if i == 5:
            return self.name
        raise IndexError(f"i={i} out of boundary")

    def __str__(self):
        els = [
            _align(self.kind._name_, 6),
            _align(str(self.dtype).replace("dtype(", "").replace(")", ""), 8),
            _align("x".join(map(str, self.shape)), 15),
            self.summary,
            _align(self.op_type or "", 10),
            self.name or "",
        ]
        return " ".join(els)


def make_summary(value: Any, length: int = 4, modulo: int = 26) -> str:
    """
    Create a short string summarizing the value (discretization).

    :param value: array
    :param length: number of value to produce
    :param module: discretization parameter
    :return: short string
    """
    assert isinstance(
        value, np.ndarray
    ), f"Unexpected type {type(value)} for value, it must be a numpy array."
    value4 = np.zeros(length, dtype=np.float64)
    if value.size <= length:
        value4[: value.size] = value.flatten().astype(np.float64)
    else:
        if value.size % length != 0:
            value2 = np.zeros(
                value.size + length - value.size % length, dtype=np.float64
            )
            value2[: value.size] = value.flatten().astype(np.float64)
        else:
            value2 = value.flatten().astype(np.float64)
        value4 = value2.reshape((4, -1)).sum(axis=1)
    value4i = value4.astype(np.int64) % modulo
    s = "".join([chr(65 + i) for i in value4i])
    return s


class YieldEvaluator:
    """
    This class implements method `enumerate_results` which iterates on
    intermediates results. By default, it uses
    :class:`onnx_array_api.reference.ExtendedReferenceEvaluator`.

    :param onnx_model: model to run
    :param recursive: dig into subgraph and functions as well
    """

    def __init__(
        self,
        onnx_model: ModelProto,
        recursive: bool = False,
        cls=ExtendedReferenceEvaluator,
    ):
        assert not recursive, "recursive=True is not yet implemented"
        self.onnx_model = onnx_model
        self.evaluator = cls(onnx_model) if cls is not None else None

    def enumerate_results(
        self,
        output_names: Optional[List[str]] = None,
        feed_inputs: Optional[Dict[str, Any]] = None,
        raise_exc: bool = True,
    ) -> Iterator[Tuple[ResultType, str, Any]]:
        """
        Executes the onnx model and enumerate all the intermediate results.

        Args:
            output_names: requested outputs by names, None for all
            feed_inputs: dictionary `{ input name: input value }`

        Returns:
            iterator on tuple(result kind, name, value, node.op_type or None)
        """
        assert isinstance(self.evaluator, ExtendedReferenceEvaluator), (
            f"This implementation only works with "
            f"ExtendedReferenceEvaluator not {type(self.evaluator)}"
        )
        attributes = {}
        if output_names is None:
            output_names = self.evaluator.output_names

        results = {"": None}
        results.update(self.evaluator.rt_inits_)
        results.update(feed_inputs)
        # step 0: initializer
        for k, v in self.evaluator.rt_inits_.items():
            yield ResultType.INITIALIZER, k, v, None
        # step 1: inputs
        for k, v in feed_inputs.items():
            yield ResultType.INPUT, k, v, None

        # step 2: execute nodes
        yield_output = True
        for node in self.evaluator.rt_nodes_:
            for i in node.input:
                if i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results {sorted(results)}, "
                        f"self.rt_inits_ has {sorted(self.evaluator.rt_inits_)}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [results[i] for i in node.input]
            linked_attributes = {}
            if node.has_linked_attribute and attributes:
                linked_attributes["linked_attributes"] = attributes

            try:
                if node.need_context():
                    outputs = node.run(*inputs, context=results, **linked_attributes)
                else:
                    outputs = node.run(*inputs, **linked_attributes)
            except Exception:
                if raise_exc:
                    raise
                yield_output = False
                break

            for name, value in zip(node.output, outputs):
                yield ResultType.RESULT, name, value, node.op_type
                results[name] = value

        # step 3: outputs
        if yield_output:
            for name in output_names:
                if name not in results:
                    raise RuntimeError(
                        f"Unable to find output name {name!r} in {sorted(results)}, proto is\n{self.proto_}"
                    )
                yield ResultType.OUTPUT, name, results[name], None

    def enumerate_summarized(
        self,
        output_names: Optional[List[str]] = None,
        feed_inputs: Optional[Dict[str, Any]] = None,
        raise_exc: bool = True,
    ) -> Iterator[ResultExecution]:
        """
        Executes the onnx model and enumerate intermediate results without their names.

        :param output_names: requested outputs by names, None for all
        :param feed_inputs: dictionary `{ input name: input value }`
        :param raise_exc: raises an exception if the execution fails or stop
            where it is
        :return: iterator on ResultExecution
        """
        for kind, name, value, op_type in self.enumerate_results(
            output_names, feed_inputs, raise_exc=raise_exc
        ):
            summary = make_summary(value)
            yield ResultExecution(
                kind, value.dtype, value.shape, summary, op_type, name
            )


class DistanceExecution:
    """
    Computes a distance between two results.
    """

    float_types = {
        np.float16,
        np.float32,
        np.float64,
        np.dtype("float16"),
        np.dtype("float32"),
        np.dtype("float64"),
    }

    def __init__(self, max_lag: int = 50):
        self.kind_cost = 1000
        self.type_cost = 10
        self.rank_cost = 100
        self.op_type_cost = 10
        self.max_lag = max_lag
        self.insert_cost = 1000

    def distance_pair(self, r1: ResultExecution, r2: ResultExecution) -> float:
        """
        (ResultType.RESULT, np.dtype("float32"), (2, 2), "CEIO", "Abs"),

        :param r1: first result
        :param r2: second result
        :return: distance
        """
        d = 0
        if r1[0] != r2[0]:
            # difference type
            d += self.kind_cost
        if r1[1] != r2[1]:
            d += self._cost_type(r1[1], r2[1]) * self.type_cost
        if r1[2] != r2[2]:
            d += self._cost_shape(r1[2], r2[2])
        if r1[3] != r2[3]:
            d += self._cost_summary(r1[3], r2[3])
        if r1[4] != r2[4]:
            d += self.op_type_cost
        return d

    def _cost_type(self, t1: "np.dtype", t2: "np.dtype") -> float:
        if t1 in self.float_types and t2 in self.float_types:
            return 0.2
        return 1

    def _cost_shape(self, s1: Tuple[int, ...], s2: Tuple[int, ...]) -> float:
        d = abs(np.prod(s1) - np.prod(s2))
        if len(s1) != len(s2):
            return self.rank_cost + d
        for i, j in zip(s1, s2):
            d += abs(i - j)
        return d

    def _cost_summary(self, s1: str, s2: str) -> float:
        if len(s1) != len(s2):
            return 1e6
        d = 0
        for a, b in zip(s1, s2):
            d += abs(ord(a) - ord(b))
        return d

    def distance_sequence(
        self, s1: List[ResultExecution], s2: List[ResultExecution]
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Computes the distance between two sequences of results.

        :param s1: first sequence
        :param s2: second sequence
        :return: distance and alignment
        """
        delay = self.max_lag
        distance = {(-1, -1): 0}
        predecessor = {(-1, -1): None}
        for i in range(len(s1)):
            for j in range(max(0, i - delay), min(len(s2), i + delay)):
                best = 1e100
                pred = None
                ki, kj = i - 1, j - 1
                if (ki, kj) in distance:
                    d = distance[ki, kj] + self.distance_pair(s1[i], s2[j])
                    if d < best:
                        best = d
                        pred = (ki, kj)
                ki, kj = i - 1, j
                if (ki, kj) in distance:
                    d = distance[ki, kj] + self.insert_cost
                    if d < best:
                        best = d
                        pred = (ki, kj)
                ki, kj = i, j - 1
                if (ki, kj) in distance:
                    d = distance[ki, kj] + self.insert_cost
                    if d < best:
                        best = d
                        pred = (ki, kj)
                distance[i, j] = best
                predecessor[i, j] = pred

        # reverse
        way = []
        last = len(s1) - 1, len(s2) - 1
        while last is not None:
            way.append(last)
            last = predecessor[last]
        return distance[len(s1) - 1, len(s2) - 1], list(reversed(way))[1:]

    def to_str(
        self,
        s1: List[ResultExecution],
        s2: List[ResultExecution],
        alignment: List[Tuple[int, int]],
        column_size: int = 60,
    ) -> str:
        """
        Prints out the alignment between two sequences into a string.
        :param s1: first sequence
        :param s2: second sequence
        :param alignment: alignment
        :param column_size: column size
        :return: test
        """
        rows = []
        last = -1, -1
        row_index = 1
        for i, j in alignment:
            assert i < len(s1), f"Unexpected value i={i} >= len(s1)={len(s1)}"
            assert j < len(s2), f"Unexpected value i={j} >= len(s2)={len(s2)}"
            expected = last[0] + 1, last[1] + 1

            if expected == (i, j):
                d1 = s1[i]
                d2 = s2[j]
                d = self.distance_pair(d1, d2)
                symbol = "=" if d == 0 else "~"
                line = f"{symbol} | {_align(str(d1), column_size)} | {_align(str(d2), column_size)}"
            elif i == last[0]:
                d2 = s2[j]
                line = (
                    f"+ | {_align('', column_size)} | {_align(str(d2), column_size)} "
                )
            else:
                d1 = s1[i]
                line = f"- | {_align(str(d1), column_size)} | {_align('', column_size)}"
            rows.append(f"{row_index: 3d} {line}")
            last = i, j
            row_index += 1
        return "\n".join(rows)


def generate_input(info: ValueInfoProto) -> np.ndarray:
    """
    Generates one input.
    """
    elem_type = info.type.tensor_type.elem_type
    shape = [
        (getattr(d, "dim_value", None) or getattr(d, "dim_param"))
        for d in info.type.tensor_type.shape.dim
    ]
    new_shape = []
    for sh in shape:
        if isinstance(sh, str):
            if len(new_shape) == 0:
                new_shape.append(1)
            else:
                new_shape.append(16)
        else:
            new_shape.append(sh)
    new_shape = tuple(new_shape)
    p = np.prod(new_shape)
    value = np.arange(p)
    if elem_type == TensorProto.INT32:
        return value.astype(np.int32).reshape(new_shape)
    if elem_type == TensorProto.INT64:
        return value.astype(np.int64).reshape(new_shape)
    if elem_type == TensorProto.FLOAT:
        return (value.astype(np.float32) / p).astype(np.float32).reshape(new_shape)
    if elem_type == TensorProto.FLOAT16:
        return (value.astype(np.float16) / p).astype(np.float16).reshape(new_shape)
    if elem_type == TensorProto.DOUBLE:
        return (value.astype(np.float64) / p).astype(np.float64).reshape(new_shape)
    raise RuntimeError(f"Unexpected element_type {elem_type} for info={info}")


def generate_inputs(model: ModelProto) -> List[np.ndarray]:
    """
    Generates inputs for a specific model.

    :param model: ModelProto
    :return: list of inputs
    """
    inputs = []
    inits = set(i.name for i in model.graph.initializer)
    for inp in model.graph.input:
        if inp.name in inits:
            break
        inputs.append(generate_input(inp))
    return inputs


def compare_onnx_execution(
    model1: ModelProto,
    model2: ModelProto,
    inputs: Optional[List[Any]] = None,
    verbose: int = 0,
    raise_exc: bool = True,
) -> Tuple[List[ResultExecution], List[ResultExecution], List[Tuple[int, int]]]:
    """
    Compares the execution of two onnx models.
    The function assumes both models takes the same inputs.
    See :ref:`l-onnx-diff-example` to see a full example using
    this function.

    :param model1: first model
    :param model2: second model
    :param inputs: inputs to use
    :param verbose: verbosity
    :param raise_exc: raise exception if the execution fails or stop at the error
    :return: four results, a sequence of results for the first model and the second model,
        the alignment between the two, DistanceExecution
    """
    if verbose:
        print("[compare_onnx_execution] generate inputs")
    if inputs is None:
        inputs = generate_inputs(model1)
    feeds1 = {i.name: v for i, v in zip(model1.graph.input, inputs)}
    feeds2 = {i.name: v for i, v in zip(model2.graph.input, inputs)}
    if verbose:
        print(f"[compare_onnx_execution] got {len(inputs)} inputs")
        print("[compare_onnx_execution] execute first model")
    res1 = list(
        YieldEvaluator(model1).enumerate_summarized(None, feeds1, raise_exc=raise_exc)
    )
    if verbose:
        print(f"[compare_onnx_execution] got {len(res1)} results")
        print("[compare_onnx_execution] execute second model")
    res2 = list(
        YieldEvaluator(model2).enumerate_summarized(None, feeds2, raise_exc=raise_exc)
    )
    if verbose:
        print(f"[compare_onnx_execution] got {len(res2)} results")
        print("[compare_onnx_execution] compute edit distance")
    dc = DistanceExecution()
    _, align = dc.distance_sequence(res1, res2)
    if verbose:
        print(f"[compare_onnx_execution] got {len(align)} pairs")
        print("[compare_onnx_execution] done")
    return res1, res2, align, dc
