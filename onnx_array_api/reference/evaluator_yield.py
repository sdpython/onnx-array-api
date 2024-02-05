from typing import Any, Dict, List, Iterator, Optional, Tuple
from enum import IntEnum
from onnx import ModelProto
from .evaluator import ExtendedReferenceEvaluator


class ResultType(IntEnum):
    RESULT = 1
    INITIALIZER = 2
    SPARSE_INITIALIZER = 4
    INPUT = 8
    OUTPUT = 16

    def __repr__(self):
        return f"{self.__class__.__name__}.{self._name_}"


class YieldEvaluator:
    """
    This class implements method `enumerate_results` which iterates on
    intermediates results. By default, it uses
    :class:`onnx_array_api.evaluator.ExtendedReferenceEvaluator`.

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
    ) -> Iterator[Tuple[ResultType, str, Any]]:
        """
        Executes the onnx model.

        Args:
            output_names: requested outputs by names, None for all
            feed_inputs: dictionary `{ input name: input value }`

        Returns:
            iterator on tuple(result kind, name, value)
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
            yield ResultType.INITIALIZER, k, v
        # step 1: inputs
        for k, v in feed_inputs.items():
            yield ResultType.INPUT, k, v

        # step 2: execute nodes
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
            if node.need_context():
                outputs = node.run(*inputs, context=results, **linked_attributes)
            else:
                outputs = node.run(*inputs, **linked_attributes)
            for name, value in zip(node.output, outputs):
                yield ResultType.RESULT, name, value
                results[name] = value

        # step 3: outputs
        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} in {sorted(results)}, proto is\n{self.proto_}"
                )
            yield ResultType.OUTPUT, name, results[name]
