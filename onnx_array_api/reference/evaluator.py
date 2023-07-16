from logging import getLogger
from typing import Any, Dict, List, Optional, Union
from onnx import FunctionProto, ModelProto
from onnx.defs import get_schema
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from .ops.op_cast_like import CastLike_15, CastLike_19
from .ops.op_concat import Concat
from .ops.op_constant_of_shape import ConstantOfShape


logger = getLogger("onnx-array-api-eval")


class ExtendedReferenceEvaluator(ReferenceEvaluator):
    """
    This class replaces the python implementation by custom implementation.
    The Array API extends many operator to all types not supported
    by the onnx specifications. The evaluator allows to test
    scenarios outside what an onnx backend bound to the official onnx
    operators definition could do.

    ::

        from onnx.reference import ReferenceEvaluator
        from onnx.reference.c_ops import Conv
        ref = ReferenceEvaluator(..., new_ops=[Conv])
    """

    default_ops = [
        Concat,
        CastLike_15,
        CastLike_19,
        ConstantOfShape,
    ]

    @staticmethod
    def filter_ops(proto, new_ops, opsets):
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best = {}
        renamed = {}
        for cl in new_ops:
            if "_" not in cl.__name__:
                continue
            vers = cl.__name__.split("_")
            try:
                v = int(vers[-1])
            except ValueError:
                # not a version
                continue
            if opsets is not None and v > opsets.get(cl.op_domain, 1):
                continue
            renamed[cl.__name__] = cl
            key = cl.op_domain, "_".join(vers[:-1])
            if key not in best or best[key][0] < v:
                best[key] = (v, cl)

        modified = []
        for cl in new_ops:
            if cl.__name__ not in renamed:
                modified.append(cl)
        for k, v in best.items():
            atts = {"domain": k[0]}
            bases = (v[1],)
            if not hasattr(v[1], "op_schema"):
                atts["op_schema"] = get_schema(k[1], v[0], domain=v[1].op_domain)
            new_cl = type(k[1], bases, atts)
            modified.append(new_cl)

        new_ops = modified
        return new_ops

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
        **kwargs,
    ):
        if new_ops is None:
            new_ops = ExtendedReferenceEvaluator.default_ops
        else:
            new_ops = new_ops.copy()
            new_ops.extend(ExtendedReferenceEvaluator.default_ops)
        new_ops = ExtendedReferenceEvaluator.filter_ops(proto, new_ops, opsets)

        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
            **kwargs,
        )

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))
        else:
            logger.debug(pattern, *args)

    def run(self, *args, **kwargs):
        """
        See :meth:`onnx.reference.ReferenceEvaluator.run`.
        """
        return ReferenceEvaluator.run(self, *args, **kwargs)
