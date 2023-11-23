from typing import List, Optional, Union
from .annotations import AI_ONNX_ML, domain


class OpsVar:
    """
    Operators taking only one input.
    """

    def ArgMax(
        self, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
    ) -> "Var":
        return self.make_node(
            "ArgMax",
            self,
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index,
        )

    def ArgMin(
        self, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
    ) -> "Var":
        return self.make_node(
            "ArgMin",
            self,
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index,
        )

    def AveragePool(
        self,
        auto_pad: str = "NOTSET",
        ceil_mode: int = 0,
        count_include_pad: int = 0,
        dilations: Optional[List[int]] = None,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "AveragePool",
            self,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

    def Bernoulli(self, dtype: int = 0, seed: float = 0.0) -> "Var":
        return self.make_node("Bernoulli", self, dtype=dtype, seed=seed)

    def BlackmanWindow(self, output_datatype: int = 1, periodic: int = 1) -> "Var":
        return self.make_node(
            "BlackmanWindow", self, output_datatype=output_datatype, periodic=periodic
        )

    def Cast(self, saturate: int = 1, to: int = 0) -> "Var":
        return self.make_node("Cast", self, saturate=saturate, to=to)

    def Celu(self, alpha: float = 1.0) -> "Var":
        return self.make_node("Celu", self, alpha=alpha)

    def DepthToSpace(self, blocksize: int = 0, mode: str = "DCR") -> "Var":
        return self.make_node("DepthToSpace", self, blocksize=blocksize, mode=mode)

    def DynamicQuantizeLinear(
        self,
    ) -> "Vars":
        return self.make_node(
            "DynamicQuantizeLinear",
            self,
        )

    def Elu(self, alpha: float = 1.0) -> "Var":
        return self.make_node("Elu", self, alpha=alpha)

    def EyeLike(self, dtype: int = 0, k: int = 0) -> "Var":
        return self.make_node("EyeLike", self, dtype=dtype, k=k)

    def Flatten(self, axis: int = 1) -> "Var":
        return self.make_node("Flatten", self, axis=axis)

    def GlobalLpPool(self, p: int = 2) -> "Var":
        return self.make_node("GlobalLpPool", self, p=p)

    def HammingWindow(self, output_datatype: int = 1, periodic: int = 1) -> "Var":
        return self.make_node(
            "HammingWindow", self, output_datatype=output_datatype, periodic=periodic
        )

    def HannWindow(self, output_datatype: int = 1, periodic: int = 1) -> "Var":
        return self.make_node(
            "HannWindow", self, output_datatype=output_datatype, periodic=periodic
        )

    def HardSigmoid(
        self, alpha: float = 0.20000000298023224, beta: float = 0.5
    ) -> "Var":
        return self.make_node("HardSigmoid", self, alpha=alpha, beta=beta)

    def Hardmax(self, axis: int = -1) -> "Var":
        return self.make_node("Hardmax", self, axis=axis)

    def If(
        self,
        then_branch: Optional[Union["Var", "Vars", "OnnxGraph"]] = None,
        else_branch: Optional[Union["Var", "Vars", "OnnxGraph"]] = None,
    ) -> Union["Var", "Vars"]:
        attr = {}
        n_outputs = None
        for name, att in zip(
            ["then_branch", "else_branch"], [then_branch, else_branch]
        ):
            if att is None:
                raise ValueError(f"Parameter {name!r} cannot be None.")
            if hasattr(att, "to_onnx"):
                # Let's overwrite the opsets.
                att.parent.opset = self.parent.opset
                att.parent.opsets = self.parent.opsets
                graph = att.to_onnx()
                attr[name] = graph
                if n_outputs is None:
                    n_outputs = len(graph.output)
                elif n_outputs != len(graph.output):
                    raise ValueError(
                        "then and else branches have different number of outputs."
                    )
            else:
                raise ValueError(f"Unexpeted type {type(att)} for parameter {name!r}.")
        return self.make_node("If", self, **attr)

    def IsInf(self, detect_negative: int = 1, detect_positive: int = 1) -> "Var":
        return self.make_node(
            "IsInf",
            self,
            detect_negative=detect_negative,
            detect_positive=detect_positive,
        )

    def LRN(
        self,
        alpha: float = 9.999999747378752e-05,
        beta: float = 0.75,
        bias: float = 1.0,
        size: int = 0,
    ) -> "Var":
        return self.make_node("LRN", self, alpha=alpha, beta=beta, bias=bias, size=size)

    def LeakyRelu(self, alpha: float = 0.009999999776482582) -> "Var":
        return self.make_node("LeakyRelu", self, alpha=alpha)

    def LogSoftmax(self, axis: int = -1) -> "Var":
        return self.make_node("LogSoftmax", self, axis=axis)

    def LpNormalization(self, axis: int = -1, p: int = 2) -> "Var":
        return self.make_node("LpNormalization", self, axis=axis, p=p)

    def LpPool(
        self,
        auto_pad: str = "NOTSET",
        ceil_mode: int = 0,
        dilations: Optional[List[int]] = None,
        kernel_shape: Optional[List[int]] = None,
        p: int = 2,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "LpPool",
            self,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            kernel_shape=kernel_shape,
            p=p,
            pads=pads,
            strides=strides,
        )

    def MeanVarianceNormalization(self, axes: Optional[List[int]] = None) -> "Var":
        axes = axes or [0, 2, 3]
        return self.make_node("MeanVarianceNormalization", self, axes=axes)

    def Multinomial(
        self, dtype: int = 6, sample_size: int = 1, seed: float = 0.0
    ) -> "Var":
        return self.make_node(
            "Multinomial", self, dtype=dtype, sample_size=sample_size, seed=seed
        )

    def RandomNormalLike(
        self, dtype: int = 0, mean: float = 0.0, scale: float = 1.0, seed: float = 0.0
    ) -> "Var":
        return self.make_node(
            "RandomNormalLike", self, dtype=dtype, mean=mean, scale=scale, seed=seed
        )

    def RandomUniformLike(
        self, dtype: int = 0, high: float = 1.0, low: float = 0.0, seed: float = 0.0
    ) -> "Var":
        return self.make_node(
            "RandomUniformLike", self, dtype=dtype, high=high, low=low, seed=seed
        )

    def ReduceL1(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceL1",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceL2(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceL2",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceLogSum(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceLogSum",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceLogSumExp(
        self, keepdims: int = 1, noop_with_empty_axes: int = 0
    ) -> "Var":
        return self.make_node(
            "ReduceLogSumExp",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMax(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMax",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMean(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMean",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMin(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMin",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceProd(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceProd",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceSum(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceSum",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceSumSquare(
        self, keepdims: int = 1, noop_with_empty_axes: int = 0
    ) -> "Var":
        return self.make_node(
            "ReduceSumSquare",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def Selu(
        self, alpha: float = 1.6732631921768188, gamma: float = 1.0507010221481323
    ) -> "Var":
        return self.make_node("Selu", self, alpha=alpha, gamma=gamma)

    def Shrink(self, bias: float = 0.0, lambd: float = 0.5) -> "Var":
        return self.make_node("Shrink", self, bias=bias, lambd=lambd)

    def Softmax(self, axis: int = -1) -> "Var":
        return self.make_node("Softmax", self, axis=axis)

    def SpaceToDepth(self, blocksize: int = 0) -> "Var":
        return self.make_node("SpaceToDepth", self, blocksize=blocksize)

    def ThresholdedRelu(self, alpha: float = 1.0) -> "Var":
        return self.make_node("ThresholdedRelu", self, alpha=alpha)

    def Transpose(self, perm: Optional[List[int]] = None) -> "Var":
        perm = perm or []
        return self.make_node("Transpose", self, perm=perm)

    @domain(AI_ONNX_ML)
    def Normalizer(self, norm: str = "MAX"):
        return self.make_node("Normalizer", self, norm=norm, domain=AI_ONNX_ML)


def _complete():
    ops_to_add = [
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitwiseNot",
        "Ceil",
        "Cos",
        "Cosh",
        "Det",
        "Erf",
        "Exp",
        "Floor",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "HardSwish",
        "Identity",
        "IsNaN",
        "Log",
        "Mish",
        "Neg",
        "NonZero",
        "Not",
        "Reciprocal",
        "Relu",
        "Round",
        "Shape",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Size",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
    ]
    for name in ops_to_add:
        if hasattr(OpsVar, name):
            continue
        setattr(OpsVar, name, lambda self, op_type=name: self.make_node(op_type, self))


_complete()
