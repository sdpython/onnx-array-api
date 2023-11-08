from typing import List, Optional


class OpsVars:
    """
    Operators taking multiple inputs.
    """

    def BitShift(self, direction: str = "") -> "Var":
        return self.make_node("BitShift", *self.vars_, direction=direction)

    def CenterCropPad(self, axes: Optional[List[int]] = None) -> "Var":
        axes = axes or []
        return self.make_node("CenterCropPad", *self.vars_, axes=axes)

    def Clip(
        self,
    ) -> "Var":
        return self.make_node(
            "Clip",
            *self.vars_,
        )

    def Col2Im(
        self,
        dilations: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "Col2Im", *self.vars_, dilations=dilations, pads=pads, strides=strides
        )

    def Compress(self, axis: int = 0) -> "Var":
        return self.make_node("Compress", *self.vars_, axis=axis)

    def Concat(self, axis: int = 0) -> "Var":
        return self.make_node("Concat", *self.vars_, axis=axis)

    def Conv(
        self,
        auto_pad: str = "NOTSET",
        dilations: Optional[List[int]] = None,
        group: int = 1,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "Conv",
            *self.vars_,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

    def ConvInteger(
        self,
        auto_pad: str = "NOTSET",
        dilations: Optional[List[int]] = None,
        group: int = 1,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "ConvInteger",
            *self.vars_,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

    def ConvTranspose(
        self,
        auto_pad: str = "NOTSET",
        dilations: Optional[List[int]] = None,
        group: int = 1,
        kernel_shape: Optional[List[int]] = None,
        output_padding: Optional[List[int]] = None,
        output_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        output_padding = output_padding or []
        output_shape = output_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "ConvTranspose",
            *self.vars_,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            output_padding=output_padding,
            output_shape=output_shape,
            pads=pads,
            strides=strides,
        )

    def CumSum(self, exclusive: int = 0, reverse: int = 0) -> "Var":
        return self.make_node(
            "CumSum", *self.vars_, exclusive=exclusive, reverse=reverse
        )

    def DFT(self, axis: int = 1, inverse: int = 0, onesided: int = 0) -> "Var":
        return self.make_node(
            "DFT", *self.vars_, axis=axis, inverse=inverse, onesided=onesided
        )

    def DeformConv(
        self,
        dilations: Optional[List[int]] = None,
        group: int = 1,
        kernel_shape: Optional[List[int]] = None,
        offset_group: int = 1,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "DeformConv",
            *self.vars_,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            offset_group=offset_group,
            pads=pads,
            strides=strides,
        )

    def DequantizeLinear(self, axis: int = 1) -> "Var":
        return self.make_node("DequantizeLinear", *self.vars_, axis=axis)

    def Einsum(self, equation: str = "") -> "Var":
        return self.make_node("Einsum", *self.vars_, equation=equation)

    def Gather(self, axis: int = 0) -> "Var":
        return self.make_node("Gather", *self.vars_, axis=axis)

    def GatherElements(self, axis: int = 0) -> "Var":
        return self.make_node("GatherElements", *self.vars_, axis=axis)

    def Gemm(
        self, alpha: float = 1.0, beta: float = 1.0, transA: int = 0, transB: int = 0
    ) -> "Var":
        return self.make_node(
            "Gemm", *self.vars_, alpha=alpha, beta=beta, transA=transA, transB=transB
        )

    def GridSample(
        self,
        align_corners: int = 0,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> "Var":
        return self.make_node(
            "GridSample",
            *self.vars_,
            align_corners=align_corners,
            mode=mode,
            padding_mode=padding_mode,
        )

    def GroupNormalization(
        self, epsilon: float = 9.999999747378752e-06, num_groups: int = 0
    ) -> "Var":
        return self.make_node(
            "GroupNormalization", *self.vars_, epsilon=epsilon, num_groups=num_groups
        )

    def InstanceNormalization(self, epsilon: float = 9.999999747378752e-06) -> "Var":
        return self.make_node("InstanceNormalization", *self.vars_, epsilon=epsilon)

    def MatMulInteger(
        self,
    ) -> "Var":
        return self.make_node(
            "MatMulInteger",
            *self.vars_,
        )

    def MaxRoiPool(
        self, pooled_shape: Optional[List[int]] = None, spatial_scale: float = 1.0
    ) -> "Var":
        pooled_shape = pooled_shape or []
        return self.make_node(
            "MaxRoiPool",
            *self.vars_,
            pooled_shape=pooled_shape,
            spatial_scale=spatial_scale,
        )

    def MaxUnpool(
        self,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "MaxUnpool",
            *self.vars_,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

    def MelWeightMatrix(self, output_datatype: int = 1) -> "Var":
        return self.make_node(
            "MelWeightMatrix", *self.vars_, output_datatype=output_datatype
        )

    def Mod(self, fmod: int = 0) -> "Var":
        return self.make_node("Mod", *self.vars_, fmod=fmod)

    def NegativeLogLikelihoodLoss(
        self, ignore_index: int = 0, reduction: str = "mean"
    ) -> "Var":
        return self.make_node(
            "NegativeLogLikelihoodLoss",
            *self.vars_,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def NonMaxSuppression(self, center_point_box: int = 0) -> "Var":
        return self.make_node(
            "NonMaxSuppression", *self.vars_, center_point_box=center_point_box
        )

    def OneHot(self, axis: int = -1) -> "Var":
        return self.make_node("OneHot", *self.vars_, axis=axis)

    def Pad(self, mode: str = "constant") -> "Var":
        return self.make_node("Pad", *self.vars_, mode=mode)

    def QLinearConv(
        self,
        auto_pad: str = "NOTSET",
        dilations: Optional[List[int]] = None,
        group: int = 1,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        dilations = dilations or []
        kernel_shape = kernel_shape or []
        pads = pads or []
        strides = strides or []
        return self.make_node(
            "QLinearConv",
            *self.vars_,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

    def QLinearMatMul(
        self,
    ) -> "Var":
        return self.make_node(
            "QLinearMatMul",
            *self.vars_,
        )

    def QuantizeLinear(self, axis: int = 1, saturate: int = 1) -> "Var":
        return self.make_node(
            "QuantizeLinear", *self.vars_, axis=axis, saturate=saturate
        )

    def RandomNormal(
        self,
        dtype: int = 1,
        mean: float = 0.0,
        scale: float = 1.0,
        seed: float = 0.0,
        shape: Optional[List[int]] = None,
    ) -> "Var":
        shape = shape or []
        return self.make_node(
            "RandomNormal",
            *self.vars_,
            dtype=dtype,
            mean=mean,
            scale=scale,
            seed=seed,
            shape=shape,
        )

    def RandomUniform(
        self,
        dtype: int = 1,
        high: float = 1.0,
        low: float = 0.0,
        seed: float = 0.0,
        shape: Optional[List[int]] = None,
    ) -> "Var":
        shape = shape or []
        return self.make_node(
            "RandomUniform",
            *self.vars_,
            dtype=dtype,
            high=high,
            low=low,
            seed=seed,
            shape=shape,
        )

    def Range(
        self,
    ) -> "Var":
        return self.make_node(
            "Range",
            *self.vars_,
        )

    def ReduceL1(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceL1",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceL2(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceL2",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceLogSum(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceLogSum",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceLogSumExp(
        self, keepdims: int = 1, noop_with_empty_axes: int = 0
    ) -> "Var":
        return self.make_node(
            "ReduceLogSumExp",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMax(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMax",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMean(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMean",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMin(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMin",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceProd(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceProd",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceSum(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceSum",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceSumSquare(
        self, keepdims: int = 1, noop_with_empty_axes: int = 0
    ) -> "Var":
        return self.make_node(
            "ReduceSumSquare",
            *self.vars_,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def Resize(
        self,
        antialias: int = 0,
        axes: Optional[List[int]] = None,
        coordinate_transformation_mode: str = "half_pixel",
        cubic_coeff_a: float = -0.75,
        exclude_outside: int = 0,
        extrapolation_value: float = 0.0,
        keep_aspect_ratio_policy: str = "stretch",
        mode: str = "nearest",
        nearest_mode: str = "round_prefer_floor",
    ) -> "Var":
        axes = axes or []
        return self.make_node(
            "Resize",
            *self.vars_,
            antialias=antialias,
            axes=axes,
            coordinate_transformation_mode=coordinate_transformation_mode,
            cubic_coeff_a=cubic_coeff_a,
            exclude_outside=exclude_outside,
            extrapolation_value=extrapolation_value,
            keep_aspect_ratio_policy=keep_aspect_ratio_policy,
            mode=mode,
            nearest_mode=nearest_mode,
        )

    def RoiAlign(
        self,
        coordinate_transformation_mode: str = "half_pixel",
        mode: str = "avg",
        output_height: int = 1,
        output_width: int = 1,
        sampling_ratio: int = 0,
        spatial_scale: float = 1.0,
    ) -> "Var":
        return self.make_node(
            "RoiAlign",
            *self.vars_,
            coordinate_transformation_mode=coordinate_transformation_mode,
            mode=mode,
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        )

    def STFT(self, onesided: int = 1) -> "Var":
        return self.make_node("STFT", *self.vars_, onesided=onesided)

    def Scatter(self, axis: int = 0) -> "Var":
        return self.make_node("Scatter", *self.vars_, axis=axis)

    def ScatterElements(self, axis: int = 0, reduction: str = "none") -> "Var":
        return self.make_node(
            "ScatterElements", *self.vars_, axis=axis, reduction=reduction
        )

    def ScatterND(self, reduction: str = "none") -> "Var":
        return self.make_node("ScatterND", *self.vars_, reduction=reduction)

    def Slice(
        self,
    ) -> "Var":
        return self.make_node(
            "Slice",
            *self.vars_,
        )

    def TopK(self, axis: int = -1, largest: int = 1, sorted: int = 1) -> "Vars":
        return self.make_node(
            "TopK",
            *self.vars_,
            axis=axis,
            largest=largest,
            sorted=sorted,
            n_outputs=2,
        )

    def Trilu(self, upper: int = 1) -> "Var":
        return self.make_node("Trilu", *self.vars_, upper=upper)

    def Upsample(self, mode: str = "nearest") -> "Var":
        return self.make_node("Upsample", *self.vars_, mode=mode)

    def Where(
        self,
    ) -> "Var":
        return self.make_node(
            "Where",
            *self.vars_,
        )


def _complete():
    ops_to_add = [
        "Add",
        "And",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "CastLike",
        "Div",
        "Equal",
        "Expand",
        "GatherND",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "MatMul",
        "Mul",
        "Or",
        "PRelu",
        "Pow",
        "Reshape",
        "StringConcat",
        "Sub",
        "Tile",
        "Unsqueeze",
        "Xor",
    ]

    for name in ops_to_add:
        if hasattr(OpsVars, name):
            continue
        setattr(
            OpsVars,
            name,
            lambda self, op_type=name: self._check_nin(2).make_node(
                op_type, *self.vars_
            ),
        )

    ops_to_add = [
        "Squeeze",
    ]

    for name in ops_to_add:
        if hasattr(OpsVars, name):
            continue
        setattr(
            OpsVars,
            name,
            lambda self, op_type=name: self.make_node(op_type, *self.vars_),
        )


_complete()
