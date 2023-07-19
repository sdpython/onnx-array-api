from typing import Tuple, Union
import array_api_compat.numpy as np_array_api
import numpy as np
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto
from onnx.helper import make_tensor, tensor_dtype_to_np_dtype
from ..reference import from_array_extended as from_array
from .npx_constants import FUNCTION_DOMAIN
from .npx_core_api import cst, make_tuple, npxapi_inline, npxapi_no_inline, var
from .npx_types import (
    DType,
    ElemType,
    OptParType,
    OptTensorType,
    ParType,
    Scalar,
    SequenceType,
    TensorType,
    TupleType,
)
from .npx_var import Var


def _cstv(x):
    if isinstance(x, Var):
        return x
    if isinstance(x, (int, float, bool, np.ndarray)):
        return cst(x)
    raise TypeError(f"Unexpected constant type {type(x)}.")


@npxapi_inline
def abs(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.absolute`."
    return var(x, op="Abs")


@npxapi_inline
def absolute(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.absolute`."
    return var(x, op="Abs")


@npxapi_inline
def all(
    x: TensorType[ElemType.bool_, "T"],
    /,
    *,
    axis: OptTensorType[ElemType.int64, "I"] = None,
    keepdims: ParType[int] = 0,
) -> TensorType[ElemType.bool_, "T"]:
    """
    See :func:`numpy.all`.
    If input x is empty, the answer is True.
    """
    xi = var(x, op="Cast", to=TensorProto.INT64)

    if axis is None:
        new_shape = cst(np.array([-1], dtype=np.int64))
        xifl = var(xi, new_shape, op="Reshape")
        # in case xifl is empty, we need to add one element
        one = cst(np.array([1], dtype=np.int64))
        xifl1 = var(xifl, one, op="Concat", axis=0)
        red = xifl1.min(keepdims=keepdims)
    else:
        if isinstance(axis, int):
            axis = [axis]
        if isinstance(axis, (tuple, list)):
            axis = cst(np.array(axis, dtype=np.int64))
        red = xi.min(axis, keepdims=keepdims)
    return var(red, cst(1), op="Equal")


@npxapi_inline
def amax(
    x: TensorType[ElemType.numerics, "T"],
    /,
    *,
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.amax`.
    """
    return var(x, op="ArgMax", axis=axis, keepdims=keepdims)


@npxapi_inline
def amin(
    x: TensorType[ElemType.numerics, "T"],
    /,
    *,
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.amin`.
    """
    return var(x, op="ArgMin", axis=axis, keepdims=keepdims)


@npxapi_inline
def any(
    x: TensorType[ElemType.bool_, "T"],
    /,
    *,
    axis: OptTensorType[ElemType.int64, "I"] = None,
    keepdims: ParType[int] = 0,
) -> TensorType[ElemType.bool_, "T"]:
    """
    See :func:`numpy.any`.
    """
    xi = var(x, op="Cast", to=TensorProto.INT64)

    if axis is None:
        new_shape = cst(np.array([-1], dtype=np.int64))
        xifl = var(xi, new_shape, op="Reshape")
        # in case xifl is empty, we need to add one element
        one = cst(np.array([0], dtype=np.int64))
        xifl1 = var(xifl, one, op="Concat", axis=0)
        red = xifl1.max(keepdims=keepdims)
    else:
        if isinstance(axis, int):
            axis = [axis]
        if isinstance(axis, (tuple, list)):
            axis = cst(np.array(axis, dtype=np.int64))
        red = xi.max(axis, keepdims=keepdims)
    return var(red, cst(1), op="Equal")


@npxapi_inline
def arange(
    start_or_stop: TensorType[
        {
            ElemType.int16,
            ElemType.int32,
            ElemType.int64,
            ElemType.float32,
            ElemType.float64,
        },
        "I",
        (1,),
    ],
    stop_or_step: OptTensorType[
        {
            ElemType.int16,
            ElemType.int32,
            ElemType.int64,
            ElemType.float32,
            ElemType.float64,
        },
        "I",
        (1,),
    ] = None,
    step: OptTensorType[
        {
            ElemType.int16,
            ElemType.int32,
            ElemType.int64,
            ElemType.float32,
            ElemType.float64,
        },
        "I",
        (1,),
    ] = None,
    dtype: OptParType[DType] = None,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arange`."
    if stop_or_step is None:
        v = var(
            cst(np.array(0, dtype=np.int64)),
            start_or_stop,
            cst(np.array(1, dtype=np.int64)),
            op="Range",
        )
    elif step is None:
        v = var(
            start_or_stop, stop_or_step, cst(np.array(1, dtype=np.int64)), op="Range"
        )
    else:
        v = var(start_or_stop, stop_or_step, step, op="Range")
    if dtype is not None:
        return var(v, op="Cast", to=dtype)
    return v


@npxapi_inline
def arccos(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arccos`."
    return var(x, op="Acos")


@npxapi_inline
def arccosh(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arccosh`."
    return var(x, op="Acosh")


@npxapi_inline
def argmax(
    x: TensorType[ElemType.numerics, "T"],
    /,
    *,
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.amax`.
    """
    return var(x, op="ArgMax", axis=axis, keepdims=keepdims)


@npxapi_inline
def argmin(
    x: TensorType[ElemType.numerics, "T"],
    /,
    *,
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op="ArgMin", axis=axis, keepdims=keepdims)


@npxapi_inline
def arcsin(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arcsin`."
    return var(x, op="Asin")


@npxapi_inline
def arcsinh(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arcsinh`."
    return var(x, op="Asinh")


@npxapi_inline
def arctan(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arctan`."
    return var(x, op="Atan")


@npxapi_inline
def arctanh(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arctanh`."
    return var(x, op="Atanh")


@npxapi_inline
def astype(
    a: TensorType[ElemType.numerics, "T1"],
    dtype: ParType[DType] = 1,
) -> TensorType[ElemType.numerics, "T2"]:
    """
    Cast an array.
    """
    if isinstance(dtype, Var):
        raise TypeError(
            f"dtype is an attribute, it cannot be a Variable of type {type(dtype)}."
        )
    if not isinstance(dtype, DType):
        if dtype is int:
            to = DType(TensorProto.INT64)
        elif dtype is float:
            to = DType(TensorProto.DOUBLE)
        elif dtype is bool:
            to = DType(TensorProto.BOOL)
        elif dtype is str:
            to = DType(TensorProto.STRING)
        else:
            raise TypeError(f"dtype must of type DType, not {type(dtype)}-{dtype}.")
    return var(a, op="Cast", to=to.code)


@npxapi_inline
def cdist(
    xa: TensorType[ElemType.numerics, "T"],
    xb: TensorType[ElemType.numerics, "T"],
    /,
    *,
    metric: OptParType[str] = "euclidean",
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`scipy.special.distance.cdist`.
    """
    return var(xa, xb, op=(FUNCTION_DOMAIN, "CDist"), metric=metric)


@npxapi_inline
def ceil(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.ceil`."
    return var(x, op="Ceil")


@npxapi_inline
def clip(
    x: TensorType[ElemType.numerics, "T"],
    /,
    a_min: TensorType[ElemType.numerics, "T"] = None,
    a_max: TensorType[ElemType.numerics, "T"] = None,
):
    "See :func:`numpy.clip`."
    args = [x]
    if a_min is not None:
        args.append(_cstv(a_min))
    else:
        args.append(None)
    if a_max is not None:
        args.append(_cstv(a_max))
    return var(*args, op="Clip")


@npxapi_inline
def compress(
    condition: TensorType[ElemType.bool_, "B"],
    x: TensorType[ElemType.numerics, "T"],
    /,
    *,
    axis: OptParType[int] = None,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.compress`.
    `np.compress(condition, x)` or `npnx.compress(x, condition)`.
    """
    if axis is None:
        return var(x, condition, op="Compress")
    return var(x, condition, op="Compress", axis=axis)


@npxapi_inline
def compute(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]],
    proto: ParType[Union[FunctionProto, ModelProto, NodeProto]] = None,
    name: ParType[str] = None,
) -> TupleType[TensorType[ElemType.numerics, "T"]]:
    """
    Executes an onnx proto.

    :param x: inputs
    :param proto: proto to execute
    :param name: model name
    :return: outputs
    """
    return var(*x, op=proto, name=name)


@npxapi_inline
def concat(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]], axis: ParType[int] = 0
) -> TensorType[ElemType.numerics, "T"]:
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    if len(x) <= 1:
        raise RuntimeError(f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op="Concat", axis=axis)


@npxapi_inline
def cos(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.cos`."
    return var(x, op="Cos")


@npxapi_inline
def cosh(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.cosh`."
    return var(x, op="Cosh")


@npxapi_inline
def cumsum(
    x: TensorType[ElemType.numerics, "T"],
    /,
    axis: OptTensorType[ElemType.int64, "I"] = None,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.cumsum`."
    if axis is None:
        m1 = cst(np.array([-1], dtype=np.int64))
        flat = var(x, m1, op="Reshape")
        axis = cst(np.array([0], dtype=np.int64))
        return var(flat, axis, op="CumSum")
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, (tuple, list)):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="CumSum")


@npxapi_inline
def det(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.linalg:det`."
    return var(x, op="Det")


@npxapi_inline
def dot(
    a: TensorType[ElemType.numerics, "T"], b: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.dot`
    dot is equivalent to `npx.matmul == np.matmul != np.dot`
    with arrays with more than 3D dimensions.
    """
    return var(a, b, op="MatMul")


@npxapi_inline
def einsum(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]], equation: ParType[str]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.einsum`."
    return var(*x, op="Einsum", equation=equation)


@npxapi_inline
def equal(
    x: TensorType[ElemType.allowed, "T"], y: TensorType[ElemType.allowed, "T"], /
) -> TensorType[ElemType.bool_, "T1"]:
    "See :func:`numpy.equal`."
    return var(x, y, op="Equal")


@npxapi_inline
def erf(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`scipy.special.erf`."
    return var(x, op="Erf")


@npxapi_inline
def exp(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.exp`."
    return var(x, op="Exp")


@npxapi_inline
def expand_dims(
    x: TensorType[ElemType.numerics, "T"], /, axis: TensorType[ElemType.int64, "I"]
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.expand_dims`.
    """
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="Unsqueeze")


@npxapi_inline
def expit(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`scipy.special.expit`."
    return var(x, op="Sigmoid")


@npxapi_inline
def eye(
    n_rows: TensorType[ElemType.int64, "I"],
    n_cols: TensorType[ElemType.int64, "I"],
    /,
    *,
    k: ParType[int] = 0,
    dtype: ParType[DType] = DType(TensorProto.DOUBLE),
):
    "See :func:`numpy.eye`."
    shape = cst(np.array([-1], dtype=np.int64))
    shape = var(
        var(n_rows, shape, op="Reshape"),
        var(n_cols, shape, op="Reshape"),
        axis=0,
        op="Concat",
    )
    zero = zeros(shape, dtype=dtype)
    res = var(zero, k=k, op="EyeLike")
    if dtype is not None:
        return var(res, to=dtype.code, op="Cast")
    return res


@npxapi_inline
def full(
    shape: TensorType[ElemType.int64, "I", (None,)],
    /,
    *,
    dtype: OptParType[DType] = None,
    fill_value: ParType[Scalar] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    """
    Implements :func:`numpy.full`.
    """
    if order != "C":
        raise RuntimeError(f"order={order!r} != 'C' not supported.")
    if fill_value is None:
        raise TypeError("fill_value cannot be None.")
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = DType(TensorProto.BOOL)
        elif isinstance(fill_value, int):
            dtype = DType(TensorProto.INT64)
        elif isinstance(fill_value, float):
            dtype = DType(TensorProto.DOUBLE)
        else:
            raise TypeError(
                f"Unexpected type {type(fill_value)} for fill_value={fill_value!r}."
            )
    if isinstance(fill_value, (float, int, bool)):
        value = make_tensor(
            name="cst", data_type=dtype.code, dims=[1], vals=[fill_value]
        )
    else:
        raise NotImplementedError(
            f"Unexpected type ({type(fill_value)} for fill_value={fill_value!r}."
        )
    return var(shape, value=value, op="ConstantOfShape")


@npxapi_inline
def full_like(
    x: TensorType[ElemType.allowed, "T"],
    /,
    *,
    fill_value: ParType[Scalar] = None,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    """
    Implements :func:`numpy.zeros`.
    """
    if order != "C":
        raise RuntimeError(f"order={order!r} != 'C' not supported.")
    if fill_value is None:
        raise TypeError("fill_value cannot be None.")
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = DType(TensorProto.BOOL)
        elif isinstance(fill_value, int):
            dtype = DType(TensorProto.INT64)
        elif isinstance(fill_value, float):
            dtype = DType(TensorProto.DOUBLE)
        else:
            raise TypeError(
                f"Unexpected type {type(fill_value)} for fill_value={fill_value!r} "
                f"and dtype={dtype!r}."
            )
    if isinstance(fill_value, (float, int, bool)):
        value = make_tensor(
            name="cst", data_type=dtype.code, dims=[1], vals=[fill_value]
        )
    else:
        raise NotImplementedError(
            f"Unexpected type ({type(fill_value)} for fill_value={fill_value!r}."
        )

    v = var(x.shape, value=value, op="ConstantOfShape")
    if dtype is None:
        return var(v, x, op="CastLike")
    return v


@npxapi_inline
def floor(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.floor`."
    return var(x, op="Floor")


@npxapi_inline
def hstack(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.hstack`."
    if len(x) <= 1:
        raise RuntimeError(f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op="Concat", axis=-1)


@npxapi_inline
def copy(x: TensorType[ElemType.allowed, "T"], /) -> TensorType[ElemType.allowed, "T"]:
    "Makes a copy."
    return var(x, op="Identity")


@npxapi_inline
def identity(
    *, n: ParType[int], dtype: OptParType[DType] = None
) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    model = var(
        cst(np.array([n, n], dtype=np.int64)),
        op="ConstantOfShape",
        value=from_array(np.array([0], dtype=np.int64)),
    )
    v = var(model, dtype=dtype, op="EyeLike")
    return v


@npxapi_no_inline
def isdtype(
    dtype: ParType[DType], kind: Union[DType, str, Tuple[Union[DType, str], ...]]
) -> bool:
    """
    See :epkg:`BaseArrayAPI:isdtype`.
    This function is not converted into an onnx graph.
    """
    if isinstance(dtype, DType):
        dti = tensor_dtype_to_np_dtype(dtype.code)
        return np_array_api.isdtype(dti, kind)
    if isinstance(dtype, int):
        raise TypeError(f"Unexpected type {type(dtype)}.")
    return np_array_api.isdtype(dtype, kind)


@npxapi_inline
def isfinite(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.bool_, "T1"]:
    "See :func:`numpy.isfinite`."
    return var(x, op="IsInf")


@npxapi_inline
def isinf(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.bool_, "T1"]:
    "See :func:`numpy.isnan`."
    return var(x, op="IsInf")


@npxapi_inline
def isnan(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.bool_, "T1"]:
    "See :func:`numpy.isnan`."
    return var(x, op="IsNaN")


@npxapi_inline
def linspace(
    start: TensorType[
        {
            ElemType.int16,
            ElemType.int32,
            ElemType.int64,
            ElemType.float32,
            ElemType.float64,
        },
        "T",
        (1,),
    ],
    stop: TensorType[
        {
            ElemType.int16,
            ElemType.int32,
            ElemType.int64,
            ElemType.float32,
            ElemType.float64,
        },
        "T1",
        (1,),
    ] = None,
    num: TensorType[
        {
            ElemType.int16,
            ElemType.int32,
            ElemType.int64,
        },
        "I",
        (1,),
    ] = None,
    dtype: OptParType[DType] = None,
    endpoint: ParType[int] = 1,
    # extend_shape: ParType[int] = 0,
) -> TensorType[
    {
        ElemType.int16,
        ElemType.int32,
        ElemType.int64,
        ElemType.float32,
        ElemType.float64,
    },
    "T2",
]:
    "See :func:`numpy.linspace`."
    zero = cst(np.array(0, dtype=np.int64))
    c1 = cst(np.array(1, dtype=np.int64))
    num_1 = var(num, c1, op="Sub") if endpoint else num
    num_p1 = var(num_1, c1, op="Add")
    steps = var(var(zero, num_p1, c1, op="Range"), op="Cast", to=TensorProto.DOUBLE)

    startc = var(start, op="Cast", to=TensorProto.DOUBLE)
    stopc = var(stop, op="Cast", to=TensorProto.DOUBLE)
    diff = var(stopc, startc, op="Sub")
    denom = var(
        var(num_1, op="Cast", to=TensorProto.DOUBLE),
        cst(np.array(1, dtype=np.float64)),
        op="Max",
    )
    div = var(diff, denom, op="Div")
    mul = var(steps, div, op="Mul")
    final = var(mul, startc, op="Add")

    if not endpoint:
        final = final[:-1]

    # shape
    shape_start = var(start, op="Shape")
    shape_zero = var(shape_start, zero, op="Greater")
    shape = var(shape_start, shape_zero, op="Compress")

    shape_full = var(final, op="Shape")
    new_shape = var(shape_full, shape, op="Concat", axis=0)
    last = var(final, new_shape, op="Reshape")

    if dtype is not None:
        return var(last, op="Cast", to=dtype)
    return var(last, start, op="CastLike")


@npxapi_inline
def log(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.log`."
    return var(x, op="Log")


@npxapi_inline
def log1p(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.log1p`."
    x1 = var(x, var(cst(np.array([1])), x, op="CastLike"), op="Add")
    return var(x1, op="Log")


@npxapi_inline
def matmul(
    a: TensorType[ElemType.numerics, "T"], b: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.matmul`."
    return var(a, b, op="MatMul")


@npxapi_inline
def ones(
    shape: TensorType[ElemType.int64, "I", (None,)],
    /,
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    """
    Implements :func:`numpy.zeros`.
    """
    if order != "C":
        raise RuntimeError(f"order={order!r} != 'C' not supported.")
    if dtype is None:
        dtype = DType(TensorProto.DOUBLE)
    return var(
        shape,
        value=make_tensor(name="one", data_type=dtype.code, dims=[1], vals=[1]),
        op="ConstantOfShape",
    )


@npxapi_inline
def ones_like(
    x: TensorType[ElemType.allowed, "T"],
    /,
    *,
    dtype: OptParType[DType] = None,
) -> TensorType[ElemType.numerics, "T"]:
    """
    Implements :func:`numpy.ones_like`.
    """
    o = make_tensor(
        name="one",
        data_type=TensorProto.INT64 if dtype is None else dtype.code,
        dims=[1],
        vals=[1],
    )
    v = var(x.shape, value=o, op="ConstantOfShape")
    if dtype is None:
        return var(v, x, op="CastLike")
    return v


@npxapi_inline
def pad(
    x: TensorType[ElemType.numerics, "T"],
    pads: TensorType[ElemType.int64, "I"],
    /,
    constant_value: OptTensorType[ElemType.numerics, "T"] = None,
    axes: OptTensorType[ElemType.int64, "I"] = None,
    mode: ParType[str] = "constant",
):
    """
    It does not implement :func:`numpy.pad` but the ONNX version.
    """
    if constant_value is None:
        if axes is None:
            return var(x, pads, op="Pad", mode=mode)
        return var(x, pads, None, axes, op="Pad", mode=mode)
    if axes is None:
        return var(x, pads, constant_value, op="Pad", mode=mode)
    return var(x, pads, constant_value, axes, op="Pad", mode=mode)


@npxapi_inline
def reciprocal(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.reciprocal`."
    return var(x, op="Reciprocal")


@npxapi_inline
def relu(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "relu"
    return var(x, op="Relu")


@npxapi_inline
def reshape(
    x: TensorType[ElemType.numerics, "T"],
    shape: TensorType[ElemType.int64, "I", (None,)],
    /,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.reshape`.

    .. warning::

        Numpy definition is tricky because onnxruntime does not handle well
        dimensions with an undefined number of dimensions.
        However the array API defines a more stricly signature for
        `reshape <https://data-apis.org/array-api/2022.12/
        API_specification/generated/array_api.reshape.html>`_.
        :epkg:`scikit-learn` updated its code to follow the Array API in
        `PR 26030 ENH Forces shape to be tuple when using Array API's reshape
        <https://github.com/scikit-learn/scikit-learn/pull/26030>`_.
    """
    if isinstance(shape, int):
        shape = cst(np.array([shape], dtype=np.int64))
        return var(x, shape, op="Reshape")
    if isinstance(shape, tuple) and len(shape) == 0:
        shape = cst(np.array([-1], dtype=np.int64))
        return var(x, shape, op="Reshape")
    shape_reshaped = var(shape, cst(np.array([-1], dtype=np.int64)), op="Reshape")
    return var(x, shape_reshaped, op="Reshape")


@npxapi_inline
def round(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.round`."
    return var(x, op="Round")


@npxapi_inline
def sigmoid(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`scipy.special.expit`."
    return var(x, op="Sigmoid")


@npxapi_inline
def sign(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sign`."
    return var(x, op="Sign")


@npxapi_inline
def sin(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sin`."
    return var(x, op="Sin")


@npxapi_inline
def sinh(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sinh`."
    return var(x, op="Sinh")


@npxapi_inline
def sqrt(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sqrt`."
    return var(x, op="Sqrt")


@npxapi_inline
def squeeze(
    x: TensorType[ElemType.numerics, "T"],
    /,
    axis: OptTensorType[ElemType.int64, "I"] = None,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.squeeze`."
    if axis is None:
        shape = x.shape
        zero = cst(np.array([0], dtype=np.int64))
        one = cst(np.array([1], dtype=np.int64))
        ind = var(zero, shape.shape, one, op="Range")
        axis = var(ind, shape == one, op="Compress")
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, (tuple, list)):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="Squeeze")


@npxapi_inline
def sum(
    x: TensorType[ElemType.numerics, "T"],
    /,
    axis: OptTensorType[ElemType.int64, "I"] = None,
    *,
    dtype: OptParType[DType] = None,
    keepdims: ParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sum`."
    if axis is None:
        m1 = cst(np.array([-1], dtype=np.int64))
        flat = var(x, m1, op="Reshape")
        axis = cst(np.array([0], dtype=np.int64))
        res = var(flat, axis, op="ReduceSum", keepdims=keepdims)
    else:
        if isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, (tuple, list)):
            axis = cst(np.array(axis, dtype=np.int64))
        res = var(x, axis, op="Sum", keepdims=keepdims)
    if dtype is None:
        return res
    return var(res, op="Cast", to=dtype.code)


@npxapi_inline
def take(
    data: TensorType[ElemType.numerics, "T"],
    indices: TensorType[ElemType.int64, "I"],
    /,
    *,
    axis: ParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.take`."
    return var(data, indices, op="Gather", axis=axis)


@npxapi_inline
def tan(x: TensorType[ElemType.numerics, "T"], /) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.tan`."
    return var(x, op="Tan")


@npxapi_inline
def tanh(
    x: TensorType[ElemType.numerics, "T"], /
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.tanh`."
    return var(x, op="Tanh")


@npxapi_inline
def topk(
    x: TensorType[ElemType.numerics, "T"],
    k: TensorType[ElemType.int64, "I", (1,)],
    /,
    *,
    axis: OptParType[int] = -1,
    largest: OptParType[int] = 1,
    sorted: OptParType[int] = 1,
) -> TupleType[TensorType[ElemType.numerics, "T"], TensorType[ElemType.int64, "I"]]:
    "See :func:`numpy.argsort`."
    return make_tuple(2, x, k, op="TopK", axis=axis, largest=largest, sorted=sorted)


@npxapi_inline
def transpose(
    x: TensorType[ElemType.numerics, "T"], /, *, perm: ParType[Tuple[int, ...]] = (1, 0)
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.transpose`."
    return var(x, op="Transpose", perm=list(perm))


@npxapi_inline
def unsqueeze(
    x: TensorType[ElemType.numerics, "T"], axis: TensorType[ElemType.int64, "I"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.expand_dims`."
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="Unsqueeze")


@npxapi_inline
def vstack(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.vstack`."
    if len(x) <= 1:
        raise RuntimeError(f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op="Concat", axis=0)


@npxapi_inline
def where(
    cond: TensorType[ElemType.bool_, "B"],
    x: TensorType[ElemType.numerics, "T"],
    y: TensorType[ElemType.numerics, "T"],
    /,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.where`."
    return var(cond, x, y, op="Where")


@npxapi_inline
def zeros(
    shape: TensorType[ElemType.int64, "I", (None,)],
    /,
    *,
    dtype: OptParType[DType] = None,
    order: OptParType[str] = "C",
) -> TensorType[ElemType.numerics, "T"]:
    """
    Implements :func:`numpy.zeros`.
    """
    if order != "C":
        raise RuntimeError(f"order={order!r} != 'C' not supported.")
    if dtype is None:
        dtype = DType(TensorProto.DOUBLE)
    return var(
        shape,
        value=make_tensor(name="zero", data_type=dtype.code, dims=[1], vals=[0]),
        op="ConstantOfShape",
    )


@npxapi_inline
def zeros_like(
    x: TensorType[ElemType.allowed, "T"],
    /,
    *,
    dtype: OptParType[DType] = None,
) -> TensorType[ElemType.numerics, "T"]:
    """
    Implements :func:`numpy.zeros_like`.
    """
    o = make_tensor(
        name="zero",
        data_type=TensorProto.INT64 if dtype is None else dtype.code,
        dims=[1],
        vals=[0],
    )
    v = var(x.shape, value=o, op="ConstantOfShape")
    if dtype is None:
        return var(v, x, op="CastLike")
    return v
