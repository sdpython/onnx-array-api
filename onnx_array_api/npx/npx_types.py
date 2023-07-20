from typing import Any, Tuple, Union

import numpy as np
from onnx import AttributeProto, TensorProto
from onnx.helper import tensor_dtype_to_np_dtype
from .._helpers import np_dtype_to_tensor_dtype


class WrapperType:
    """
    WrapperType.
    """

    pass


class DType(WrapperType):
    """
    Type of the element type returned by tensors
    following the :epkg:`Array API`.

    :param code: element type based on onnx definition,
        if str, it looks into class :class:`onnx.TensorProto`
        to retrieve the code
    """

    __slots__ = ["code_"]

    def __init__(self, code: Union[int, str]):
        if isinstance(code, str):
            self.code_ = getattr(TensorProto, code)
        elif isinstance(code, int):
            self.code_ = code
        else:
            raise TypeError(f"Unsupported type {type(code)}:{code!r}")

    def __repr__(self) -> str:
        "usual"
        return f"DType({self.code_})"

    def __str__(self) -> str:
        "usual"
        return f"DT{self.code_}"

    def __hash__(self) -> int:
        return self.code_

    @property
    def code(self) -> int:
        return self.code_

    @property
    def np_dtype(self) -> "np.dtype":
        return tensor_dtype_to_np_dtype(self.code_)

    def __eq__(self, dt: "DType") -> bool:
        "Compares two types."
        if dt is None:
            return False
        if dt.__class__ is DType:
            return self.code_ == dt.code_
        if isinstance(dt, (int, bool, str, float)):
            return False
        if dt is int:
            return self.code_ == TensorProto.INT64
        if dt is str:
            return self.code_ == TensorProto.STRING
        if dt is bool:
            return self.code_ == TensorProto.BOOL
        if dt is float:
            return self.code_ == TensorProto.DOUBLE
        if isinstance(dt, list):
            return False
        if dt in ElemType.numpy_map:
            dti = ElemType.numpy_map[dt]
            return self.code_ == dti.code_
        if isinstance(dt, type) and issubclass(dt, ElemType):
            return self.code_ == dt.dtype.code_
        try:
            dti = np_dtype_to_tensor_dtype(dt)
        except KeyError:
            raise TypeError(f"dt must be DType not {type(dt)} - {dt!r}.")
        return self.code_ == dti

    def __lt__(self, dt: "DType") -> bool:
        "Compares two types."
        if dt.__class__ is DType:
            return self.code_ < dt.code_
        if isinstance(dt, int):
            raise TypeError(f"dt must be DType not {type(dt)}.")
        try:
            dti = np_dtype_to_tensor_dtype(dt)
        except KeyError:
            raise TypeError(f"dt must be DType not {type(dt)} - {dt}.")
        return self.code_ < dti

    @classmethod
    def type_name(cls) -> str:
        "Returns its full name."
        raise NotImplementedError()


class _DType2(DType):
    "Wraps a type into a different type."
    pass


class _DTypes(DType):
    "Wraps a type into a different type."
    pass


class ElemTypeCstInner(WrapperType):
    """
    Defines all possible types and tensor element type.
    """

    __slots__ = []

    undefined = DType(TensorProto.UNDEFINED)  # 0
    bool_ = DType(TensorProto.BOOL)  # 9
    int8 = DType(TensorProto.INT8)  # 3
    int16 = DType(TensorProto.INT16)  # 5
    int32 = DType(TensorProto.INT32)  # 6
    int64 = DType(TensorProto.INT64)  # 7
    uint8 = DType(TensorProto.UINT8)  # 2
    uint16 = DType(TensorProto.UINT16)  # 4
    uint32 = DType(TensorProto.UINT32)  # 12
    uint64 = DType(TensorProto.UINT64)  # 13
    float16 = DType(TensorProto.FLOAT16)  # 10
    float32 = DType(TensorProto.FLOAT)  # 1
    float64 = DType(TensorProto.DOUBLE)  # 11
    bfloat16 = DType(TensorProto.BFLOAT16)  # 16
    complex64 = DType(TensorProto.COMPLEX64)  # 14
    complex128 = DType(TensorProto.COMPLEX128)  # 15
    str_ = DType(TensorProto.STRING)  # 8


class ElemTypeCstSet(ElemTypeCstInner):
    """
    Sets of element types.
    """

    allowed = set(DType(i) for i in range(1, 17))

    ints = {
        ElemTypeCstInner.int8,
        ElemTypeCstInner.int16,
        ElemTypeCstInner.int32,
        ElemTypeCstInner.int64,
        ElemTypeCstInner.uint8,
        ElemTypeCstInner.uint16,
        ElemTypeCstInner.uint32,
        ElemTypeCstInner.uint64,
    }

    floats = {
        ElemTypeCstInner.float16,
        ElemTypeCstInner.bfloat16,
        ElemTypeCstInner.float32,
        ElemTypeCstInner.float64,
    }

    numerics = {
        ElemTypeCstInner.int8,
        ElemTypeCstInner.int16,
        ElemTypeCstInner.int32,
        ElemTypeCstInner.int64,
        ElemTypeCstInner.uint8,
        ElemTypeCstInner.uint16,
        ElemTypeCstInner.uint32,
        ElemTypeCstInner.uint64,
        ElemTypeCstInner.float16,
        ElemTypeCstInner.bfloat16,
        ElemTypeCstInner.float32,
        ElemTypeCstInner.float64,
    }

    strings = {ElemTypeCstInner.str_}

    @staticmethod
    def combined(type_set):
        "Combines all types into a single integer by using power of 2."
        s = 0
        for dt in type_set:
            s += 1 << dt.code
        return _DTypes(s)


class ElemTypeCst(ElemTypeCstSet):
    """
    Combination of element types.
    """

    Undefined = _DType2(0)
    Bool = _DType2(1 << ElemTypeCstInner.bool_.code)
    Int8 = _DType2(1 << ElemTypeCstInner.int8.code)
    Int16 = _DType2(1 << ElemTypeCstInner.int16.code)
    Int32 = _DType2(1 << ElemTypeCstInner.int32.code)
    Int64 = _DType2(1 << ElemTypeCstInner.int64.code)
    UInt8 = _DType2(1 << ElemTypeCstInner.uint8.code)
    UInt16 = _DType2(1 << ElemTypeCstInner.uint16.code)
    UInt32 = _DType2(1 << ElemTypeCstInner.uint32.code)
    UInt64 = _DType2(1 << ElemTypeCstInner.uint64.code)
    BFloat16 = _DType2(1 << ElemTypeCstInner.bfloat16.code)
    Float16 = _DType2(1 << ElemTypeCstInner.float16.code)
    Float32 = _DType2(1 << ElemTypeCstInner.float32.code)
    Float64 = _DType2(1 << ElemTypeCstInner.float64.code)
    Complex64 = _DType2(1 << ElemTypeCstInner.complex64.code)
    Complex128 = _DType2(1 << ElemTypeCstInner.complex128.code)
    String = _DType2(1 << ElemTypeCstInner.str_.code)

    Numerics = ElemTypeCstSet.combined(ElemTypeCstSet.numerics)
    Floats = ElemTypeCstSet.combined(ElemTypeCstSet.floats)
    Ints = ElemTypeCstSet.combined(ElemTypeCstSet.ints)
    Strings = ElemTypeCstSet.combined(ElemTypeCstSet.strings)


class ElemType(ElemTypeCst):
    """
    Allowed element type based on numpy dtypes.

    :param dtype: DType or a string
    """

    names_int = {
        att: getattr(ElemTypeCstInner, att)
        for att in dir(ElemTypeCstInner)
        if isinstance(getattr(ElemTypeCstInner, att), DType)
    }

    int_names = {
        getattr(ElemTypeCstInner, att): att
        for att in dir(ElemTypeCstInner)
        if isinstance(getattr(ElemTypeCstInner, att), DType)
    }

    set_names = {
        getattr(ElemTypeCst, att): att
        for att in dir(ElemTypeCst)
        if isinstance(getattr(ElemTypeCst, att), int) and "A" <= att[0] <= "Z"
    }

    numpy_map = {
        **{
            getattr(np, att): getattr(ElemTypeCst, att)
            for att in dir(ElemTypeCst)
            if isinstance(getattr(ElemTypeCst, att), DType) and hasattr(np, att)
        },
        **{
            np.dtype(att): getattr(ElemTypeCst, att)
            for att in dir(ElemTypeCst)
            if isinstance(getattr(ElemTypeCst, att), DType) and hasattr(np, att)
        },
    }

    __slots__ = ["dtype"]

    @classmethod
    def __class_getitem__(cls, dtype: Union[str, DType]):
        """
        Returns a subclass of this one with attribute `dtype`.
        """
        if isinstance(dtype, str):
            dtype = ElemType.names_int[dtype]
        elif dtype in ElemType.numpy_map:
            dtype = ElemType.numpy_map[dtype]
        elif dtype == DType(0):
            pass
        elif dtype not in ElemType.allowed:
            raise ValueError(f"Unexpected dtype {dtype} not in {ElemType.allowed}.")
        newt = type(f"{cls.__name__}{dtype}", (cls,), dict(dtype=dtype))
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        return newt

    def __eq__(self, t):
        "Compares types."
        return self.dtype == t.dtype

    @classmethod
    def type_name(cls) -> str:
        "Returns its fullname."
        s = ElemType.int_names[cls.dtype]
        return s

    @classmethod
    def get_set_name(cls, dtypes):
        "Returns the set name."
        tt = []
        for dt in dtypes:
            if isinstance(dt, int):
                tt.append(dt)
            else:
                tt.append(dt.dtype)
        dtypes = set(tt)
        for d in dir(cls):
            att = getattr(cls, d)
            if not isinstance(att, set):
                continue
            if dtypes == att:
                return d
        return None


class Scalar:
    """
    Defines a scalar.
    """

    def __init__(self, value: Union[float, int, bool]):
        self.value = value

    def __repr__(self):
        "usual"
        return f"Scalar({self.value!r})"


class ParType(WrapperType):
    """
    Defines a parameter type.

    :param dtype: parameter type
    :param optional: is optional or not
    """

    map_names = {
        int: "int",
        float: "float",
        str: "str",
        DType: "DType",
        bool: "bool",
        Scalar: "Scalar",
    }

    @classmethod
    def __class_getitem__(cls, dtype):
        if isinstance(dtype, (int, float, bool)):
            msg = str(dtype)
        else:
            msg = getattr(dtype, "__name__", str(dtype))
        newt = type(f"{cls.__name__}{msg}", (cls,), dict(dtype=dtype))
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        return newt

    @classmethod
    def type_name(cls) -> str:
        "Returns its full name."
        if cls.dtype in ParType.map_names:
            newt = f"ParType[{ParType.map_names[cls.dtype]}]"
        else:
            newt = f"ParType[{cls.dtype}]"
        if "<" in newt or "{" in newt:
            raise NameError(f"Name is wrong {newt!r}.")
        return newt

    @classmethod
    def onnx_type(cls):
        "Returns the onnx corresponding type."
        if cls.dtype == int:
            return AttributeProto.INT
        if cls.dtype == float:
            return AttributeProto.FLOAT
        if cls.dtype == bool:
            return AttributeProto.BOOL
        if cls.dtype == str:
            return AttributeProto.STRING
        raise RuntimeError(
            f"Unsupported attribute type {cls.dtype!r} " f"for parameter {cls!r}."
        )


class OptParType(ParType):
    """
    Defines an optional parameter type.

    :param dtype: parameter type
    """

    @classmethod
    def __class_getitem__(cls, dtype):
        if isinstance(dtype, (int, float, bool)):
            msg = str(dtype)
        else:
            msg = dtype.__name__
        newt = type(f"{cls.__name__}{msg}", (cls,), dict(dtype=dtype))
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        return newt

    @classmethod
    def type_name(cls) -> str:
        "Returns its full name."
        newt = f"OptParType[{ParType.map_names[cls.dtype]}]"
        if "<" in newt or "{" in newt:
            raise NameError(f"Name is wrong {newt!r}.")
        return newt


class ShapeType(Tuple[int, ...]):
    """
    Defines a shape type.
    """

    @classmethod
    def __class_getitem__(cls, *args):
        if any(map(lambda t: t is not None and not isinstance(t, (int, str)), args)):
            raise TypeError(
                f"Unexpected value for args={args}, every element should int or str."
            )
        ext = "_".join(map(str, args))
        newt = type(f"{cls.__name__}{ext}", (cls,), dict(shape=args))
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        return newt

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}[{self.shape}]"

    def __str__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}[{self.shape}]"


class TensorType(WrapperType):
    """
    Used to annotate functions.

    :param dtypes: tuple of :class:`ElemType`
    :param shape: tuple of integer or strings or None
    :param name: name of the type
    """

    main_name = "TensorType"

    @classmethod
    def __class_getitem__(cls, *args):
        """
        Returns a subclass of this one with two attributes `dtypes`
        and `shape`.
        """
        if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        name = None
        dtypes = None
        shape = None
        for a in args:
            if isinstance(a, str):
                if hasattr(ElemType, a):
                    if dtypes is not None:
                        raise TypeError(f"Unexpected type {type(a)} in {args}.")
                    v = getattr(ElemType, a)
                    dtypes = tuple(v) if isinstance(v, set) else (v,)
                else:
                    name = a
                continue
            if isinstance(a, set):
                dtypes = tuple(a)
                continue
            if isinstance(a, tuple):
                shape = a
                continue
            if isinstance(a, DType):
                if dtypes is not None:
                    raise TypeError(f"Unexpected type {type(a)} in {args}.")
                dtypes = (a,)
                continue
            if a is None:
                continue
            if a in ElemType.numpy_map:
                if dtypes is not None:
                    raise TypeError(f"Unexpected type {type(a)} in {args}.")
                dtypes = (ElemType.numpy_map[a],)
                continue
            raise TypeError(f"Unexpected type {type(a)} in {args}.")

        if isinstance(dtypes, ElemType):
            dtypes = (dtypes,)
        elif (
            isinstance(dtypes, str)
            or dtypes in ElemType.allowed
            or dtypes in ElemType.numpy_map
        ):
            dtypes = (ElemType[dtypes],)
        if not isinstance(dtypes, tuple):
            raise TypeError(f"dtypes must be a tuple not {type(dtypes)}, args={args}.")
        check = []
        for dt in dtypes:
            if isinstance(dt, ElemType):
                check.append(dt)
            elif dt in ElemType.allowed:
                check.append(ElemType[dt])
            elif isinstance(dt, DType):
                check.append(ElemType[dt])
            else:
                raise TypeError(f"Unexpected type {type(dt)} in {dtypes}, args={args}.")

        dtypes = tuple(check)
        if isinstance(shape, int):
            shape = (shape,)
        msg = []
        if name:
            msg.append(name)
        if dtypes is not None:
            msg.append("_".join(map(lambda t: str(t.dtype), dtypes)))
        if shape is not None:
            msg.append("_".join(map(str, shape)))
        final = "__".join(msg)
        if final:
            final = "_" + final
        newt = type(
            f"{cls.__name__}{final}",
            (cls,),
            dict(name=name, dtypes=dtypes, shape=shape),
        )
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        if newt.name is None:
            raise RuntimeError(
                f"A constraint needs a name but none is given: args={args}."
            )
        return newt

    @classmethod
    def supports_dtype(cls, dtype: DType) -> bool:
        """
        Determines if the element type `dtype`
        is within `dtypes`.
        """
        return dtype in cls.dtypes

    @classmethod
    def type_name(cls) -> str:
        "Returns its full name."
        set_name = ElemType.get_set_name(cls.dtypes)
        if not set_name:
            st = (
                cls.dtypes[0].type_name()
                if len(cls.dtypes) == 1
                else set(t.type_name() for t in cls.dtypes)
            )
            set_name = repr(st)
        if cls.shape:
            if cls.name:
                newt = f"{cls.main_name}[{set_name}, {cls.shape!r}, {cls.name!r}]"
            else:
                newt = f"{cls.main_name}[{set_name}, {cls.shape!r}]"
        elif cls.name:
            newt = f"{cls.main_name}[{set_name}, {cls.name!r}]"
        else:
            newt = f"{cls.main_name}[{set_name}]"
        if "<" in newt:
            raise NameError(f"Name is wrong {newt!r}.")
        return newt

    def _name_set(self):
        s = 0
        for dt in self.dtypes:
            s += 1 << dt.dtype
        try:
            return ElemType.set_names[s]
        except KeyError:
            raise RuntimeError(
                f"Unable to guess element type name for {s}: "
                f"{repr(self)} in {ElemType.set_names}."
            )

    @classmethod
    def issuperset(cls, tensor_type: type) -> bool:
        """
        Tells if *cls* is a superset of *tensor_type*.
        """
        set1 = set(t.dtype for t in cls.dtypes)
        set2 = set(t.dtype for t in tensor_type.dtypes)
        if not set1.issuperset(set2):
            return False
        if cls.shape is None:
            return True
        if tensor_type.shape is None:
            return False
        if len(cls.shape) != len(tensor_type.shape):
            return False
        for a, b in zip(cls.shape, tensor_type.shape):
            if isinstance(a, int):
                if a != b:
                    return False
        return True


class OptTensorType(TensorType):
    """
    Defines an optional tensor type.

    :param dtype: element type
    """

    main_name = "OptTensorType"


class SequenceType(WrapperType):
    """
    Defines a sequence of tensors.
    """

    @classmethod
    def __class_getitem__(cls, elem_type: Any, *args) -> "SequenceType":
        name = None
        if len(args) == 1:
            name = args[0]
        elif len(args) > 1:
            raise ValueError(f"Unexected value {args}.")
        if name:
            newt = type(
                f"{cls.__name__}_{name}_{elem_type.__name__}",
                (cls,),
                dict(name=name, elem_type=elem_type),
            )
        else:
            newt = type(
                f"{cls.__name__}{elem_type.__name__}",
                (cls,),
                dict(name=name, elem_type=elem_type),
            )
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        return newt

    @classmethod
    def type_name(cls) -> str:
        "Returns its full name."
        if cls.name:
            newt = f"SequenceType[{cls.elem_type.type_name()}], {cls.name!r})"
        else:
            newt = f"SequenceType[{cls.elem_type.type_name()!r}]"
        if "<" in newt or "{" in newt:
            raise NameError(f"Name is wrong {newt!r}.")
        return newt


class TupleType(WrapperType):
    """
    Defines a sequence of tensors.
    """

    @classmethod
    def __class_getitem__(cls, *args) -> "TupleType":
        if len(args) == 1 and isinstance(args[0], int):
            return cls.elem_types[args[0]]
        if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        name = None
        elem_types = []
        for a in args:
            if isinstance(a, str):
                name = a
            elif isinstance(a, type) and issubclass(a, TensorType):
                elem_types.append(a)
            elif a in (int, float, str):
                elem_types.append(a)
            else:
                raise TypeError(
                    f"Unexpected value type={type(a)}, value={a} in {args}."
                )
        msg = []
        if name:
            msg.append(name)
        for t in elem_types:
            msg.append(t.__name__)
        final = "_".join(msg)
        newt = type(
            f"{cls.__name__}_{final}",
            (cls,),
            dict(name=name, elem_types=tuple(elem_types)),
        )
        if "<" in newt.__name__:
            raise NameError(f"Name is wrong {newt.__name__!r}.")
        return newt

    @classmethod
    def len(cls):
        "Returns the number of types."
        return len(cls.elem_types)

    @classmethod
    def type_name(cls) -> str:
        "Returns its full name."
        dts = ", ".join(map(lambda s: s.type_name(), cls.elem_types))
        if cls.name:
            newt = f"TupleType[{dts}, {cls.name!r}]"
        else:
            newt = f"TupleType[{dts}]"
        if "<" in newt or "{" in newt:
            raise NameError(f"Name is wrong {newt!r}.")
        return newt


def _make_type(name: str, elem_type: int):
    def class_getitem(cls, shape: Union[int, ShapeType]) -> TensorType:
        if isinstance(shape, int):
            shape = (shape,)
        return TensorType[elem_type, shape, f"mtx{elem_type}"]

    new_type = type(name, tuple(), {})
    new_type.__class_getitem__ = classmethod(class_getitem)
    return new_type


Bool = _make_type("Bool", ElemType.bool_)

BFloat16 = _make_type("BFloat16", ElemType.bfloat16)
Float16 = _make_type("Float16", ElemType.float16)
Float32 = _make_type("Float32", ElemType.float32)
Float64 = _make_type("Float32", ElemType.float64)

Int8 = _make_type("int8", ElemType.int8)
Int16 = _make_type("int16", ElemType.int16)
Int32 = _make_type("int32", ElemType.int32)
Int64 = _make_type("int64", ElemType.int64)

UInt8 = _make_type("uint8", ElemType.uint8)
UInt16 = _make_type("uint16", ElemType.uint16)
UInt32 = _make_type("uint32", ElemType.uint32)
UInt64 = _make_type("uint64", ElemType.uint64)
