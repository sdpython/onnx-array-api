import numpy as np

from onnx.reference.op_run import OpRun


class ConstantOfShape(OpRun):
    @staticmethod
    def _process(value):
        cst = value[0] if isinstance(value, np.ndarray) else value
        if isinstance(cst, int):
            cst = np.int64(cst)
        elif isinstance(cst, float):
            cst = np.float64(cst)
        elif cst is None:
            cst = np.float32(0)
        if not isinstance(
            cst,
            (
                np.float16,
                np.float32,
                np.float64,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
                np.uint64,
                np.uint32,
                np.uint16,
                np.uint8,
                np.bool_,
            ),
        ):
            raise TypeError(f"value must be a real not {type(cst)}")

    def _run(self, data, value=None):
        cst = self._process(value)
        try:
            res = np.full(tuple(data), cst)
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape "
                f"{data!r} with value {cst!r} "
                f"(raw value={value!r})."
            ) from e
        return (res,)
