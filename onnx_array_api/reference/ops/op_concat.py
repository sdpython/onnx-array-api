import numpy as np

from onnx.reference.op_run import OpRun


class Concat(OpRun):
    def _preprocess(self, a: np.ndarray, axis: int) -> np.ndarray:
        if axis >= len(a.shape):  # type: ignore
            new_shape = a.shape + (1,) * (axis + 1 - len(a.shape))  # type: ignore
            return a.reshape(new_shape)
        return a

    def _run(self, *args, axis=None):  # type: ignore
        targs = tuple(self._preprocess(a, axis) for a in args)
        return (np.concatenate(targs, axis),)  # type: ignore
