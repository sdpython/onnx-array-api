import numpy as np

from onnx.reference.op_run import OpRun


def scatter_elements(data, indices, updates, axis=0, reduction=None):  # type: ignore
    if reduction == "add":

        def f(x, y):
            return x + y

    elif reduction == "min":

        def f(x, y):
            return min(x, y)

    elif reduction == "max":

        def f(x, y):
            return max(x, y)

    else:

        def f(x, y):
            return y

    if axis < 0:
        axis = data.ndim + axis

    if len(data.shape) == 1 and axis == 0:
        scattered = np.copy(data)
        for pos, up in zip(indices, updates):
            scattered[pos] = f(scattered[pos], up)
        return scattered

    if len(indices.shape) == 2:
        scattered = np.copy(data)
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    scattered[indices[i, j], j] = f(
                        scattered[indices[i, j], j], updates[i, j]
                    )
        else:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    scattered[i, indices[i, j]] = f(
                        scattered[i, indices[i, j]], updates[i, j]
                    )
        return scattered

    if len(indices.shape) == 3:
        scattered = np.copy(data)
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[indices[i, j, k], j, k] = f(
                            scattered[indices[i, j, k], j, k], updates[i, j, k]
                        )
        elif axis == 1:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[i, indices[i, j, k], k] = f(
                            scattered[i, indices[i, j, k], k], updates[i, j, k]
                        )
        elif axis == 2:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[i, j, indices[i, j, k]] = f(
                            scattered[i, j, indices[i, j, k]], updates[i, j, k]
                        )
        return scattered

    if len(indices.shape) == 4:
        scattered = np.copy(data)
        if axis == 3:
            for a in range(indices.shape[0]):
                for i in range(indices.shape[1]):
                    for j in range(indices.shape[2]):
                        for k in range(indices.shape[3]):
                            scattered[a, i, j, indices[a, i, j, k]] = f(
                                scattered[a, i, j, indices[a, i, j, k]],
                                updates[a, i, j, k],
                            )
            return scattered

    raise RuntimeError(
        f"Not implemented for indices.shape={indices.shape} and axis={axis}"
    )


class ScatterElements(OpRun):
    def _run(self, data, indices, updates, axis=None, reduction=None):  # type: ignore
        res = scatter_elements(data, indices, updates, axis=axis, reduction=reduction)
        return (res,)
