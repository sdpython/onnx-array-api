import numpy as np
from onnx.reference.op_run import OpRun


class FusedMatMul(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        A,
        B,
        alpha: float = 1,
        transA: int = 0,
        transB: int = 0,
        transBatchA: int = 0,
        transBatchB: int = 0,
    ):
        assert (
            transBatchA == 0
        ), f"Not implemented for transBatchA==1 and {A.shape}x{B.shape}"
        assert (
            transBatchB == 0
        ), f"Not implemented for transBatchB==1 and {A.shape}x{B.shape}"
        if transA:
            perm = list(range(len(A.shape)))
            dim = len(perm)
            perm[dim - 2], perm[dim - 1] = perm[dim - 1], perm[dim - 2]
            A = np.transpose(A, perm)
        if transB:
            perm = list(range(len(B.shape)))
            dim = len(perm)
            perm[dim - 2], perm[dim - 1] = perm[dim - 1], perm[dim - 2]
            B = np.transpose(B, perm)
        a = np.array(alpha, dtype=A.dtype)
        return (np.matmul(A, B) * a,)
