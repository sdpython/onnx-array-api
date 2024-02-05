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
            dim = len(A.shape)
            A = A.transpose(axes=(dim - 2, dim - 1))
        if transB:
            dim = len(B.shape)
            B = B.transpose(axes=(dim - 2, dim - 1))
        a = np.array(alpha, dtype=A.dtype)
        return (A @ B * a,)
