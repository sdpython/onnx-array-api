from onnx.reference.op_run import OpRun


class MemcpyFromHost(OpRun):
    def _run(self, x):
        return (x,)


class MemcpyToHost(OpRun):
    def _run(self, x):
        return (x,)
