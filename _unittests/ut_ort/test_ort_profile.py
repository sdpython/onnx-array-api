import unittest
import os
import numpy as np
from pandas import DataFrame, read_excel
from onnx_array_api.npx import absolute, jit_onnx
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.ort.ort_optimizers import ort_optimized_model
from onnx_array_api.ort.ort_profile import ort_profile, merge_ort_profile
from onnxruntime.capi._pybind_state import (
    OrtValue as C_OrtValue,
    OrtDevice as C_OrtDevice,
)


class TestOrtProfile(ExtTestCase):
    def test_ort_profile(self):
        def l1_loss(x, y):
            return absolute(x - y).sum()

        def l2_loss(x, y):
            return ((x - y) ** 2).sum()

        def myloss(x, y):
            return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])

        jitted_myloss = jit_onnx(myloss)
        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)
        jitted_myloss(x, y)
        onx = jitted_myloss.get_onnx()
        feeds = {"x0": x, "x1": y}
        self.assertRaise(lambda: ort_optimized_model(onx, "NO"), ValueError)
        optimized = ort_optimized_model(onx)
        prof = ort_profile(optimized, feeds)
        self.assertIsInstance(prof, DataFrame)
        prof = ort_profile(optimized, feeds, as_df=False)
        self.assertIsInstance(prof, list)

    def test_ort_profile_ort_value(self):
        def to_ort_value(m):
            device = C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
            ort_value = C_OrtValue.ortvalue_from_numpy(m, device)
            return ort_value

        def l1_loss(x, y):
            return absolute(x - y).sum()

        def l2_loss(x, y):
            return ((x - y) ** 2).sum()

        def myloss(x, y):
            return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])

        jitted_myloss = jit_onnx(myloss)
        x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)
        jitted_myloss(x, y)
        onx = jitted_myloss.get_onnx()
        np_feeds = {"x0": x, "x1": y}
        feeds = {k: to_ort_value(v) for k, v in np_feeds.items()}

        self.assertRaise(lambda: ort_optimized_model(onx, "NO"), ValueError)
        optimized = ort_optimized_model(onx)
        prof = ort_profile(optimized, feeds)
        self.assertIsInstance(prof, DataFrame)
        prof = ort_profile(optimized, feeds, as_df=False)
        self.assertIsInstance(prof, list)

    def test_merge_ort_profile(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        df1 = read_excel(os.path.join(data, "prof_base.xlsx"))
        df2 = read_excel(os.path.join(data, "prof_opti.xlsx"))
        merged, gr = merge_ort_profile(df1, df2)
        self.assertEqual(merged.shape, (23, 9))
        self.assertEqual(
            list(merged.columns),
            [
                "args_op_name",
                "args_output_type_shape",
                "args_input_type_shape",
                "args_provider",
                "idx",
                "durbase",
                "countbase",
                "duropti",
                "countopti",
            ],
        )
        self.assertEqual(gr.shape, (19, 4))
        self.assertEqual(
            list(gr.columns), ["durbase", "duropti", "countbase", "countopti"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
