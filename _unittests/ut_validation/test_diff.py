import unittest
from onnx import load
from onnx.checker import check_model
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.ort.ort_optimizers import ort_optimized_model
from onnx_array_api.validation.diff import text_diff, html_diff


class TestDiff(ExtTestCase):
    def test_diff_optimized(self):
        data = self.relative_path(__file__, "data", "small.onnx")
        with open(data, "rb") as f:
            model = load(f)
        optimized = ort_optimized_model(model)
        check_model(optimized)
        diff = text_diff(model, optimized)
        self.assertIn("^^^^^^^^^^^^^^^^", diff)
        ht = html_diff(model, optimized)
        self.assertIn("<html><body>", ht)


if __name__ == "__main__":
    unittest.main(verbosity=2)
