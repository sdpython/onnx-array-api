import unittest
from onnx import load
from onnx.checker import check_model
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.validation.tools import randomize_proto


class TestTools(ExtTestCase):
    def test_randomize_proto(self):
        data = self.relative_path(__file__, "data", "small.onnx")
        with open(data, "rb") as f:
            model = load(f)
        check_model(model)
        rnd = randomize_proto(model)
        self.assertEqual(len(model.SerializeToString()), len(rnd.SerializeToString()))
        check_model(rnd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
