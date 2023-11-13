import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api._command_lines_parser import (
    get_main_parser,
    get_parser_translate,
    main,
)


class TestCommandLines1(ExtTestCase):
    def test_main_parser(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("translate", text)

    def test_parser_translate(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_translate().print_help()
        text = st.getvalue()
        self.assertIn("model", text)

    def test_command_translate(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [5, 6])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["res"]),
                make_node("Cos", ["res"], ["Z"]),
            ],
            "g",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])

        with tempfile.TemporaryDirectory() as root:
            model_file = os.path.join(root, "model.onnx")
            with open(model_file, "wb") as f:
                f.write(onnx_model.SerializeToString())

            args = ["translate", "-m", model_file]
            st = StringIO()
            with redirect_stdout(st):
                main(args)

            code = st.getvalue()
            self.assertIn("model = make_model(", code)

            args = ["translate", "-m", model_file, "-a", "light"]
            st = StringIO()
            with redirect_stdout(st):
                main(args)

            code = st.getvalue()
            self.assertIn("start(opset=", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
