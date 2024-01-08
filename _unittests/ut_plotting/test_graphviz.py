import os
import unittest
import onnx.parser
from onnx_array_api.ext_test_case import (
    ExtTestCase,
    skipci_apple,
    skipif_ci_windows,
    skipif_ci_apple,
)
from onnx_array_api.plotting.dot_plot import to_dot
from onnx_array_api.plotting.graphviz_helper import draw_graph_graphviz, plot_dot


class TestGraphviz(ExtTestCase):
    @classmethod
    def _get_graph(cls):
        return onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }"""
        )

    @skipif_ci_windows("graphviz not installed")
    @skipif_ci_apple("graphviz not installed")
    def test_draw_graph_graphviz(self):
        fout = "test_draw_graph_graphviz.png"
        dot = to_dot(self._get_graph())
        draw_graph_graphviz(dot, image=fout)
        self.assertExists(os.path.exists(fout))

    @skipif_ci_windows("graphviz not installed")
    @skipif_ci_apple("graphviz not installed")
    def test_draw_graph_graphviz_proto(self):
        fout = "test_draw_graph_graphviz_proto.png"
        dot = self._get_graph()
        draw_graph_graphviz(dot, image=fout)
        self.assertExists(os.path.exists(fout))

    @skipif_ci_windows("graphviz not installed")
    @skipif_ci_apple("graphviz not installed")
    def test_plot_dot(self):
        dot = to_dot(self._get_graph())
        ax = plot_dot(dot)
        ax.get_figure().savefig("test_plot_dot.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
