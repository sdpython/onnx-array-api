import os
import unittest
import pandas
import matplotlib.pyplot as plt
from onnx_array_api.ext_test_case import ExtTestCase, matplotlib_test
from onnx_array_api.plotting.stat_plot import plot_ort_profile


class TestStatPlot(ExtTestCase):
    @matplotlib_test()
    def test_plot_ort_profile(self):
        data = os.path.join(os.path.dirname(__file__), "data", "prof.csv")
        df = pandas.read_csv(data)
        _, ax = plt.subplots(2, 1)
        plot_ort_profile(df, ax0=ax[0], ax1=ax[1])


if __name__ == "__main__":
    unittest.main()
