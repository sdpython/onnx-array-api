"""
@brief      test tree node (time=5s)
"""
import os
import sys
import time
import unittest
from io import StringIO
from pstats import SortKey

import pandas

from onnx_array_api import __file__ as rootfile
from onnx_array_api.ext_test_case import ExtTestCase, ignore_warnings
from onnx_array_api.profiling import ProfileNode, profile, profile2df, profile2graph


class TestProfiling(ExtTestCase):
    def test_profile(self):
        def simple():
            df = pandas.DataFrame(
                [{"A": "x", "AA": "xx", "AAA": "xxx"}, {"AA": "xxxxxxx", "AAA": "xxx"}]
            )
            return df.to_csv(StringIO())

        rootrem = os.path.normpath(
            os.path.abspath(os.path.join(os.path.dirname(rootfile), ".."))
        )
        ps, res = profile(simple, rootrem=rootrem)
        res = res.replace("\\", "/")
        self.assertIn("function calls", res)
        self.assertNotEmpty(ps)

        ps, res = profile(simple)
        res = res.replace("\\", "/")
        self.assertIn("function calls", res)
        self.assertNotEmpty(ps)

    @ignore_warnings(FutureWarning)
    def test_profile_df(self):
        def simple():
            def simple2():
                df = pandas.DataFrame(
                    [
                        {"A": "x", "AA": "xx", "AAA": "xxx"},
                        {"AA": "xxxxxxx", "AAA": "xxx"},
                    ]
                )
                return df.to_csv(StringIO())

            return simple2()

        rootrem = os.path.normpath(
            os.path.abspath(os.path.join(os.path.dirname(rootfile), ".."))
        )
        ps, df = profile(simple, rootrem=rootrem, as_df=True)
        self.assertIsInstance(df, pandas.DataFrame)
        self.assertEqual(df.loc[0, "namefct"].split("-")[-1], "simple")
        self.assertNotEmpty(ps)
        df = profile2df(ps, False)
        self.assertIsInstance(df, list)
        self.assertIsInstance(df[0], dict)
        df = profile2df(ps, True)
        self.assertIsInstance(df, pandas.DataFrame)

    def test_profile_df_verbose(self):
        calls = [0]

        def f0(t):
            calls[0] += 1
            time.sleep(t)

        def f1(t):
            calls[0] += 1
            time.sleep(t)

        def f2():
            calls[0] += 1
            f1(0.1)
            f1(0.01)

        def f3():
            calls[0] += 1
            f0(0.2)
            f1(0.5)

        def f4():
            calls[0] += 1
            f2()
            f3()

        ps = profile(f4)[0]
        df = self.capture(lambda: profile2df(ps, verbose=True, fLOG=print))[0]
        dfi = df.set_index("fct")
        self.assertEqual(dfi.loc["f4", "ncalls1"], 1)
        self.assertEqual(dfi.loc["f4", "ncalls2"], 1)

    @unittest.skipIf(sys.version_info[:2] < (3, 7), reason="not supported")
    def test_profile_graph(self):
        calls = [0]

        def f0(t):
            calls[0] += 1
            time.sleep(t)

        def f1(t):
            calls[0] += 1
            time.sleep(t)

        def f2():
            calls[0] += 1
            f1(0.1)
            f1(0.01)

        def f3():
            calls[0] += 1
            f0(0.2)
            f1(0.5)

        def f4():
            calls[0] += 1
            f2()
            f3()

        ps = profile(f4)[0]
        profile2df(ps, verbose=False, clean_text=lambda x: x.split("/")[-1])
        root, nodes = profile2graph(ps, clean_text=lambda x: x.split("/")[-1])
        self.assertEqual(len(nodes), 6)
        self.assertIsInstance(nodes, dict)
        self.assertIsInstance(root, ProfileNode)
        self.assertIn("(", str(root))
        dicts = root.as_dict()
        self.assertEqual(10, len(dicts))
        text = root.to_text()
        self.assertIn("1  1", text)
        self.assertIn("        f1", text)
        text = root.to_text(fct_width=20)
        self.assertIn("...", text)
        root.to_text(sort_key=SortKey.CUMULATIVE)
        root.to_text(sort_key=SortKey.TIME)
        self.assertRaise(
            lambda: root.to_text(sort_key=SortKey.NAME), NotImplementedError
        )
        js = root.to_json(indent=2)
        self.assertIn('"details"', js)
        js = root.to_json(as_str=False)
        self.assertIsInstance(js, dict)

    def test_profile_graph_recursive2(self):
        def f0(t):
            if t < 0.2:
                time.sleep(t)
            else:
                f1(t - 0.1)

        def f1(t):
            if t < 0.1:
                time.sleep(t)
            else:
                f0(t)

        def f4():
            f1(0.3)

        ps = profile(f4)[0]
        profile2df(ps, verbose=False, clean_text=lambda x: x.split("/")[-1])
        root, nodes = profile2graph(ps, clean_text=lambda x: x.split("/")[-1])
        self.assertEqual(len(nodes), 4)
        text = root.to_text()
        self.assertIn("        f1", text)
        js = root.to_json(indent=2)
        self.assertIn('"details"', js)

    def test_profile_graph_recursive1(self):
        def f0(t):
            if t < 0.1:
                time.sleep(t)
            else:
                f0(t - 0.1)

        def f4():
            f0(0.15)

        ps = profile(f4)[0]
        profile2df(ps, verbose=False, clean_text=lambda x: x.split("/")[-1])
        root, nodes = profile2graph(ps, clean_text=lambda x: x.split("/")[-1])
        self.assertEqual(len(nodes), 3)
        text = root.to_text()
        self.assertIn("    f0", text)
        js = root.to_json(indent=2)
        self.assertIn('"details"', js)


if __name__ == "__main__":
    unittest.main()
