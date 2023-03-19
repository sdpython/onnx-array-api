import unittest
import os
import sys
import importlib
import subprocess
import time
from onnx_array_api.ext_test_case import ExtTestCase


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def test_documentation_examples(self):
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, "..", "..", "_doc", "examples"))
        found = os.listdir(fold)
        tested = 0
        for name in found:
            if name.startswith("plot_") and name.endswith(".py"):
                perf = time.perf_counter()
                try:
                    mod = import_source(fold, os.path.splitext(name)[0])
                    assert mod is not None
                except FileNotFoundError:
                    # try another way
                    cmds = [sys.executable, "-u", os.path.join(fold, name)]
                    p = subprocess.Popen(
                        cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    res = p.communicate()
                    out, err = res
                    st = err.decode("ascii", errors="ignore")
                    if len(st) > 0 and "Traceback" in st:
                        if '"dot" not found in path.' in st:
                            # dot not installed, this part
                            # is tested in onnx framework
                            print(f"failed: {name!r} due to missing dot.")
                            continue
                        raise AssertionError(
                            "Example '{}' (cmd: {} - exec_prefix='{}') "
                            "failed due to\n{}"
                            "".format(name, cmds, sys.exec_prefix, st)
                        )
                dt = time.perf_counter() - perf
                print(f"{dt:.3f}: run {name!r}")
                tested += 1
        if tested == 0:
            raise AssertionError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
