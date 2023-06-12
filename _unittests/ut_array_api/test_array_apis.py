import unittest
from inspect import isfunction, ismethod
import numpy as np
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_numpy as xpn
from onnx_array_api.array_api import onnx_ort as xpo

# from onnx_array_api.npx.npx_numpy_tensors import EagerNumpyTensor
# from onnx_array_api.ort.ort_tensors import EagerOrtTensor


class TestArraysApis(ExtTestCase):
    def test_zeros_numpy_1(self):
        c = xpn.zeros(1)
        d = c.numpy()
        self.assertEqualArray(np.array([0], dtype=np.float32), d)

    def test_zeros_ort_1(self):
        c = xpo.zeros(1)
        d = c.numpy()
        self.assertEqualArray(np.array([0], dtype=np.float32), d)

    def test_ffinfo(self):
        dt = np.float32
        fi1 = np.finfo(dt)
        fi2 = xpn.finfo(dt)
        fi3 = xpo.finfo(dt)
        dt1 = fi1.dtype
        dt2 = fi2.dtype
        dt3 = fi3.dtype
        self.assertEqual(dt2, dt3)
        self.assertNotEqual(dt1.__class__, dt2.__class__)
        mi1 = fi1.min
        mi2 = fi2.min
        self.assertEqual(mi1, mi2)
        mi1 = fi1.smallest_normal
        mi2 = fi2.smallest_normal
        self.assertEqual(mi1, mi2)
        for n in dir(fi1):
            if n.startswith("__"):
                continue
            if n in {"machar"}:
                continue
            v1 = getattr(fi1, n)
            with self.subTest(att=n):
                v2 = getattr(fi2, n)
                v3 = getattr(fi3, n)
                if isfunction(v1) or ismethod(v1):
                    try:
                        v1 = v1()
                    except TypeError:
                        continue
                    v2 = v2()
                    v3 = v3()
                if v1 != v2:
                    raise AssertionError(
                        f"12: info disagree on name {n!r}: {v1} != {v2}, "
                        f"type(v1)={type(v1)}, type(v2)={type(v2)}, "
                        f"ismethod={ismethod(v1)}."
                    )
                if v2 != v3:
                    raise AssertionError(
                        f"23: info disagree on name {n!r}: {v2} != {v3}, "
                        f"type(v1)={type(v1)}, type(v2)={type(v2)}, "
                        f"ismethod={ismethod(v1)}."
                    )

    def test_iiinfo(self):
        dt = np.int64
        fi1 = np.iinfo(dt)
        fi2 = xpn.iinfo(dt)
        fi3 = xpo.iinfo(dt)
        dt1 = fi1.dtype
        dt2 = fi2.dtype
        dt3 = fi3.dtype
        self.assertEqual(dt2, dt3)
        self.assertNotEqual(dt1.__class__, dt2.__class__)
        mi1 = fi1.min
        mi2 = fi2.min
        self.assertEqual(mi1, mi2)
        for n in dir(fi1):
            if n.startswith("__"):
                continue
            if n in {"machar"}:
                continue
            v1 = getattr(fi1, n)
            with self.subTest(att=n):
                v2 = getattr(fi2, n)
                v3 = getattr(fi3, n)
                if isfunction(v1) or ismethod(v1):
                    try:
                        v1 = v1()
                    except TypeError:
                        continue
                    v2 = v2()
                    v3 = v3()
                if v1 != v2:
                    raise AssertionError(
                        f"12: info disagree on name {n!r}: {v1} != {v2}, "
                        f"type(v1)={type(v1)}, type(v2)={type(v2)}, "
                        f"ismethod={ismethod(v1)}."
                    )
                if v2 != v3:
                    raise AssertionError(
                        f"23: info disagree on name {n!r}: {v2} != {v3}, "
                        f"type(v1)={type(v1)}, type(v2)={type(v2)}, "
                        f"ismethod={ismethod(v1)}."
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
