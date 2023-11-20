import os
import pprint
import struct
import unittest
import warnings
import numpy
import pandas
from onnx import TensorProto
from onnx_array_api.validation.f8 import (
    CastFloat8,
    UndefinedCastError,
    display_fe4m3,
    display_fe5m2,
    display_float16,
    display_float32,
    fe4m3_to_float32,
    fe5m2_to_float32,
    fe4m3_to_float32_float,
    fe5m2_to_float32_float,
    float32_to_fe4m3,
    float32_to_fe5m2,
    search_float32_into_fe4m3,
    search_float32_into_fe5m2,
)
from onnx_array_api.ext_test_case import ExtTestCase
from ml_dtypes import float8_e4m3fn, float8_e5m2


def new_cvt_float32_to_e4m3fn(x):
    return numpy.array(x, dtype=numpy.float32).astype(float8_e4m3fn)


def new_cvt_e4m3fn_to_float32(x):
    return numpy.array(x, dtype=float8_e4m3fn).astype(numpy.float32)


def new_cvt_float32_to_e5m2(x):
    return numpy.array(x, dtype=numpy.float32).astype(float8_e5m2)


def new_cvt_e5m2_to_float32(x):
    return numpy.array(x, dtype=float8_e5m2).astype(numpy.float32)


class TestF8(ExtTestCase):
    def test_fe4m3fn_to_float32_float_paper(self):
        self.assertEqual(fe4m3_to_float32_float(int("1111110", 2)), 448)
        self.assertEqual(fe4m3_to_float32_float(int("1000", 2)), 2 ** (-6))
        self.assertEqual(fe4m3_to_float32_float(int("1", 2)), 2 ** (-9))
        self.assertEqual(fe4m3_to_float32_float(int("111", 2)), 0.875 * 2 ** (-6))
        self.assertRaise(lambda: fe4m3_to_float32_float(256), ValueError)

    def test_fe4m3fn_to_float32_paper(self):
        self.assertEqual(fe4m3_to_float32(int("1111110", 2)), 448)
        self.assertEqual(fe4m3_to_float32(int("1000", 2)), 2 ** (-6))
        self.assertEqual(fe4m3_to_float32(int("1", 2)), 2 ** (-9))
        self.assertEqual(fe4m3_to_float32(int("111", 2)), 0.875 * 2 ** (-6))
        self.assertRaise(lambda: fe4m3_to_float32(256), ValueError)

    def test_fe5m2_to_float32_float_paper(self):
        self.assertEqual(fe5m2_to_float32_float(int("1111011", 2)), 57344)
        self.assertEqual(fe5m2_to_float32_float(int("100", 2)), 2 ** (-14))
        self.assertEqual(fe5m2_to_float32_float(int("11", 2)), 0.75 * 2 ** (-14))
        self.assertEqual(fe5m2_to_float32_float(int("1", 2)), 2 ** (-16))
        self.assertRaise(lambda: fe5m2_to_float32_float(256), ValueError)
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("1111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("1111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("1111111", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("11111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("11111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32_float(int("11111111", 2))))
        self.assertEqual(fe5m2_to_float32_float(int("1111100", 2)), numpy.inf)
        self.assertEqual(fe5m2_to_float32_float(int("11111100", 2)), -numpy.inf)

    def test_fe5m2_to_float32_paper(self):
        self.assertEqual(fe5m2_to_float32(int("1111011", 2)), 57344)
        self.assertEqual(fe5m2_to_float32(int("100", 2)), 2 ** (-14))
        self.assertEqual(fe5m2_to_float32(int("11", 2)), 0.75 * 2 ** (-14))
        self.assertEqual(fe5m2_to_float32(int("1", 2)), 2 ** (-16))
        self.assertRaise(lambda: fe5m2_to_float32(256), ValueError)
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("1111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("1111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("1111111", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("11111101", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("11111110", 2))))
        self.assertTrue(numpy.isnan(fe5m2_to_float32(int("11111111", 2))))
        self.assertEqual(fe5m2_to_float32(int("1111100", 2)), numpy.inf)
        self.assertEqual(fe5m2_to_float32(int("11111100", 2)), -numpy.inf)

    def test_fe4m3fn_to_float32_all(self):
        for i in range(0, 256):
            a = fe4m3_to_float32_float(i)
            b = fe4m3_to_float32(i)
            if numpy.isnan(a):
                self.assertTrue(numpy.isnan(b))
                continue
            self.assertEqual(a, b)

    def test_fe4m3fn_to_float32_all_ml_types(self):
        for i in range(0, 256):
            a = fe4m3_to_float32_float(i)
            b = fe4m3_to_float32(i)
            c = new_cvt_float32_to_e4m3fn(b)
            if numpy.isnan(a):
                self.assertTrue(numpy.isnan(b))
                continue
            self.assertEqual(float(a), float(c))
            self.assertEqual(a, b)

    def test_display_float(self):
        f = 45
        s = display_float32(f)
        self.assertEqual(s, "0.10000100.01101000000000000000000")
        s = display_fe4m3(f)
        self.assertEqual(s, "0.0101.101")
        s = display_fe5m2(f)
        self.assertEqual(s, "0.01011.01")
        s = display_float16(numpy.float16(f))
        self.assertEqual(s, "0.10100.0110100000")

    def test_search_float32_into_fe4m3fn_simple(self):
        values = [
            (480, 448),
            (0.001953125, 0.001953125),
            (416, 416),
            (192.5, 192),
            (304, 320),
            (368, 384),
            (248, 256),
            (432, 448),
            (100, 96),
            (400, 384),
            (336, 320),
            (272, 256),
            (23.5, 24),
            (-447.5, -448),
            (79.5, 80),
        ]
        for v, expected in values:
            with self.subTest(v=v, expected=expected):
                try:
                    b = search_float32_into_fe4m3(v)
                except UndefinedCastError:
                    b = None
                if b is not None:
                    got = fe4m3_to_float32_float(b)
                    self.assertEqual(expected, got)
                b = float32_to_fe4m3(v)
                got = fe4m3_to_float32_float(b)
                self.assertEqual(expected, got)

    def test_search_float32_into_fe5m2_simple(self):
        values = [
            (73728, 57344),
            (61440, 57344),
            (0.0017089844, 0.0017089844),
            (20480, 20480),
            (20480.5, 20480),
            (14.5, 14),
            (-3584.5, -3584),
            (352, 384),
            (416, 384),
            (0.4, 0.375),
            (0.4068359375, 0.4375),
        ]
        for v, expected in values:
            with self.subTest(v=v, expected=expected):
                if v == expected:
                    b = search_float32_into_fe5m2(v)
                    got = fe5m2_to_float32_float(b)
                    self.assertLess(abs(expected - got), 1e-5)
                    b = float32_to_fe5m2(v)
                    got = fe5m2_to_float32_float(b)
                    self.assertLess(abs(expected - got), 1e-5)
                else:
                    try:
                        b1 = search_float32_into_fe5m2(v)
                    except UndefinedCastError:
                        b1 = None
                    if b1 is not None:
                        got1 = fe5m2_to_float32_float(b1)
                        self.assertEqual(got1, expected)

                    b2 = float32_to_fe5m2(v)
                    got2 = fe5m2_to_float32(b2)
                    self.assertEqual(got2, expected)
                    if b1 is not None:
                        self.assertEqual(b1, b2)

    def test_search_float32_into_fe4m3fn_equal(self):
        values = [(fe4m3_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        for value, expected in values:
            with self.subTest(
                value=value, expected=expected, bin=display_float32(value)
            ):
                b = search_float32_into_fe4m3(value)
                nf = float32_to_fe4m3(value)
                if expected in (127, 255):
                    self.assertIn(b, (127, 255))
                    self.assertIn(nf, (127, 255))
                elif value != 0:
                    self.assertEqual(expected, b)
                    self.assertEqual(expected, nf)
                else:
                    self.assertIn(b, (0, 128))
                    self.assertIn(nf, (0, 128))

    def test_search_float32_into_fe5m2_equal(self):
        values = [(fe5m2_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        for value, expected in values:
            with self.subTest(
                value=value, expected=expected, bin=display_float32(value)
            ):
                b = search_float32_into_fe5m2(value, saturate=False)
                nf = float32_to_fe5m2(value, saturate=False)
                cf = new_cvt_float32_to_e5m2(value)
                if expected in {253, 254, 255, 125, 126, 127}:  # nan
                    self.assertIn(b, {253, 254, 255, 125, 126, 127})
                    self.assertIn(nf, {253, 254, 255, 125, 126, 127})
                elif value != 0:
                    self.assertEqual(expected, b)
                    self.assertEqual(expected, nf)
                else:
                    self.assertIn(b, (0, 128))
                    self.assertIn(nf, (0, 128))
                if numpy.isnan(float(cf)):
                    self.assertTrue(numpy.isnan(fe5m2_to_float32(nf)))
                    continue
                self.assertEqual(fe5m2_to_float32(nf), float(cf))

    def test_search_float32_into_fe4m3fn(self):
        values = [(fe4m3_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        obs = []
        values += [(1e-9, 0), (-1e-9, 0), (1e8, 448), (-1e-8, -448)]
        wrong = 0
        for value, origin in values:
            for add in [
                0,
                -0.4,
                -1e-4,
                1e-4,
                0.4,
                (3, "x"),
                (0.3, "x"),
                16,
                32,
                64,
                -16,
                -32,
                -64,
            ]:
                if isinstance(add, tuple):
                    v = value * add[0]
                    add = v - value
                else:
                    v = value + add
                try:
                    b = search_float32_into_fe4m3(v)
                except UndefinedCastError:
                    if add == 0:
                        b = search_float32_into_fe4m3(origin)
                    else:
                        continue
                nf = float32_to_fe4m3(v)
                if b != nf:
                    # signed, not signed zero?
                    if (nf & 0x7F) == 0 and (b & 0x7F) == 0:
                        continue
                    wrong += 1
                    obs.append(
                        dict(
                            origin=origin,
                            value=v,
                            bin_value=display_float32(v),
                            expected_search=b,
                            float_expected=fe4m3_to_float32_float(b),
                            bin_expected=display_fe4m3(b),
                            got_bit=nf,
                            bin_got=display_fe4m3(nf),
                            float_got=fe4m3_to_float32_float(nf),
                            ok="" if b == nf else "WRONG",
                            true=value,
                            add=add,
                            exponent=(
                                int.from_bytes(
                                    struct.pack("<f", numpy.float32(v)), "little"
                                )
                                & 0x7F800000
                            )
                            >> 23,
                            d1=v - fe4m3_to_float32_float(nf),
                            d2=v - fe4m3_to_float32_float(b),
                        )
                    )
        if wrong > 0:
            output = os.path.join(
                os.path.dirname(__file__), "temp_search_float32_into_fe4m3fn.xlsx"
            )
            pandas.DataFrame(obs).to_excel(output)
            raise AssertionError(
                f"{wrong} conversion are wrong\n{pprint.pformat(obs[:2])}"
            )

    def test_search_float32_into_fe5m2(self):
        values = [(fe5m2_to_float32_float(i), i) for i in range(0, 256)]
        values.sort()

        obs = []
        values += [
            (1e-8, 0),
            (-1e-8, 0),
            (1e8, 57344),
            (-1e8, -57344),
            (352, 384),
            (416, 384),
        ]
        wrong = 0
        for value, origin in values:
            for add in [
                0,
                -0.4,
                -1e-4,
                1e-4,
                0.4,
                (3, "x"),
                (0.3, "x"),
                16,
                32,
                64,
                -16,
                -32,
                -64,
            ]:
                if isinstance(add, tuple):
                    v = value * add[0]
                    with warnings.catch_warnings(record=True) as w:
                        if numpy.isinf(value):
                            add = value
                        else:
                            add = v - value
                            if w:
                                raise AssertionError(
                                    f"A warning was thrown for v={v}, "
                                    f"value={value}, w={w[0]}."
                                )
                else:
                    v = value + add
                try:
                    b = search_float32_into_fe5m2(v)
                except UndefinedCastError:
                    if add == 0:
                        b = search_float32_into_fe5m2(origin)
                    else:
                        continue
                nf = float32_to_fe5m2(v)
                if b != nf:
                    # signed, not signed zero?
                    if (nf & 0x7F) == 0 and (b & 0x7F) == 0:
                        continue
                    wrong += 1
                    obs.append(
                        dict(
                            value=v,
                            bin_value=display_float32(v),
                            expected=b,
                            float_expected=fe5m2_to_float32_float(b),
                            bin_expected=display_fe5m2(b),
                            got=nf,
                            bin_got=display_fe5m2(nf),
                            float_got=fe5m2_to_float32_float(nf),
                            ok="" if b == nf else "WRONG",
                            true=value,
                            add=add,
                        )
                    )
        if wrong > 0:
            output = os.path.join(
                os.path.dirname(__file__), "temp_search_float32_into_fe5m2.xlsx"
            )
            pandas.DataFrame(obs).to_excel(output)
            raise AssertionError(
                f"{wrong} conversion are wrong\n{pprint.pformat(obs[:2])}"
            )

    def test_inf_nan(self):
        np_fp32 = numpy.array(
            [
                "0.47892547",
                "0.48033667",
                "0.49968487",
                "0.81910545",
                "0.47031248",
                "0.816468",
                "0.21087195",
                "0.7229038",
                "NaN",
                "INF",
                "+INF",
                "-INF",
            ],
            dtype=numpy.float32,
        )
        v_fe4m3_to_float32 = numpy.vectorize(fe4m3_to_float32)
        v_float32_to_fe4m3 = numpy.vectorize(float32_to_fe4m3)
        v_float32_to_fe5m2 = numpy.vectorize(float32_to_fe5m2)
        v_fe5m2_to_float32 = numpy.vectorize(fe5m2_to_float32)

        got = v_fe4m3_to_float32(v_float32_to_fe4m3(np_fp32, saturate=False))
        expected = numpy.array(
            [
                0.46875,
                0.46875,
                0.5,
                0.8125,
                0.46875,
                0.8125,
                0.203125,
                0.75,
                numpy.nan,
                numpy.nan,
                numpy.nan,
                -numpy.nan,
            ],
            dtype=numpy.float32,
        )
        self.assertEqualArray(expected, got)
        got = v_fe5m2_to_float32(v_float32_to_fe5m2(np_fp32, saturate=False))
        expected = numpy.array(
            [
                0.5,
                0.5,
                0.5,
                0.875,
                0.5,
                0.875,
                0.21875,
                0.75,
                numpy.nan,
                numpy.inf,
                numpy.inf,
                -numpy.inf,
            ],
            dtype=numpy.float32,
        )
        self.assertEqualArray(expected, got)

    def test_search_e4m3_pow(self):
        self.assertTrue(hasattr(CastFloat8, "values_e4m3fn"))
        for p in range(1, 40):
            v = 2 ** (-p)
            try:
                r1 = search_float32_into_fe4m3(v)
            except UndefinedCastError:
                continue
            r2 = float32_to_fe4m3(v)
            if r1 != r2:
                ex = abs(v - fe4m3_to_float32(r1)) == abs(v - fe4m3_to_float32(r2))
                raise AssertionError(
                    f"p={p}, v={v}, "
                    f"search={r1}:{display_fe4m3(r1)}={fe4m3_to_float32(r1)} != "
                    f"bit={r2}:{display_fe4m3(r2)}={fe4m3_to_float32(r2)} "
                    f"d1={v - fe4m3_to_float32(r1)} d2={v - fe4m3_to_float32(r2)} "
                    f"|d1|==|d2|={ex}"
                )
        for p in range(1, 40):
            v = -(2 ** (-p))
            try:
                r1 = search_float32_into_fe4m3(v)
            except UndefinedCastError:
                continue
            r2 = float32_to_fe4m3(v)
            if r1 != r2:
                ex = abs(v - fe4m3_to_float32(r1)) == abs(v - fe4m3_to_float32(r2))
                raise AssertionError(
                    f"p={p}, v={v}, "
                    f"search={r1}:{display_fe4m3(r1)}={fe4m3_to_float32(r1)} != "
                    f"bit={r2}:{display_fe4m3(r2)}={fe4m3_to_float32(r2)} "
                    f"d1={v - fe4m3_to_float32(r1)} d2={v - fe4m3_to_float32(r2)} "
                    f"|d1|==|d2|={ex}"
                )

    def test_search_e5m2_pow(self):
        self.assertTrue(hasattr(CastFloat8, "values_e5m2"))
        for p in range(1, 40):
            v = 2 ** (-p)
            try:
                r1 = search_float32_into_fe5m2(v)
            except UndefinedCastError:
                continue
            r2 = float32_to_fe5m2(v)
            if r1 != r2:
                ex = abs(v - fe5m2_to_float32(r1)) == abs(v - fe5m2_to_float32(r2))
                raise AssertionError(
                    f"p={p}, v={v}, "
                    f"search={r1}:{display_fe5m2(r1)}={fe5m2_to_float32(r1)} != "
                    f"bit={r2}:{display_fe5m2(r2)}={fe5m2_to_float32(r2)} "
                    f"d1={v - fe4m3_to_float32(r1)} d2={v - fe5m2_to_float32(r2)} "
                    f"|d1|==|d2|={ex}"
                )
        for p in range(1, 40):
            v = -(2 ** (-p))
            try:
                r1 = search_float32_into_fe5m2(v)
            except UndefinedCastError:
                continue
            r2 = float32_to_fe5m2(v)
            if r1 != r2:
                ex = abs(v - fe5m2_to_float32(r1)) == abs(v - fe5m2_to_float32(r2))
                raise AssertionError(
                    f"p={p}, v={v}, "
                    f"search={r1}:{display_fe5m2(r1)}={fe5m2_to_float32(r1)} != "
                    f"bit={r2}:{display_fe5m2(r2)}={fe5m2_to_float32(r2)} "
                    f"d1={v - fe4m3_to_float32(r1)} d2={v - fe5m2_to_float32(r2)} "
                    f"|d1|==|d2|={ex}"
                )

    def test_float32_to_fe4m3fn_inf(self):
        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(numpy.inf)
        a = search_float32_into_fe4m3(v0, saturate=False)
        b = search_float32_into_fe4m3(v1, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(numpy.inf)
        a = float32_to_fe4m3(v0, saturate=False)
        b = float32_to_fe4m3(v1, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(-numpy.nan)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe4m3(v0)
        b = search_float32_into_fe4m3(v1)
        self.assertEqual(a, b)

        v0 = numpy.float32(-numpy.nan)
        v1 = numpy.float32(-numpy.inf)
        a = float32_to_fe4m3(v0, saturate=False)
        b = float32_to_fe4m3(v1, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = search_float32_into_fe4m3(v0)
        b = search_float32_into_fe4m3(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe4m3(v0)
        b = search_float32_into_fe4m3(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = float32_to_fe4m3(v0)
        b = float32_to_fe4m3(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = float32_to_fe4m3(v0)
        b = float32_to_fe4m3(v1)
        self.assertNotEqual(a, b)

    def test_float32_to_fe5m2_inf(self):
        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(numpy.inf)
        a = search_float32_into_fe5m2(v0)
        b = search_float32_into_fe5m2(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(numpy.inf)
        a = float32_to_fe5m2(v0)
        b = float32_to_fe5m2(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = search_float32_into_fe5m2(v0)
        b = search_float32_into_fe5m2(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe5m2(v0)
        b = search_float32_into_fe5m2(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = float32_to_fe5m2(v0)
        b = float32_to_fe5m2(v1)
        self.assertNotEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = float32_to_fe5m2(v0)
        b = float32_to_fe5m2(v1)
        self.assertNotEqual(a, b)

    # UZ

    def test_search_float32_into_fe4m3fnuz_simple(self):
        values = [
            (-0.0146484375, -0.0146484375),  # 143
            (0, 0),
            (4, 4),  # 80
            (-240, -240),
            (0.04296875, 0.04296875),
            (239.5, 240),
        ]
        for v, expected in values:
            with self.subTest(v=v, expected=expected):
                b = search_float32_into_fe4m3(v, uz=True)
                got = fe4m3_to_float32_float(b, uz=True)
                self.assertEqual(expected, got)
                b = float32_to_fe4m3(v, uz=True)
                self.assertTrue(b >= 0)
                self.assertTrue(b < 256)
                got = fe4m3_to_float32_float(b, uz=True)
                self.assertEqual(expected, got)

    def test_search_float32_into_fe5m2fnuz_simple(self):
        values = [
            (73728, 57344),
            (61440, 57344),
            (100000000, 57344),
            (-7, -7),  # 203
            (4, 4),  # 72
            (-57344, -57344),
            (1792.0, 1792.0),
            (0.046875, 0.046875),  # 46
        ]
        for v, expected in values:
            with self.subTest(v=v, expected=expected):
                b = search_float32_into_fe5m2(v, fn=True, uz=True)
                got = fe5m2_to_float32_float(b, fn=True, uz=True)
                self.assertEqual(expected, got)
                b = float32_to_fe5m2(v, fn=True, uz=True)
                self.assertTrue(b >= 0)
                self.assertTrue(b < 256)
                got = fe5m2_to_float32_float(b, fn=True, uz=True)
                self.assertEqual(expected, got)

    def test_fe4m3fnuz_to_float32_all(self):
        for i in range(0, 256):
            a = fe4m3_to_float32_float(i, uz=True)
            b = fe4m3_to_float32(i, uz=True)
            if numpy.isnan(a):
                self.assertTrue(numpy.isnan(b))
                continue
            self.assertEqual(a, b)

    def test_fe5m2fnuz_to_float32_all(self):
        for i in range(0, 256):
            a = fe5m2_to_float32_float(i, fn=True, uz=True)
            b = fe5m2_to_float32(i, fn=True, uz=True)
            if numpy.isnan(a):
                self.assertTrue(numpy.isnan(b))
                continue
            self.assertEqual(a, b)

    def test_search_float32_into_fe4m3fnuz(self):
        values = [(fe4m3_to_float32_float(i, uz=True), i) for i in range(0, 256)]
        values.sort()

        obs = []
        values += [(1e-9, 0), (-1e-9, 0), (1e8, 448), (-1e-8, -448)]
        wrong = 0
        for value, origin in values:
            for add in [0, -0.4, -1e-4, 1e-4, 0.4, (3, "x"), (0.3, "x")]:
                if isinstance(add, tuple):
                    v = value * add[0]
                    add = v - value
                else:
                    v = value + add
                try:
                    b = search_float32_into_fe4m3(v, uz=True)
                except UndefinedCastError:
                    continue
                nf = float32_to_fe4m3(v, uz=True)
                if b != nf:
                    wrong += 1
                    obs.append(
                        dict(
                            origin=origin,
                            value=v,
                            bin_value=display_float32(v),
                            expected_search=b,
                            float_expected=fe4m3_to_float32_float(b, uz=True),
                            bin_expected=display_fe4m3(b),
                            got_bit=nf,
                            bin_got=display_fe4m3(nf),
                            float_got=fe4m3_to_float32_float(nf, uz=True),
                            ok="" if b == nf else "WRONG",
                            true=value,
                            add=add,
                        )
                    )
        if wrong > 0:
            output = os.path.join(
                os.path.dirname(__file__), "temp_search_float32_into_fe4m3fn.xlsx"
            )
            pandas.DataFrame(obs).to_excel(output)
            raise AssertionError(
                f"{wrong} conversion are wrong\n{pprint.pformat(obs[:2])}"
            )

    def test_search_float32_into_fe5m2fnuz(self):
        values = [
            (fe5m2_to_float32_float(i, fn=True, uz=True), i) for i in range(0, 256)
        ]
        values.sort()

        obs = []
        values += [(1e-9, 0), (-1e-9, 0), (1e8, 448), (-1e-8, -448)]
        wrong = 0
        for value, origin in values:
            for add in [0, -0.4, -1e-4, 1e-4, 0.4, (3, "x"), (0.3, "x")]:
                if isinstance(add, tuple):
                    v = value * add[0]
                    add = v - value
                else:
                    v = value + add
                try:
                    b = search_float32_into_fe5m2(v, fn=True, uz=True)
                except UndefinedCastError:
                    continue
                nf = float32_to_fe5m2(v, fn=True, uz=True)
                if b != nf:
                    wrong += 1
                    obs.append(
                        dict(
                            origin=origin,
                            value=v,
                            bin_value=display_float32(v),
                            expected_search=b,
                            float_expected=fe5m2_to_float32_float(b, fn=True, uz=True),
                            bin_expected=display_fe4m3(b),
                            got_bit=nf,
                            bin_got=display_fe5m2(nf),
                            float_got=fe5m2_to_float32_float(nf, fn=True, uz=True),
                            ok="" if b == nf else "WRONG",
                            true=value,
                            add=add,
                        )
                    )
        if wrong > 0:
            output = os.path.join(
                os.path.dirname(__file__), "temp_search_float32_into_fe4m3fn.xlsx"
            )
            pandas.DataFrame(obs).to_excel(output)
            raise AssertionError(
                f"{wrong} conversion are wrong\n{pprint.pformat(obs[:2])}"
            )

    def test_search_float32_to_fe4m3fnuz_inf(self):
        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(numpy.inf)
        a = search_float32_into_fe4m3(v0, uz=True, saturate=False)
        b = search_float32_into_fe4m3(v1, uz=True, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(-numpy.nan)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe4m3(v0, uz=True, saturate=False)
        b = search_float32_into_fe4m3(v1, uz=True, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = search_float32_into_fe4m3(v0, uz=True)
        b = search_float32_into_fe4m3(v1, uz=True)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe4m3(v0, uz=True, saturate=False)
        b = search_float32_into_fe4m3(v1, uz=True, saturate=False)
        self.assertEqual(a, b)

    def test_float32_to_fe4m3fnuz_inf(self):
        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = float32_to_fe4m3(v0, uz=True)
        b = float32_to_fe4m3(v1, uz=True)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = float32_to_fe4m3(v0, uz=True, saturate=False)
        b = float32_to_fe4m3(v1, uz=True, saturate=False)
        self.assertEqual(a, b)

    def test_float32_to_fe5m2fnuz_inf(self):
        mx = numpy.nan
        v0 = numpy.float32(mx)
        v1 = numpy.float32(numpy.inf)
        a = search_float32_into_fe5m2(v0, fn=True, uz=True, saturate=False)
        b = search_float32_into_fe5m2(v1, fn=True, uz=True, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(mx)
        v1 = numpy.float32(numpy.inf)
        a = float32_to_fe5m2(v0, fn=True, uz=True, saturate=False)
        b = float32_to_fe5m2(v1, fn=True, uz=True, saturate=False)
        self.assertEqual(a, b)

        mi = numpy.nan
        v0 = numpy.float32(mi)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe5m2(v0, fn=True, uz=True, saturate=False)
        b = search_float32_into_fe5m2(v1, fn=True, uz=True, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(mi)
        v1 = numpy.float32(-numpy.inf)
        a = float32_to_fe5m2(v0, fn=True, uz=True, saturate=False)
        b = float32_to_fe5m2(v1, fn=True, uz=True, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = search_float32_into_fe5m2(v0, fn=True, uz=True)
        b = search_float32_into_fe5m2(v1, fn=True, uz=True)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = search_float32_into_fe5m2(v0, fn=True, uz=True, saturate=False)
        b = search_float32_into_fe5m2(v1, fn=True, uz=True, saturate=False)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.nan)
        v1 = numpy.float32(-numpy.nan)
        a = float32_to_fe5m2(v0, fn=True, uz=True)
        b = float32_to_fe5m2(v1, fn=True, uz=True)
        self.assertEqual(a, b)

        v0 = numpy.float32(numpy.inf)
        v1 = numpy.float32(-numpy.inf)
        a = float32_to_fe5m2(v0, fn=True, uz=True, saturate=False)
        b = float32_to_fe5m2(v1, fn=True, uz=True, saturate=False)
        self.assertEqual(a, b)

    def test_simple_fe4m3(self):
        values = [448]
        cvt2 = [float32_to_fe4m3(v, uz=True) for v in values]
        cvt1 = [search_float32_into_fe4m3(v, uz=True) for v in values]
        back1 = [fe4m3_to_float32(c, uz=True) for c in cvt1]
        back2 = [fe4m3_to_float32(c, uz=True) for c in cvt2]
        self.assertEqual(cvt1, cvt2)
        self.assertEqual(back1, back2)

        values = [0, 0.5, 1, 240, 10]
        cvt = [search_float32_into_fe4m3(v, uz=True) for v in values]
        back = [fe4m3_to_float32(c, uz=True) for c in cvt]
        self.assertEqual(values, back)

        values = [0, 0.5, 1, 240, 10]
        cvt = [float32_to_fe4m3(v, uz=True) for v in values]
        back = [fe4m3_to_float32(c, uz=True) for c in cvt]
        self.assertEqual(values, back)

    # ml-dtypes

    def test_inf_nan_ml_dtypes(self):
        x = numpy.float32(numpy.inf)
        g1 = float32_to_fe4m3(x, saturate=False)
        g2 = float32_to_fe5m2(x, saturate=False)
        i1 = fe4m3_to_float32(g1)
        i2 = fe5m2_to_float32(g2)
        self.assertNotEqual(i1, 448)
        self.assertTrue(numpy.isinf(i2))
        m1 = new_cvt_float32_to_e4m3fn(x)
        m2 = new_cvt_float32_to_e5m2(x)
        self.assertTrue(numpy.isnan(m1))  # different from ONNX choice
        self.assertTrue(numpy.isinf(m2))

        x = numpy.float32(numpy.nan)
        g1 = float32_to_fe4m3(x)
        g2 = float32_to_fe5m2(x)
        i1 = fe4m3_to_float32(g1)
        i2 = fe5m2_to_float32(g2)
        self.assertTrue(numpy.isnan(i1))
        self.assertTrue(numpy.isnan(i2))
        m1 = new_cvt_float32_to_e4m3fn(x)
        m2 = new_cvt_float32_to_e5m2(x)
        self.assertTrue(numpy.isnan(m1))
        self.assertTrue(numpy.isnan(m2))

    def test_float8_e4m3fn_inf(self):
        x = numpy.float32(numpy.inf)
        to = float32_to_fe4m3(x)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, 448)

        x = numpy.float32(numpy.inf)
        to = float32_to_fe4m3(x, saturate=False)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe4m3(x)
        self.assertEqual(to & 0x80, 0x80)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, -448)

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe4m3(x, saturate=False)
        self.assertEqual(to & 0x80, 0x80)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e4m3fnuz_inf(self):
        x = numpy.float32(numpy.inf)
        to = float32_to_fe4m3(x, uz=True)
        back = fe4m3_to_float32(to, uz=True)
        self.assertEqual(back, 240)

        x = numpy.float32(numpy.inf)
        to = float32_to_fe4m3(x, uz=True, saturate=False)
        back = fe4m3_to_float32(to, uz=True)
        self.assertTrue(numpy.isnan(back))

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe4m3(x, uz=True)
        back = fe4m3_to_float32(to, uz=True)
        self.assertEqual(back, -240)

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe4m3(x, uz=True, saturate=False)
        back = fe4m3_to_float32(to, uz=True)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e5m2_inf(self):
        x = numpy.float32(numpy.inf)
        to = float32_to_fe5m2(x)
        back = fe5m2_to_float32(to)
        self.assertEqual(back, 57344)

        x = numpy.float32(numpy.inf)
        to = float32_to_fe5m2(x, saturate=False)
        back = fe5m2_to_float32(to)
        self.assertTrue(numpy.isinf(back))

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe5m2(x)
        self.assertEqual(to & 0x80, 0x80)
        back = fe5m2_to_float32(to)
        self.assertEqual(back, -57344)

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe5m2(x, saturate=False)
        self.assertEqual(to & 0x80, 0x80)
        back = fe5m2_to_float32(to)
        self.assertTrue(numpy.isinf(back))
        self.assertTrue(back < 0)

    def test_float8_e5m2fnuz_inf(self):
        x = numpy.float32(numpy.inf)
        to = float32_to_fe5m2(x, fn=True, uz=True)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 57344)

        x = numpy.float32(numpy.inf)
        to = float32_to_fe5m2(x, fn=True, uz=True, saturate=False)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(numpy.isnan(back))

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe5m2(x, fn=True, uz=True)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, -57344)

        x = numpy.float32(-numpy.inf)
        to = float32_to_fe5m2(x, fn=True, uz=True, saturate=False)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e4m3fn_out_of_range(self):
        x = numpy.float32(1000000)
        to = float32_to_fe4m3(x)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, 448)

        x = numpy.float32(1000000)
        to = float32_to_fe4m3(x, saturate=False)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

        x = numpy.float32(-1000000)
        to = float32_to_fe4m3(x)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, -448)

        x = numpy.float32(-1000000)
        to = float32_to_fe4m3(x, saturate=False)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e4m3fnuz_out_of_range(self):
        x = numpy.float32(1000000)
        to = float32_to_fe4m3(x, uz=True)
        back = fe4m3_to_float32(to, uz=True)
        self.assertEqual(back, 240)

        x = numpy.float32(1000000)
        to = float32_to_fe4m3(x, uz=True, saturate=False)
        back = fe4m3_to_float32(to, uz=True)
        self.assertTrue(numpy.isnan(back))

        x = numpy.float32(-1000000)
        to = float32_to_fe4m3(x, uz=True)
        back = fe4m3_to_float32(to, uz=True)
        self.assertEqual(back, -240)

        x = numpy.float32(-1000000)
        to = float32_to_fe4m3(x, uz=True, saturate=False)
        back = fe4m3_to_float32(to, uz=True)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e5m2_out_of_range(self):
        x = numpy.float32(1000000)
        to = float32_to_fe5m2(x)
        back = fe5m2_to_float32(to)
        self.assertEqual(back, 57344)

        x = numpy.float32(1000000)
        to = float32_to_fe5m2(x, saturate=False)
        back = fe5m2_to_float32(to)
        self.assertTrue(numpy.isinf(back))

        x = numpy.float32(-1000000)
        to = float32_to_fe5m2(x)
        back = fe5m2_to_float32(to)
        self.assertEqual(back, -57344)

        x = numpy.float32(-1000000)
        to = float32_to_fe5m2(x, saturate=False)
        back = fe5m2_to_float32(to)
        self.assertTrue(numpy.isinf(back))

    def test_float8_e5m2fnuz_out_of_range(self):
        x = numpy.float32(1000000)
        to = float32_to_fe5m2(x, fn=True, uz=True)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 57344)

        x = numpy.float32(1000000)
        to = float32_to_fe5m2(x, fn=True, uz=True, saturate=False)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(numpy.isnan(back))

        x = numpy.float32(-1000000)
        to = float32_to_fe5m2(x, fn=True, uz=True)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, -57344)

        x = numpy.float32(-1000000)
        to = float32_to_fe5m2(x, fn=True, uz=True, saturate=False)
        back = fe5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e4m3fn_negative_zero(self):
        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe4m3(x)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, 0)

        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe4m3(x, saturate=False)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, 0)

    def test_float8_e4m3fnuz_negative_zero(self):
        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe4m3(x, uz=True)
        self.assertEqual(to, 0)
        back = fe4m3_to_float32(to, uz=True)
        self.assertEqual(back, 0)

        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe4m3(x, uz=True, saturate=False)
        back = fe4m3_to_float32(to, uz=True)
        self.assertEqual(back, 0)
        self.assertEqual(to, 0)

    def test_float8_e5m2_negative_zero(self):
        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe5m2(x)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, 0)

        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe5m2(x, saturate=False)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to)
        self.assertEqual(back, 0)

    def test_float8_e5m2fnuz_negative_zero(self):
        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe5m2(x, fn=True, uz=True)
        self.assertEqual(to, 0)
        back = fe4m3_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 0)

        x = fe5m2_to_float32(0x80)  # -0
        to = float32_to_fe5m2(x, fn=True, uz=True, saturate=False)
        self.assertEqual(to, 0)
        back = fe4m3_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 0)

    def test_float8_e4m3fn_negative_nan(self):
        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe4m3(x)
        self.assertEqual(to, 255)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe4m3(x, saturate=False)
        self.assertEqual(to, 255)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e4m3fnuz_negative_nan(self):
        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe4m3(x, uz=True)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to, uz=True)
        self.assertTrue(numpy.isnan(back))

        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe4m3(x, uz=True, saturate=False)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to, uz=True)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e5m2_negative_nan(self):
        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe5m2(x)
        self.assertEqual(to, 255)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe5m2(x, saturate=False)
        self.assertEqual(to, 255)
        back = fe4m3_to_float32(to)
        self.assertTrue(numpy.isnan(back))

    def test_float8_e5m2fnuz_negative_nan(self):
        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe5m2(x, fn=True, uz=True)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to, fn=True, uz=True)
        self.assertTrue(numpy.isnan(back))

        x = fe5m2_to_float32(255)  # -nan
        to = float32_to_fe5m2(x, fn=True, uz=True, saturate=False)
        self.assertEqual(to, 0x80)
        back = fe4m3_to_float32(to, fn=True, uz=True)
        self.assertTrue(numpy.isnan(back))

    def test_fe4m3fn_to_float32_bug(self):
        cases = [
            (0.00439453125, 0.00390625, TensorProto.FLOAT8E4M3FN),
            (0.005859375, 0.005859375, TensorProto.FLOAT8E4M3FN),
            (0.005759375, 0.005859375, TensorProto.FLOAT8E4M3FN),
            (0.0046875, 0.00390625, TensorProto.FLOAT8E4M3FN),
            (0.001953125, 0.001953125, TensorProto.FLOAT8E4M3FN),
            (0.0029296875, 0.00390625, TensorProto.FLOAT8E4M3FN),
            (0.002053125, 0.001953125, TensorProto.FLOAT8E4M3FN),
            (0.00234375, 0.001953125, TensorProto.FLOAT8E4M3FN),
            (0.0087890625, 0.0078125, TensorProto.FLOAT8E4M3FN),
            (0.001171875, 0.001953125, TensorProto.FLOAT8E4M3FN),
            (1.8131605, 1.875, TensorProto.FLOAT8E4M3FN),
            (-100, -96, TensorProto.FLOAT8E4M3FNUZ),
            (416, 384, TensorProto.FLOAT8E5M2FNUZ),
        ]
        for val, expected, pt in cases:
            with self.subTest(value=val, expected=expected, proto=pt):
                if pt == TensorProto.FLOAT8E4M3FN:
                    res = fe4m3_to_float32(search_float32_into_fe4m3(val))
                    self.assertEqual(expected, res)
                    res = fe4m3_to_float32(float32_to_fe4m3(val))
                    self.assertEqual(expected, res)
                    continue
                if pt == TensorProto.FLOAT8E4M3FNUZ:
                    res = fe4m3_to_float32(
                        search_float32_into_fe4m3(val, uz=True), uz=True
                    )
                    self.assertEqual(expected, res)
                    res = fe4m3_to_float32(float32_to_fe4m3(val, uz=True), uz=True)
                    self.assertEqual(expected, res)
                    continue
                if pt == TensorProto.FLOAT8E5M2FNUZ:
                    res = fe5m2_to_float32(
                        search_float32_into_fe5m2(val, fn=True, uz=True),
                        fn=True,
                        uz=True,
                    )
                    self.assertEqual(expected, res)
                    res = fe5m2_to_float32(
                        float32_to_fe5m2(val, fn=True, uz=True), fn=True, uz=True
                    )
                    self.assertEqual(expected, res)
                    continue
                raise AssertionError(f"Unexpected value for pt={pt}.")

    def test_inf(self):
        for x, e in [(numpy.float32(numpy.inf), 126), (numpy.float32(-numpy.inf), 254)]:
            f8 = float32_to_fe4m3(x)
            self.assertEqual(e, f8)

    def test_nan(self):
        expected = 127
        values = [
            (
                None,
                int.from_bytes(struct.pack("<f", numpy.float32(numpy.nan)), "little"),
                numpy.float32(numpy.nan),
                expected,
            )
        ]
        for i in range(0, 23):
            v = 0x7F800000 | (1 << i)
            f = numpy.uint32(v).view(numpy.float32)
            values.append((i, v, f, expected))
            values.append((i, v, -f, expected | 128))

        for i, v, x, e in values:
            with self.subTest(x=x, e=e, h=hex(v), i=i):
                f8 = float32_to_fe4m3(x)
                self.assertEqual(e, f8)

    def test_negative_zero_uz(self):
        self.assertEqual(numpy.float32(-0.0), numpy.float32(0.0))
        self.assertEqual(float32_to_fe4m3(-0.00000001, fn=True, uz=False), 128)
        self.assertEqual(float32_to_fe4m3(0.00000001, fn=True, uz=True), 0)
        self.assertEqual(float32_to_fe4m3(-0.00000001, fn=True, uz=True), 0)
        self.assertEqual(float32_to_fe5m2(-0.00000001, fn=False, uz=False), 128)
        self.assertEqual(float32_to_fe5m2(0.00000001, fn=True, uz=True), 0)
        self.assertEqual(float32_to_fe5m2(-0.00000001, fn=True, uz=True), 0)
        self.assertEqual(float32_to_fe4m3(-0.0001, fn=True, uz=False), 128)
        self.assertEqual(float32_to_fe4m3(-0.0001, fn=True, uz=True), 0)
        self.assertEqual(search_float32_into_fe4m3(-0.0001, fn=True, uz=False), 128)
        self.assertEqual(search_float32_into_fe4m3(-0.0001, fn=True, uz=True), 0)
        self.assertEqual(search_float32_into_fe5m2(-0.000001, fn=False, uz=False), 128)
        self.assertEqual(search_float32_into_fe5m2(-0.000001, fn=True, uz=True), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
