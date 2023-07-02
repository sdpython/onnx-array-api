import unittest
import warnings
from os import getenv
from functools import reduce
import numpy as np
from operator import mul
from hypothesis import given
from onnx_array_api.ext_test_case import ExtTestCase
from onnx_array_api.array_api import onnx_numpy as onxp
from hypothesis import strategies
from hypothesis.extra import array_api


def prod(seq):
    return reduce(mul, seq, 1)


@strategies.composite
def array_api_kwargs(draw, **kw):
    result = {}
    for k, strat in kw.items():
        if draw(strategies.booleans()):
            result[k] = draw(strat)
    return result


def shapes(xp, **kw):
    kw.setdefault("min_dims", 0)
    kw.setdefault("min_side", 0)

    def sh(x):
        return x

    return xp.array_shapes(**kw).filter(
        lambda shape: prod(i for i in sh(shape) if i)
        < TestHypothesisArraysApis.MAX_ARRAY_SIZE
    )


class TestHypothesisArraysApis(ExtTestCase):
    MAX_ARRAY_SIZE = 10000
    SQRT_MAX_ARRAY_SIZE = int(10000**0.5)
    VERSION = "2021.12"

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from numpy import array_api as xp

        api_version = getenv(
            "ARRAY_API_TESTS_VERSION",
            getattr(xp, "__array_api_version__", TestHypothesisArraysApis.VERSION),
        )
        cls.xps = array_api.make_strategies_namespace(xp, api_version=api_version)
        api_version = getenv(
            "ARRAY_API_TESTS_VERSION",
            getattr(onxp, "__array_api_version__", TestHypothesisArraysApis.VERSION),
        )
        cls.onxps = array_api.make_strategies_namespace(onxp, api_version=api_version)

    def test_strategies(self):
        self.assertNotEmpty(self.xps)
        self.assertNotEmpty(self.onxps)

    def test_scalar_strategies(self):
        dtypes = dict(
            integer_dtypes=self.xps.integer_dtypes(),
            uinteger_dtypes=self.xps.unsigned_integer_dtypes(),
            floating_dtypes=self.xps.floating_dtypes(),
            numeric_dtypes=self.xps.numeric_dtypes(),
            boolean_dtypes=self.xps.boolean_dtypes(),
            scalar_dtypes=self.xps.scalar_dtypes(),
        )

        dtypes_onnx = dict(
            integer_dtypes=self.onxps.integer_dtypes(),
            uinteger_dtypes=self.onxps.unsigned_integer_dtypes(),
            floating_dtypes=self.onxps.floating_dtypes(),
            numeric_dtypes=self.onxps.numeric_dtypes(),
            boolean_dtypes=self.onxps.boolean_dtypes(),
            scalar_dtypes=self.onxps.scalar_dtypes(),
        )

        for k, vnp in dtypes.items():
            vonxp = dtypes_onnx[k]
            anp = self.xps.arrays(dtype=vnp, shape=shapes(self.xps))
            aonxp = self.onxps.arrays(dtype=vonxp, shape=shapes(self.onxps))
            self.assertNotEmpty(anp)
            self.assertNotEmpty(aonxp)

        args_np = []

        xx = self.xps.arrays(dtype=dtypes["integer_dtypes"], shape=shapes(self.xps))
        kws = array_api_kwargs(dtype=strategies.none() | self.xps.scalar_dtypes())

        @given(
            x=xx,
            kw=kws,
        )
        def fctnp(x, kw):
            asa1 = np.asarray(x)
            asa2 = np.asarray(x, **kw)
            self.assertEqual(asa1.shape, asa2.shape)
            args_np.append((x, kw))

        fctnp()
        self.assertEqual(len(args_np), 100)

        args_onxp = []

        xshape = shapes(self.onxps)
        xx = self.onxps.arrays(dtype=dtypes_onnx["integer_dtypes"], shape=xshape)
        kws = array_api_kwargs(dtype=strategies.none() | self.onxps.scalar_dtypes())

        @given(x=xx, kw=kws)
        def fctonx(x, kw):
            asa = np.asarray(x.numpy())
            try:
                asp = onxp.asarray(x)
            except Exception as e:
                raise AssertionError(f"asarray fails with x={x!r}, asp={asa!r}.") from e
            try:
                self.assertEqualArray(asa, asp.numpy())
            except AssertionError as e:
                raise AssertionError(
                    f"x={x!r} kw={kw!r} asa={asa!r}, asp={asp!r}"
                ) from e
            if kw:
                try:
                    asp2 = onxp.asarray(x, **kw)
                except Exception as e:
                    raise AssertionError(
                        f"asarray fails with x={x!r}, kw={kw!r}, asp={asa!r}."
                    ) from e
                self.assertEqual(asp.shape, asp2.shape)
            args_onxp.append((x, kw))

        fctonx()
        self.assertEqual(len(args_onxp), len(args_np))

    def test_square_sizes_strategies(self):
        dtypes = dict(
            integer_dtypes=self.xps.integer_dtypes(),
            uinteger_dtypes=self.xps.unsigned_integer_dtypes(),
            floating_dtypes=self.xps.floating_dtypes(),
            numeric_dtypes=self.xps.numeric_dtypes(),
            boolean_dtypes=self.xps.boolean_dtypes(),
            scalar_dtypes=self.xps.scalar_dtypes(),
        )

        dtypes_onnx = dict(
            integer_dtypes=self.onxps.integer_dtypes(),
            uinteger_dtypes=self.onxps.unsigned_integer_dtypes(),
            floating_dtypes=self.onxps.floating_dtypes(),
            numeric_dtypes=self.onxps.numeric_dtypes(),
            boolean_dtypes=self.onxps.boolean_dtypes(),
            scalar_dtypes=self.onxps.scalar_dtypes(),
        )

        for k, vnp in dtypes.items():
            vonxp = dtypes_onnx[k]
            anp = self.xps.arrays(dtype=vnp, shape=shapes(self.xps))
            aonxp = self.onxps.arrays(dtype=vonxp, shape=shapes(self.onxps))
            self.assertNotEmpty(anp)
            self.assertNotEmpty(aonxp)

        args_np = []

        kws = array_api_kwargs(k=strategies.integers(), dtype=self.xps.numeric_dtypes())
        sqrt_sizes = strategies.integers(0, self.SQRT_MAX_ARRAY_SIZE)
        ncs = strategies.none() | sqrt_sizes

        @given(n_rows=sqrt_sizes, n_cols=ncs, kw=kws)
        def fctnp(n_rows, n_cols, kw):
            base = np.asarray(0)
            e = np.eye(n_rows, n_cols)
            self.assertNotEmpty(e.dtype)
            self.assertIsInstance(e, base.__class__)
            e = np.eye(n_rows, n_cols, **kw)
            self.assertNotEmpty(e.dtype)
            self.assertIsInstance(e, base.__class__)
            args_np.append((n_rows, n_cols, kw))

        fctnp()
        self.assertEqual(len(args_np), 100)

        args_onxp = []

        kws = array_api_kwargs(
            k=strategies.integers(), dtype=self.onxps.numeric_dtypes()
        )
        sqrt_sizes = strategies.integers(0, self.SQRT_MAX_ARRAY_SIZE)
        ncs = strategies.none() | sqrt_sizes

        @given(n_rows=sqrt_sizes, n_cols=ncs, kw=kws)
        def fctonx(n_rows, n_cols, kw):
            base = onxp.asarray(0)
            e = onxp.eye(n_rows, n_cols)
            self.assertIsInstance(e, base.__class__)
            self.assertNotEmpty(e.dtype)
            e = onxp.eye(n_rows, n_cols, **kw)
            self.assertNotEmpty(e.dtype)
            self.assertIsInstance(e, base.__class__)
            args_onxp.append((n_rows, n_cols, kw))

        fctonx()
        self.assertEqual(len(args_onxp), len(args_np))


if __name__ == "__main__":
    # cl = TestHypothesisArraysApis()
    # cl.setUpClass()
    # cl.test_scalar_strategies()
    # import logging

    # logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
