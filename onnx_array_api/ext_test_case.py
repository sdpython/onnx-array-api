import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional
import numpy
from numpy.testing import assert_allclose


def is_azure() -> bool:
    "Tells if the job is running on Azure DevOps."
    return os.environ.get("AZURE_HTTP_USER_AGENT", "undefined") != "undefined"


def is_windows() -> bool:
    return sys.platform == "win32"


def skipif_ci_windows(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_windows() and is_azure():
        msg = f"Test does not work on azure pipeline (linux). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns: warnings to ignore
    """

    def wrapper(fct):
        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def matplotlib_test() -> Callable:
    """
    Decorator for every test checking matplotlib graphs.
    Cleans matplotlib after its completion.
    """

    def wrapper(fct):
        import matplotlib

        def call_f(self):
            orig_units_registry = matplotlib.units.registry.copy()
            try:
                return fct(self)
            finally:
                matplotlib.units.registry.clear()
                matplotlib.units.registry.update(orig_units_registry)
                matplotlib.pyplot.close("all")

        return call_f

    return wrapper


def example_path(path: str) -> str:
    """
    Fixes a path for the examples.
    Helps running the example within a unit test.
    """
    if os.path.exists(path):
        return path
    this = os.path.abspath(os.path.dirname(__file__))
    full = os.path.normpath(os.path.join(this, "..", "_doc", "examples", path))
    if os.path.exists(full):
        return full
    raise FileNotFoundError(f"Unable to find path {path!r} or {full!r}.")


def measure_time(
    stmt: Callable,
    context: Optional[Dict[str, Any]] = None,
    repeat: int = 10,
    number: int = 50,
    div_by_number: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param div_by_number: divide by the number of executions
    :param max_time: execute the statement until the total goes
        beyond this time (approximatively), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    .. runpython::
        :showcode:

        from onnx_array_api.ext_test_case import measure_time
        from math import cos

        res = measure_time(lambda: cos(0.5))
        print(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.

    .. versionchanged:: 0.4
        Parameter *max_time* was added.
    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(
            f"stmt is not callable or a string but is of type {type(stmt)!r}."
        )
    if context is None:
        context = {}

    import numpy

    if isinstance(stmt, str):
        tim = Timer(stmt, globals=context)
    else:
        tim = Timer(stmt)

    if max_time is not None:
        if not div_by_number:
            raise ValueError(
                "div_by_number must be set to True of max_time is defined."
            )
        i = 1
        total_time = 0
        results = []
        while True:
            for j in (1, 2):
                number = i * j
                time_taken = tim.timeit(number)
                results.append((number, time_taken))
                total_time += time_taken
                if total_time >= max_time:
                    break
            if total_time >= max_time:
                break
            ratio = (max_time - total_time) / total_time
            ratio = max(ratio, 1)
            i = int(i * ratio)

        res = numpy.array(results)
        tw = res[:, 0].sum()
        ttime = res[:, 1].sum()
        mean = ttime / tw
        ave = res[:, 1] / res[:, 0]
        dev = (((ave - mean) ** 2 * res[:, 0]).sum() / tw) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(ave),
            max_exec=numpy.max(ave),
            repeat=1,
            number=tw,
            ttime=ttime,
        )
    else:
        res = numpy.array(tim.repeat(repeat=repeat, number=number))
        if div_by_number:
            res /= number

        mean = numpy.mean(res)
        dev = numpy.mean(res**2)
        dev = (dev - mean**2) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(res),
            max_exec=numpy.max(res),
            repeat=repeat,
            number=number,
            ttime=res.sum(),
        )

    if "values" in context:
        if hasattr(context["values"], "shape"):
            mes["size"] = context["values"].shape[0]
        else:
            mes["size"] = len(context["values"])
    else:
        mes["context_size"] = sys.getsizeof(context)
    return mes


class ExtTestCase(unittest.TestCase):
    _warns = []

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertRaise(self, fct: Callable, exc_type: Exception):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertHasAttr(self, cls: type, name: str):
        if not hasattr(cls, name):
            raise AssertionError(f"Class {cls} has no attribute {name!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if value:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}")

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout):
            with redirect_stderr(serr):
                res = fct()
        return res, sout.getvalue(), serr.getvalue()

    def relative_path(self, filename: str, *names: List[str]) -> str:
        """
        Returns a path relative to the folder *filename*
        is in. The function checks the path existence.

        :param filename: filename
        :param names: additional path pieces
        :return: new path
        """
        dir = os.path.abspath(os.path.dirname(filename))
        name = os.path.join(dir, *names)
        if not os.path.exists(name):
            raise FileNotFoundError(f"Path {name!r} does not exists.")
        return name
