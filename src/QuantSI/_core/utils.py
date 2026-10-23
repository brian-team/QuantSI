"""Small helpers shared by QuantSI's internal modules.

In this file:

- ``set_module`` and ``numpy_docstring``: decorators that set a function's
  ``__module__`` (pickles record it) and give a method NumPy's docstring
- ``_flatten`` and ``_short_str``: flatten nested lists, and short text for error
  messages
- ``_latex_number``: write a number in LaTeX the way SymPy did
"""

import math
import numbers
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

_Function = TypeVar("_Function", bound=Callable[..., Any])


def set_module(module: str) -> Callable[[_Function], _Function]:
    """Decorator that makes a function report ``module`` as its ``__module__``.

    pickle saves a function as its ``__module__`` and ``__qualname__``, and imports
    it from there again when loading. Two functions that QuantSI's pickles refer to
    used to live in ``QuantSI.fundamentalunits``. With this decorator they keep that
    path after moving to a private module, so older releases and Brian2 can still
    read new pickles. NumPy does the same (``numpy._utils.set_module``).
    """

    def decorator(func: _Function) -> _Function:
        func.__module__ = module
        return func

    return decorator


def numpy_docstring(numpy_function: Callable[..., Any]) -> Callable[[_Function], _Function]:
    """Decorator: give a method the docstring of the NumPy function it overrides.

    The docstring's examples are about NumPy arrays, so the method is also marked
    ``_do_not_run_doctests`` (the test suite, like Brian2's, skips its doctests).
    """

    def decorate(method: _Function) -> _Function:
        method.__doc__ = numpy_function.__doc__
        setattr(method, "_do_not_run_doctests", True)  # for tools, not type checkers
        return method

    return decorate


def _flatten(iterable):
    """
    Flatten a given list `iterable`.
    """
    for e in iterable:
        if isinstance(e, list):
            yield from _flatten(e)
        else:
            yield e


def _short_str(arr):
    """
    Return a short string representation of an array, suitable for use in
    error messages.
    """
    arr = np.asanyarray(arr)
    old_printoptions = np.get_printoptions()
    np.set_printoptions(edgeitems=2, threshold=5)
    arr_string = str(arr)
    np.set_printoptions(**old_printoptions)
    return arr_string


def _latex_number(value):
    r"""Write a number in LaTeX, the way SymPy's ``latex()`` prints Python numbers.

    Integers print as digits; floats with 15 significant digits, always with a
    decimal point (``2.0``), and in scientific notation as ``1.0 \cdot 10^{-20}``.
    QuantSI used to call SymPy for this (only for unit exponents such as the 3 in
    ``metre ** 3``), which made SymPy a required dependency and about 80% of
    QuantSI's import time.
    """
    if isinstance(value, numbers.Integral):
        return str(int(value))
    value = float(value)
    if math.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    if math.isnan(value):
        return r"\text{NaN}"
    if value == 0:
        value = 0.0  # SymPy prints -0.0 as 0.0
    mantissa, _, exponent = format(value, ".15g").partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    if not exponent:
        return mantissa
    return rf"{mantissa} \cdot 10^{{{int(exponent)}}}"
