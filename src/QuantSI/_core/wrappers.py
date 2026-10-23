"""Wrappers that make a NumPy function unit-aware, for QuantSI.unitsafefunctions.

Each wrapper attaches the ``_arg_units``/``_return_unit`` metadata that Brian2's
code generation reads (as `check_units` does). From Brian2's fundamentalunits.

In this file: ``wrap_function_dimensionless``, ``wrap_function_change_dimensions``
and ``wrap_function_remove_dimensions``.
"""

import numpy as np

from .dimension import fail_for_dimension_mismatch
from .quantity import Quantity


def wrap_function_dimensionless(func):
    """
    Returns a new function that wraps the given function `func` so that it
    raises a DimensionMismatchError if the function is called on a quantity
    with dimensions (excluding dimensionless quantities). Quantities are
    transformed to unitless numpy arrays before calling `func`.

    These checks/transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(
            x,
            error_message=("%s expects a dimensionless argument but got {value}" % func.__name__),
            value=x,
        )
        return func(np.asarray(x), *args, **kwds)

    f._arg_units = [1]
    f._return_unit = 1
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


def wrap_function_change_dimensions(func, change_dim_func):
    """
    Returns a new function that wraps the given function `func` so that it
    changes the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling `func`, the output is a quantity
    with the original dimensions passed through the function
    `change_dim_func`. A typical use would be a ``sqrt`` function that uses
    ``lambda d: d ** 0.5`` as ``change_dim_func``.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        ar = np.asarray(x)
        return Quantity(func(ar, *args, **kwds), dim=change_dim_func(ar, x.dim))

    f._arg_units = [None]
    f._return_unit = change_dim_func
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


def wrap_function_remove_dimensions(func):
    """
    Returns a new function that wraps the given function `func` so that it
    removes any dimensions from its input. Useful for functions that are
    returning integers (indices) or booleans, irrespective of the datatype
    contained in the array.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        return func(np.asarray(x), *args, **kwds)

    f._arg_units = [None]
    f._return_unit = 1
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f
