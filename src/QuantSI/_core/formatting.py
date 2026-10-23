"""Turning quantities into text.

`format_quantity` writes a quantity in a display unit; ``str``, ``repr``,
``in_unit`` and ``in_best_unit`` all go through it. Explicit numeric format
specifications use NumPy's formatting of the raw SI value instead.

In this file: ``format_quantity`` (the core) and its helper ``_with_unit``,
``format_quantity_latex``, and the module-level ``in_unit`` and ``in_best_unit``.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .dimension import fail_for_dimension_mismatch, is_dimensionless
from .quantity import Quantity
from .unit import Unit


def format_quantity(
    quantity: Quantity,
    unit: Quantity,
    precision: int | None = None,
    python_code: bool = False,
) -> str:
    """Write ``quantity`` in ``unit`` (which must have the same dimensions).

    With ``python_code=False`` the text is for people (``'3. mV'``); with
    ``python_code=True`` it is an expression that evaluates back to the quantity
    (``'3. * mvolt'``). NumPy formats the numbers, following its print options;
    ``precision`` overrides the print option of that name.
    """
    value = np.asarray(quantity / unit)
    if value.shape == ():
        # A scalar is written as a one-element array without its brackets: NumPy
        # scalars ignore the print options, and 0-d arrays are written with the
        # scalar's repr in NumPy's legacy print mode (legacy="1.13", which Brian2's
        # tests use). One-element arrays follow the print options in every mode.
        text = np.array2string(value.reshape(1), precision=precision)[1:-1]
    elif python_code:
        text = np.array_repr(value, precision=precision)
    else:
        text = np.array_str(value, precision=precision)
    return _with_unit(text.strip(), quantity, unit, python_code)


def _with_unit(text: str, quantity: Quantity, unit: Quantity, python_code: bool) -> str:
    """Append the unit to the text of the numbers (or mark a dimensionless value)."""
    if unit.is_dimensionless:
        # A value without unit, made recognisable in Python code.
        return f"{type(quantity).__name__}({text})" if python_code else text
    name = unit if isinstance(unit, Unit) else unit.dim
    return f"{text} * {name!r}" if python_code else f"{text} {name!s}"


def format_quantity_latex(quantity: Quantity) -> str:
    """Write a scalar, 1-d or 2-d quantity as LaTeX, in its best unit.

    Used as the "rich representation" in Jupyter notebooks. The numbers are
    formatted with `numpy.array2string` and therefore follow NumPy's print options
    such as ``precision``. Writing every number of a large array is rarely useful,
    so the ``threshold`` print option is divided by 100 (the default of 1000
    becomes 10). The ``max_line_width`` print option is ignored.
    """
    best_unit = quantity.get_best_unit()
    if isinstance(best_unit, Unit):
        best_unit_latex = best_unit._latex()
    else:  # A quantity
        best_unit_latex = best_unit.dimensions._latex()
    unitless = np.asarray(quantity / best_unit)
    threshold = np.get_printoptions()["threshold"] // 100
    number: float | str
    if unitless.ndim == 0:
        number = float(unitless)
    elif unitless.ndim == 1:
        array_str = np.array2string(
            unitless,
            separator=" & ",
            threshold=threshold,
            max_line_width=sys.maxsize,
        )
        # Replace [ and ]
        number = (
            r"\left[\begin{matrix}"
            + array_str[1:-1].replace("...", r"\dots")
            + r"\end{matrix}\right]"
        )
    elif unitless.ndim == 2:
        array_str = np.array2string(
            unitless,
            separator=" & ",
            threshold=threshold,
            max_line_width=sys.maxsize,
        )
        array_str = array_str[1:-1].replace("...", r"\dots")
        array_str = array_str.replace("[", "").replace("] &", r"\\").replace("]", "\n")
        lines = array_str.split("\n")
        n_cols = lines[0].count("&") + 1
        new_lines = []
        for line in lines:
            if line.strip() == r"\dots &":
                new_lines.append(" & ".join([r"\vdots"] * n_cols) + r"\\")
            else:
                new_lines.append(line)
        number = r"\left[\begin{matrix}" + "\n" + "\n".join(new_lines) + r"\end{matrix}\right]"
    else:
        raise NotImplementedError(
            f"Cannot create a LaTeX representation for a {unitless.ndim}-d matrix."
        )
    return f"{number}\\,{best_unit_latex}"


def in_unit(x: Any, u: Quantity, precision: int | None = None) -> str:
    """
    Display a value in a certain unit with a given precision.

    Parameters
    ----------
    x : {`Quantity`, array-like, number}
        The value to display
    u : {`Quantity`, `Unit`}
        The unit to display the value `x` in.
    precision : `int`, optional
        The number of digits of precision (in the given unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.

    Returns
    -------
    s : `str`
        A string representation of `x` in units of `u`.

    Examples
    --------
    >>> from QuantSI import *
    >>> in_unit(3 * volt, mvolt)
    '3000. mV'
    >>> in_unit(123123 * msecond, second, 2)
    '123.12 s'
    >>> in_unit(10 * uA / cm**2, nA / um**2)
    '1.e-04 nA/(um^2)'
    >>> in_unit(10 * mV, ohm * amp)
    '0.01 ohm A'
    >>> in_unit(10 * nS, ohm)  # doctest: +NORMALIZE_WHITESPACE
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Non-matching unit for method "in_unit",
    dimensions were (m^-2 kg^-1 s^3 A^2) (m^2 kg s^-3 A^-2)

    See Also
    --------
    Quantity.in_unit
    """
    if is_dimensionless(x):
        fail_for_dimension_mismatch(x, u, 'Non-matching unit for function "in_unit"')
        return str(np.asarray(x / u))
    else:
        return x.in_unit(u, precision=precision)


def in_best_unit(x: Any, precision: int | None = None) -> str:
    """
    Represent the value in the "best" unit.

    Parameters
    ----------
    x : {`Quantity`, array-like, number}
        The value to display
    precision : `int`, optional
        The number of digits of precision (in the best unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.

    Returns
    -------
    representation : `str`
        A string representation of this `Quantity`.

    Examples
    --------
    >>> from QuantSI.allunits import *
    >>> in_best_unit(0.00123456 * volt)
    '1.23456 mV'
    >>> in_best_unit(0.00123456 * volt, 2)
    '1.23 mV'
    >>> in_best_unit(0.123456)
    '0.123456'
    >>> in_best_unit(0.123456, 2)
    '0.12'

    See Also
    --------
    Quantity.in_best_unit
    """
    if is_dimensionless(x):
        if precision is None:
            precision = np.get_printoptions()["precision"]
        return str(np.round(x, precision))

    u = x.get_best_unit()
    return x.in_unit(u, precision=precision)
