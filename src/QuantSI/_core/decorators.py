"""The check_units decorator.

Besides checking units when the function is called, check_units stores some
information on the function it wraps (``_arg_units``, ``_arg_names``,
``_return_unit``, ``_returns_bool``, ``_orig_func``, ``_orig_arg_names``).
Brian2's code generation reads these attributes, so don't rename them.

In this file: ``check_units``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from .dimension import get_dimensions, have_same_dimensions
from .errors import DimensionMismatchError
from .quantity import Quantity
from .registry import get_unit_for_display

_Function = TypeVar("_Function", bound=Callable[..., Any])


def check_units(**au: object) -> Callable[[_Function], _Function]:
    """Decorator to check units of arguments passed to a function

    Examples
    --------
    >>> from QuantSI import mV, nA
    >>> from QuantSI.allunits import *
    >>> @check_units(I=amp, R=ohm, wibble=metre, result=volt)
    ... def getvoltage(I, R, **k):
    ...     return I * R

    You don't have to check the units of every variable in the function, and
    you can define what the units should be for variables that aren't
    explicitly named in the definition of the function. For example, the code
    above checks that the variable wibble should be a length, so writing

    >>> getvoltage(1 * amp, 1 * ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

    fails, but

    >>> getvoltage(1 * amp, 1 * ohm, wibble=1 * metre)
    1. * volt

    passes. String arguments or ``None`` are not checked

    >>> getvoltage(1 * amp, 1 * ohm, wibble="hello")
    1. * volt

    By using the special name ``result``, you can check the return value of the
    function.

    You can also use ``1`` or ``bool`` as a special value to check for a
    unitless number or a boolean value, respectively:

    >>> @check_units(value=1, absolute=bool, result=bool)
    ... def is_high(value, absolute=False):
    ...     if absolute:
    ...         return abs(value) >= 5
    ...     else:
    ...         return value >= 5

    This will then again raise an error if the argument if not of the expected
    type:

    >>> is_high(7)
    True
    >>> is_high(-7, True)
    True
    >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

    If the return unit depends on the unit of an argument, you can also pass
    a function that takes the units of all the arguments as its inputs (in the
    order specified in the function header):

    >>> @check_units(result=lambda d: d**2)
    ... def square(value):
    ...     return value**2

    If several arguments take arbitrary units but they have to be
    consistent among each other, you can state the name of another argument as
    a string to state that it uses the same unit as that argument.

    >>> @check_units(summand_1=None, summand_2="summand_1")
    ... def multiply_sum(multiplicand, summand_1, summand_2):
    ...     "Calculates multiplicand*(summand_1 + summand_2)"
    ...     return multiplicand * (summand_1 + summand_2)
    >>> multiply_sum(3, 4 * mV, 5 * mV)
    27. * mvolt
    >>> multiply_sum(3 * nA, 4 * mV, 5 * mV)
    27. * pwatt
    >>> multiply_sum(3 * nA, 4 * mV, 5 * nA)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    QuantSI.fundamentalunits.DimensionMismatchError: Function 'multiply_sum' expected the same arguments for arguments 'summand_1', 'summand_2', but argument 'summand_1' has unit V, while argument 'summand_2' has unit A.

    Raises
    ------

    DimensionMismatchError
        In case the input arguments or the return value do not have the
        expected dimensions.
    TypeError
        If an input argument or return value was expected to be a boolean but
        is not.

    Notes
    -----
    This decorator will destroy the signature of the original function, and
    replace it with the signature ``(*args, **kwds)``. Other decorators will
    do the same thing, and this decorator critically needs to know the signature
    of the function it is acting on, so it is important that it is the first
    decorator to act on a function. It cannot be used in combination with
    another decorator that also needs to know the signature of the function.

    Note that the ``bool`` type is "strict", i.e. it expects a proper
    boolean value and does not accept 0 or 1. This is not the case the other
    way round, declaring an argument or return value as "1" *does* allow for a
    ``True`` or ``False`` value.
    """

    def do_check_units(f: _Function) -> _Function:
        # Everything that depends only on f and on the decorator's arguments is
        # worked out once, here, instead of on every call.
        n_positional = f.__code__.co_argcount
        positional_names = f.__code__.co_varnames[0:n_positional]
        result_unit = au.get("result")
        result_is_bool = "result" in au and result_unit in (bool, np.bool_)
        result_is_function = callable(result_unit) and not result_is_bool

        @functools.wraps(f)
        def new_f(*args, **kwds):
            newkeyset = kwds.copy()
            for n, v in zip(positional_names, args[0:n_positional]):
                if not isinstance(v, (Quantity, str, bool, np.bool_)) and v is not None and n in au:
                    try:
                        # allow e.g. to pass a Python list of values
                        v = Quantity(v)
                    except TypeError:
                        if have_same_dimensions(au[n], 1):
                            raise TypeError(f"Argument {n} is not a unitless value/array.")
                        else:
                            raise TypeError(
                                f"Argument '{n}' is not a quantity, "
                                "expected a quantity with dimensions "
                                f"{au[n]}"
                            )
                newkeyset[n] = v

            for k in newkeyset:
                # string variables are allowed to pass, the presumption is they
                # name another variable. None is also allowed, useful for
                # default parameters
                if (
                    k in au
                    and not isinstance(newkeyset[k], str)
                    and newkeyset[k] is not None
                    and au[k] is not None
                ):
                    if au[k] in (bool, np.bool_):
                        if not isinstance(newkeyset[k], (bool, np.bool_)):
                            value = newkeyset[k]
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected a boolean value "
                                f"for argument '{k}' but got "
                                f"'{value}'"
                            )
                            raise TypeError(error_message)
                    elif isinstance(au[k], str):
                        if au[k] not in newkeyset:
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected its argument to have the "
                                f"same units as argument '{k}', but "
                                "there is no argument of that name"
                            )
                            raise TypeError(error_message)
                        if not have_same_dimensions(newkeyset[k], newkeyset[au[k]]):
                            d1 = get_dimensions(newkeyset[k])
                            d2 = get_dimensions(newkeyset[au[k]])
                            error_message = (
                                f"Function '{f.__name__}' expected "
                                f"the argument '{k}' to have the same "
                                f"units as argument '{au[k]}', but "
                                f"argument '{k}' has "
                                f"unit {get_unit_for_display(d1)}, "
                                f"while argument '{au[k]}' "
                                f"has unit {get_unit_for_display(d2)}."
                            )
                            raise DimensionMismatchError(error_message)
                    elif not have_same_dimensions(newkeyset[k], au[k]):
                        unit = repr(au[k])
                        value = newkeyset[k]
                        error_message = (
                            f"Function '{f.__name__}' "
                            "expected a quantity with unit "
                            f"{unit} for argument '{k}' but got "
                            f"'{value}'"
                        )
                        raise DimensionMismatchError(error_message, get_dimensions(newkeyset[k]))

            result = f(*args, **kwds)
            if "result" in au:
                if result_is_function:
                    expected_result = result_unit(*[get_dimensions(a) for a in args])
                else:
                    expected_result = result_unit
                if result_is_bool:
                    if not isinstance(result, (bool, np.bool_)):
                        error_message = (
                            "The return value of function "
                            f"'{f.__name__}' was expected to be "
                            "a boolean value, but was of type "
                            f"{type(result)}"
                        )
                        raise TypeError(error_message)
                elif not have_same_dimensions(result, expected_result):
                    unit = get_unit_for_display(expected_result)
                    error_message = (
                        "The return value of function "
                        f"'{f.__name__}' was expected to have "
                        f"unit {unit} but was "
                        f"'{result}'"
                    )
                    raise DimensionMismatchError(error_message, get_dimensions(result))
            return result

        wrapper: Any = new_f  # carries the metadata attributes below
        wrapper._orig_func = f
        wrapper.__doc__ = f.__doc__
        wrapper.__name__ = f.__name__
        # store the information in the function, necessary when using the
        # function in expressions or equations
        if hasattr(f, "_orig_arg_names"):
            arg_names = f._orig_arg_names
        else:
            arg_names = f.__code__.co_varnames[: f.__code__.co_argcount]
        wrapper._arg_names = arg_names
        wrapper._arg_units = [au.get(name, None) for name in arg_names]
        return_unit = au.get("result", None)
        if return_unit is None:
            wrapper._return_unit = None
        else:
            wrapper._return_unit = return_unit
        if return_unit is bool:
            wrapper._returns_bool = True
        else:
            wrapper._returns_bool = False
        wrapper._orig_arg_names = arg_names

        # copy any annotation attributes
        if hasattr(f, "_annotation_attributes"):
            for attrname in f._annotation_attributes:
                setattr(wrapper, attrname, getattr(f, attrname))
        wrapper._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
            "_arg_units",
            "_arg_names",
            "_return_unit",
            "_orig_func",
            "_returns_bool",
        ]
        return wrapper

    return do_check_units
