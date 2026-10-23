"""check_units: call-time checks and the metadata Brian2 reads."""

import inspect

import numpy as np
import pytest

from QuantSI import DimensionMismatchError, check_units
from QuantSI.allunits import amp, metre, ohm, second, volt


@check_units(i=amp, r=ohm, result=volt)
def voltage(i, r):
    """Ohm's law."""
    return i * r


def test_metadata_contract():
    # Brian2's code generation reads exactly these attributes.
    assert voltage._arg_names == ("i", "r")
    assert voltage._arg_units == [amp, ohm]
    assert voltage._return_unit is volt
    assert voltage._returns_bool is False
    assert voltage._orig_func.__name__ == "voltage"
    assert voltage._orig_arg_names == ("i", "r")


def test_wraps_keeps_name_doc_and_signature():
    assert voltage.__name__ == "voltage"
    assert voltage.__doc__ == "Ohm's law."
    assert list(inspect.signature(voltage).parameters) == ["i", "r"]


def test_checks_arguments_and_result():
    assert voltage(2 * amp, 3 * ohm) == 6 * volt
    with pytest.raises(DimensionMismatchError):
        voltage(2 * second, 3 * ohm)


def test_result_function_and_bool():
    @check_units(result=lambda d: d**2)
    def square(x):
        return x**2

    assert square(3 * metre) == 9 * metre**2

    @check_units(x=metre, result=bool)
    def is_long(x):
        return x > 1 * metre

    assert is_long(2 * metre) is np.True_
    with pytest.raises(TypeError):

        @check_units(result=bool)
        def not_bool():
            return 1

        not_bool()


def test_stacked_decorators_check_the_original_arguments():
    @check_units(x=metre)
    @check_units(x=metre)
    def double(x, factor=2):
        return x * factor

    assert double(1 * metre) == 2 * metre
    assert double._arg_names == ("x", "factor")
    with pytest.raises(DimensionMismatchError):
        double(1 * second)


def test_unitless_argument_rejects_non_numeric_values():
    @check_units(x=1)
    def identity(x):
        return x

    with pytest.raises(TypeError, match="Argument x is not a unitless value/array"):
        identity(object())


def test_argument_unit_reference_must_be_present():
    @check_units(x="missing")
    def identity(x):
        return x

    with pytest.raises(TypeError, match="there is no argument of that name"):
        identity(3 * metre)
