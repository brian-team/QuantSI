"""Arithmetic with Unit operands: units combine into units, everything else into quantities."""

import numpy as np
import pytest

from QuantSI.allunits import metre, mvolt, ohm, second, volt
from QuantSI.fundamentalunits import Quantity, Unit
from QuantSI.stdunits import mV

CASES = [
    # (expression, expected type, expected value in SI, expected dimension)
    (lambda: 3 * mvolt, Quantity, 0.003, mvolt.dim),
    (lambda: mvolt * 3, Quantity, 0.003, mvolt.dim),
    (lambda: np.array([1.0, 2.0]) * mvolt, Quantity, [0.001, 0.002], mvolt.dim),
    (lambda: (2 * ohm) * mvolt, Quantity, 0.002, (ohm * volt).dim),
    (lambda: 3 / mvolt, Quantity, 3000.0, (1 / volt).dim),
    (lambda: mvolt / 2, Quantity, 0.0005, mvolt.dim),
    (lambda: (4 * metre) / second, Quantity, 4.0, (metre / second).dim),
    (lambda: metre / second, Unit, 1.0, (metre / second).dim),
    (lambda: mvolt * ohm, Unit, 0.001, (volt * ohm).dim),
    (lambda: 1 / second, Unit, 1.0, (1 / second).dim),
]


@pytest.mark.parametrize(("expression", "kind", "value", "dim"), CASES)
def test_unit_arithmetic(expression, kind, value, dim):
    result = expression()
    assert type(result) is kind
    assert result.dim is dim
    np.testing.assert_allclose(np.asarray(result), value)


@pytest.mark.parametrize(
    ("left", "right"),
    [(volt, volt), (volt, mvolt), (mvolt, mV), (volt, ohm), (metre, second)],
)
def test_unit_inequality_is_the_opposite_of_equality(left, right):
    # Units compare by dimension and scale, so units of different kinds are simply
    # not equal (`volt != ohm` used to raise DimensionMismatchError instead).
    assert (left != right) is (not (left == right))


def test_unit_power_with_array_exponent():
    result = mvolt ** np.array([2.0])
    assert type(result) is Quantity
    assert result.dim is volt.dim**2
    np.testing.assert_allclose(np.asarray(result), [1e-6])


def test_unit_power_rejects_multiple_exponents():
    with pytest.raises(TypeError, match="Only length-1 arrays"):
        metre ** np.array([2.0, 3.0])


def test_unit_value_must_match_its_scale():
    with pytest.raises(AssertionError, match="Unit value has to be 10\\*\\*scale"):
        Unit(2.0, scale=0)


def test_unit_ufunc_reduction_is_not_supported():
    with pytest.raises(TypeError, match="NotImplemented"):
        np.add.reduce(metre)
