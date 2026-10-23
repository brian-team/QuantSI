"""ufuncs with ``out=``: the output's label always matches its contents."""

import numpy as np
import pytest

from QuantSI import DimensionMismatchError
from QuantSI.allunits import metre, mvolt, ohm, second, volt
from QuantSI.fundamentalunits import DIMENSIONLESS, Quantity


@pytest.fixture
def arrays():
    return np.array([1.0, 2.0]) * volt, np.array([3.0, 4.0]) * volt, np.array([5.0, 6.0]) * ohm


def test_output_with_other_dimensions_is_relabelled(arrays):
    a, b, _ = arrays
    out = np.zeros(2) * second
    result = np.add(a, b, out=out)
    assert result is out
    assert out.dim is volt.dim
    np.testing.assert_array_equal(np.asarray(out), [4.0, 6.0])


def test_inplace_operator_changing_dimensions(arrays):
    a, _, r = arrays
    original = a
    a *= r
    assert a is original
    assert a.dim is (volt * ohm).dim
    np.testing.assert_array_equal(np.asarray(a), [5.0, 12.0])


def test_inplace_operator_keeping_dimensions(arrays):
    a, b, _ = arrays
    original = a
    a += b
    assert a is original and a.dim is volt.dim
    np.testing.assert_array_equal(np.asarray(a), [4.0, 6.0])


def test_mismatch_raises_before_writing(arrays):
    a, _, _ = arrays
    with pytest.raises(DimensionMismatchError):
        a += 1 * second
    assert a.dim is volt.dim
    np.testing.assert_array_equal(np.asarray(a), [1.0, 2.0])


def test_dimensionless_result_into_quantity_output(arrays):
    a, b, _ = arrays
    out = np.zeros(2) * metre
    np.divide(a, b, out=out)
    assert out.dim is DIMENSIONLESS


def test_scalar_quantities_behave_like_floats():
    x = 3 * mvolt
    y = x
    x += 2 * mvolt
    assert x is not y
    assert y == 3 * mvolt and x == 5 * mvolt


def test_plain_ndarray_output_receives_si_values(arrays):
    a, _, r = arrays
    out = np.zeros(2)
    result = np.multiply(a, r, out=out)
    np.testing.assert_array_equal(out, [5.0, 12.0])
    assert isinstance(result, Quantity) and result.dim is (volt * ohm).dim
    assert np.shares_memory(np.asarray(result), out)


def test_units_cannot_be_outputs():
    with pytest.raises(TypeError, match="in-place"):
        np.multiply(2.0, 3.0, out=mvolt)
    assert float(mvolt) == 0.001


def test_numpy_internals_that_write_in_place(arrays):
    # np.var squares a temporary in place: the relabelling must keep it right.
    a, _, _ = arrays
    assert a.var().dim is (volt**2).dim
    assert a.std().dim is volt.dim
    np.testing.assert_allclose(np.asarray(a.std()), np.std([1.0, 2.0]))
