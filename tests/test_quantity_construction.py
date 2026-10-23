"""The internal fast constructor must behave exactly like Quantity(values, dim=...)."""

import numpy as np
import pytest

from QuantSI._core.quantity import _new_quantity
from QuantSI.allunits import second, volt
from QuantSI.fundamentalunits import DIMENSIONLESS, Quantity

VALUES = [
    np.asarray(3.0),
    np.float64(3.0),
    np.float32(1.5),
    np.int64(4),
    np.bool_(True),
    np.arange(4.0),
    np.arange(6).reshape(2, 3),
    np.array([], dtype=float),
    np.arange(3, dtype=np.float32),
]


@pytest.mark.parametrize("dim", [DIMENSIONLESS, volt.dim], ids=["dimensionless", "volt"])
@pytest.mark.parametrize("values", VALUES, ids=lambda v: f"{type(v).__name__}-{np.shape(v)}")
def test_fast_constructor_matches_quantity(values, dim):
    fast = _new_quantity(values, dim)
    slow = Quantity(values, dim=dim)
    assert type(fast) is type(slow)
    assert np.asarray(fast).dtype == np.asarray(slow).dtype
    np.testing.assert_array_equal(np.asarray(fast), np.asarray(slow))
    if isinstance(slow, Quantity):
        assert fast.dim is slow.dim
    if isinstance(values, np.ndarray) and values.size > 0 and values.ndim > 0:
        # no copy: the result is a view of NumPy's result
        assert np.shares_memory(np.asarray(fast), values)


def test_non_iterable_array_like_scalar():
    class ArrayScalar:
        def __array__(self, dtype=None, copy=None):
            return np.array(3.5, dtype=dtype)

    result = Quantity(ArrayScalar())
    assert result.dim is DIMENSIONLESS
    assert result.shape == ()
    assert result.item() == 3.5


def test_dimensions_property_can_be_assigned():
    q = 3 * volt
    q.dimensions = second.dim
    assert q.dim is second.dim
    assert q.dimensions is second.dim
    assert np.asarray(q).item() == 3
