"""Product reductions: a product of n lengths is a length**n."""

import numpy as np
import pytest

from QuantSI.allunits import metre, volt
from QuantSI.fundamentalunits import Quantity

LENGTHS = np.array([1.0, 2.0, 3.0]) * metre
GRID = np.arange(1.0, 7.0).reshape(2, 3) * metre


@pytest.mark.parametrize(
    ("expression", "value", "exponent"),
    [
        (lambda: np.multiply.reduce(LENGTHS), 6.0, 3),
        (lambda: LENGTHS.prod(), 6.0, 3),
        (lambda: np.prod(LENGTHS), 6.0, 3),
        (lambda: LENGTHS.prod(initial=2), 12.0, 3),  # initial is a plain number
        (lambda: LENGTHS.prod(where=[True, False, True]), 3.0, 2),
        (lambda: GRID.prod(), 720.0, 6),
        (lambda: GRID.prod(axis=0), [4.0, 10.0, 18.0], 2),
        (lambda: GRID.prod(axis=1), [6.0, 120.0], 3),
        (lambda: np.multiply.reduce(GRID), [4.0, 10.0, 18.0], 2),  # ufunc default: axis 0
        (lambda: np.multiply.reduce(GRID, axis=None), 720.0, 6),
        (lambda: GRID.prod(axis=1, keepdims=True), [[6.0], [120.0]], 3),
    ],
)
def test_product_dimensions(expression, value, exponent):
    result = expression()
    assert isinstance(result, Quantity)
    assert result.dim is (metre**exponent).dim
    np.testing.assert_allclose(np.asarray(result), value)


def test_mask_with_different_counts_per_result_is_refused():
    with pytest.raises(TypeError, match="different number of factors"):
        GRID.prod(axis=1, where=[[True, True, True], [True, False, True]])


def test_empty_product_is_dimensionless():
    assert (np.array([]) * metre).prod() == 1.0


def test_product_with_empty_result_is_dimensionless():
    result = (np.empty((2, 0)) * metre).prod(axis=0)
    assert type(result) is np.ndarray
    assert result.shape == (0,)


def test_sums_keep_their_dimension():
    assert LENGTHS.sum().dim is metre.dim
    assert np.add.reduce(GRID, axis=1).dim is metre.dim
    assert (LENGTHS * volt).prod().dim is ((metre * volt) ** 3).dim
