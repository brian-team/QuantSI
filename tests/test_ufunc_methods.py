"""ufunc methods (outer, accumulate, reduceat, at): allowed exactly when units allow."""

import numpy as np
import pytest

from QuantSI import DimensionMismatchError
from QuantSI.allunits import metre, second
from QuantSI.fundamentalunits import Quantity


@pytest.fixture
def lengths():
    return np.array([1.0, 2.0, 3.0]) * metre


def test_outer(lengths):
    times = np.array([1.0, 2.0]) * second
    product = np.multiply.outer(lengths, times)
    assert product.shape == (3, 2) and product.dim is (metre * second).dim
    assert np.add.outer(lengths, lengths).dim is metre.dim
    assert np.less.outer(lengths, lengths).dtype == bool
    with pytest.raises(DimensionMismatchError):
        np.add.outer(lengths, times)


def test_accumulate(lengths):
    total = np.add.accumulate(lengths)
    assert total.dim is metre.dim
    np.testing.assert_array_equal(np.asarray(total), [1.0, 3.0, 6.0])
    assert np.maximum.accumulate(lengths).dim is metre.dim
    assert np.cumsum(lengths).dim is metre.dim
    with pytest.raises(TypeError, match="partial results"):
        np.multiply.accumulate(lengths)
    np.testing.assert_array_equal(np.multiply.accumulate(np.array([1.0, 2.0, 3.0])), [1, 2, 6])


def test_reduceat(lengths):
    sums = np.add.reduceat(lengths, [0, 2])
    assert isinstance(sums, Quantity) and sums.dim is metre.dim
    np.testing.assert_array_equal(np.asarray(sums), [3.0, 3.0])
    with pytest.raises(TypeError, match="partial results"):
        np.multiply.reduceat(lengths, [0, 2])


def test_at(lengths):
    np.add.at(lengths, [0, 0], 1 * metre)
    np.testing.assert_array_equal(np.asarray(lengths), [3.0, 2.0, 3.0])
    np.multiply.at(lengths, [1], 10)
    np.negative.at(lengths, [2])
    np.testing.assert_array_equal(np.asarray(lengths), [3.0, 20.0, -3.0])
    assert lengths.dim is metre.dim


def test_at_refuses_changing_some_elements(lengths):
    with pytest.raises(DimensionMismatchError):
        np.add.at(lengths, [0], 1 * second)
    with pytest.raises(TypeError, match="only some elements"):
        np.multiply.at(lengths, [0], 2 * second)
    np.testing.assert_array_equal(np.asarray(lengths), [1.0, 2.0, 3.0])
