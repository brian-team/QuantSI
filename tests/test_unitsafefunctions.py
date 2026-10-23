"""Unit-aware replacements for NumPy functions (from Brian2)."""

import numpy as np
import pytest

from QuantSI import DimensionMismatchError, unitsafefunctions as unitsafe
from QuantSI.allunits import metre, msecond, mvolt, second

DIMENSIONLESS_FUNCTIONS = [
    (unitsafe.sin, np.sin),
    (unitsafe.sinh, np.sinh),
    (unitsafe.arcsin, np.arcsin),
    (unitsafe.arcsinh, np.arcsinh),
    (unitsafe.cos, np.cos),
    (unitsafe.cosh, np.cosh),
    (unitsafe.arccos, np.arccos),
    (unitsafe.arccosh, np.arccosh),
    (unitsafe.tan, np.tan),
    (unitsafe.tanh, np.tanh),
    (unitsafe.arctan, np.arctan),
    (unitsafe.arctanh, np.arctanh),
    (unitsafe.log, np.log),
    (unitsafe.log10, np.log10),
    (unitsafe.exp, np.exp),
    (unitsafe.expm1, np.expm1),
    (unitsafe.log1p, np.log1p),
]


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@pytest.mark.parametrize(("function", "numpy_function"), DIMENSIONLESS_FUNCTIONS)
def test_functions_of_dimensionless_values(function, numpy_function):
    for value in [3 * mvolt, np.array([1, 2]) * mvolt]:
        with pytest.raises(DimensionMismatchError):
            function(value)
    for value in [0.3, np.array([0.1, 0.2]), np.ones((2, 2)) * 0.5]:
        np.testing.assert_allclose(function(value), numpy_function(value))


@pytest.mark.parametrize(("function", "numpy_function"), DIMENSIONLESS_FUNCTIONS)
def test_code_generation_metadata(function, numpy_function):
    # Brian2's code generation reads these attributes.
    assert function._arg_units == [1]
    assert function._return_unit == 1
    assert function.__name__ == numpy_function.__name__


def test_exprel():
    assert unitsafe.exprel(0) == 1.0
    np.testing.assert_allclose(
        unitsafe.exprel(np.array([1e-20, 1.0, 1000.0])), [1.0, np.e - 1, np.inf]
    )
    assert unitsafe.exprel._arg_units == [1] and unitsafe.exprel._return_unit == 1
    with pytest.raises(DimensionMismatchError):
        unitsafe.exprel(3 * mvolt)


@pytest.mark.parametrize("value", [1, -1.0, 1e-20, 718.0])
def test_scalar_exprel_matches_array_result(value):
    scalar = unitsafe.exprel(value)
    array = unitsafe.exprel(np.array([value]))
    np.testing.assert_allclose(scalar, array[0])


def test_arange():
    # np.arange never sees the units of its arguments; this one does. With units,
    # the step needs them too (the default step is the plain number 1).
    result = unitsafe.arange(3 * msecond, step=1 * msecond)
    assert result.dim is second.dim
    np.testing.assert_allclose(np.asarray(result), [0, 0.001, 0.002])
    result = unitsafe.arange(1 * msecond, 3 * msecond, 1 * msecond)
    np.testing.assert_allclose(np.asarray(result), [0.001, 0.002])
    np.testing.assert_array_equal(unitsafe.arange(5), np.arange(5))
    for arguments in [(1 * msecond, 3 * metre), (1 * msecond, 3 * msecond), (1, 3 * msecond)]:
        with pytest.raises(DimensionMismatchError):
            unitsafe.arange(*arguments)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((3,), {"stop": 4}, "Duplicate definition of 'stop'"),
        ((1, 3), {"start": 2}, "Duplicate definition of 'start'"),
        ((1, 3), {"stop": 4}, "Duplicate definition of 'stop'"),
        ((1, 3, 1), {"start": 2}, "Duplicate definition of 'start'"),
        ((1, 3, 1), {"stop": 4}, "Duplicate definition of 'stop'"),
        ((1, 3, 1), {"step": 2}, "Duplicate definition of 'step'"),
        ((1, 2, 3, 4), {}, "Need between 1 and 3 non-keyword arguments"),
        ((), {}, "Missing stop argument"),
    ],
)
def test_arange_rejects_invalid_arguments(args, kwargs, message):
    with pytest.raises(TypeError, match=message):
        unitsafe.arange(*args, **kwargs)


def test_arange_with_keyword_arguments():
    result = unitsafe.arange(start=1 * msecond, stop=4 * msecond, step=1 * msecond)
    assert result.dim is second.dim
    np.testing.assert_allclose(np.asarray(result), [0.001, 0.002, 0.003])


def test_linspace():
    result = unitsafe.linspace(0 * mvolt, 2 * mvolt, 3)
    assert result.dim is mvolt.dim
    np.testing.assert_allclose(np.asarray(result), [0, 0.001, 0.002])
    with pytest.raises(DimensionMismatchError):
        unitsafe.linspace(0 * mvolt, 2 * second, 3)


def test_where():
    condition = np.array([True, False])
    a, b = np.array([1.0, 2.0]) * mvolt, np.array([3.0, 4.0]) * mvolt
    result = unitsafe.where(condition, a, b)
    assert result.dim is mvolt.dim
    np.testing.assert_allclose(np.asarray(result), [0.001, 0.004])
    with pytest.raises(DimensionMismatchError):
        unitsafe.where(condition, a, np.ones(2) * second)


def test_where_without_units():
    condition = np.array([True, False, True])
    (indices,) = unitsafe.where(condition)
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(unitsafe.where(condition, [1, 2, 3], 0), [1, 0, 3])
    with pytest.raises(ValueError, match="either both or neither"):
        unitsafe.where(condition, [1, 2, 3])


def test_unit_free_and_unit_keeping_helpers():
    q = np.arange(4.0).reshape(2, 2) * mvolt
    ones = unitsafe.ones_like(q)
    assert not hasattr(ones, "dim") and ones.shape == (2, 2)
    assert unitsafe.ptp(q).dim is mvolt.dim
    assert unitsafe.ravel(q).dim is mvolt.dim
    assert unitsafe.diagonal(q).dim is mvolt.dim
    assert unitsafe.trace(q).dim is mvolt.dim
    assert unitsafe.dot(q, q).dim is (mvolt**2).dim


def test_method_wrapper_on_plain_arrays():
    values = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(unitsafe.diagonal(values, offset=1), [1, 5])


def test_dimension_changing_wrapper():
    from QuantSI.fundamentalunits import wrap_function_change_dimensions

    def mean_square(values, axis=None, *, keepdims=False):
        """Average the squared values along an axis."""
        assert type(values) is np.ndarray
        return np.mean(values**2, axis=axis, keepdims=keepdims)

    def squared_dimensions(values, dim):
        assert type(values) is np.ndarray
        return dim**2

    wrapped = wrap_function_change_dimensions(mean_square, squared_dimensions)
    values = np.array([[1.0, 2.0], [3.0, 4.0]]) * metre
    result = wrapped(values, 1, keepdims=True)
    assert result.dim is metre.dim**2
    np.testing.assert_allclose(np.asarray(result), [[2.5], [12.5]])
    assert wrapped._arg_units == [None]
    assert wrapped._return_unit is squared_dimensions
    assert wrapped.__name__ == mean_square.__name__
    assert wrapped.__doc__ == mean_square.__doc__
    assert wrapped._do_not_run_doctests is True
