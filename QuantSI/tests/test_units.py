import itertools
import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_equal

from QuantSI.allunits import *
from QuantSI.fundamentalunits import (
    DIMENSIONLESS,
    UFUNCS_DIMENSIONLESS,
    UFUNCS_DIMENSIONLESS_TWOARGS,
    UFUNCS_INTEGERS,
    UFUNCS_LOGICAL,
    DimensionMismatchError,
    Quantity,
    Unit,
    check_units,
    fail_for_dimension_mismatch,
    get_dimensions,
    get_or_create_dimension,
    get_unit,
    have_same_dimensions,
    in_unit,
    is_dimensionless,
    is_scalar_type,
    quantity_with_dimensions
)
from QuantSI.stdunits import Hz, cm, kHz, mM, ms, mV, nA, nS

def assert_quantity(q, values, unit):
    assert isinstance(q, Quantity) or (
        have_same_dimensions(unit, 1)
        and (values.shape == () or isinstance(q, np.ndarray))
    ), q
    assert np.allclose(np.asarray(q), values)
    assert have_same_dimensions(
        q, unit
    ), f"Dimension mismatch: ({get_dimensions(q)}) ({get_dimensions(unit)})"


def test_construction():
    """Test the construction of quantity objects"""
    q = 500 * ms
    assert_quantity(q, 0.5, second)
    q = np.float64(500) * ms
    assert_quantity(q, 0.5, second)
    q = np.array(500) * ms
    assert_quantity(q, 0.5, second)
    q = np.array([500, 1000]) * ms
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity(500)
    assert_quantity(q, 500, 1)
    q = Quantity(500, dim=second.dim)
    assert_quantity(q, 500, second)
    q = Quantity([0.5, 1], dim=second.dim)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity(np.array([0.5, 1]), dim=second.dim)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity([500 * ms, 1 * second])
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity.with_dimensions(np.array([0.5, 1]), second=1)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = [0.5, 1] * second
    assert_quantity(q, np.array([0.5, 1]), second)

    # dimensionless quantities
    q = Quantity([1, 2, 3])
    assert_quantity(q, np.array([1, 2, 3]), Unit(1))
    q = Quantity(np.array([1, 2, 3]))
    assert_quantity(q, np.array([1, 2, 3]), Unit(1))
    q = Quantity([])
    assert_quantity(q, np.array([]), Unit(1))

    # copying/referencing a quantity
    q1 = Quantity.with_dimensions(np.array([0.5, 1]), second=1)
    q2 = Quantity(q1)  # no copy
    assert_quantity(q2, np.asarray(q1), q1)
    q2[0] = 3 * second
    assert_equal(q1[0], 3 * second)

    q1 = Quantity.with_dimensions(np.array([0.5, 1]), second=1)
    q2 = Quantity(q1, copy=True)  # copy
    assert_quantity(q2, np.asarray(q1), q1)
    q2[0] = 3 * second
    assert_equal(q1[0], 0.5 * second)

    # Illegal constructor calls
    with pytest.raises(TypeError):
        Quantity([500 * ms, 1])
    with pytest.raises(TypeError):
        Quantity(["some", "nonsense"])
    with pytest.raises(DimensionMismatchError):
        Quantity([500 * ms, 1 * volt])


def test_get_dimensions():
    """
    Test various ways of getting/comparing the dimensions of a quantity.
    """
    q = 500 * ms
    assert get_dimensions(q) is get_or_create_dimension(q.dimensions._dims)
    assert get_dimensions(q) is q.dimensions
    assert q.has_same_dimensions(3 * second)
    dims = q.dimensions
    assert_equal(dims.get_dimension("time"), 1.0)
    assert_equal(dims.get_dimension("length"), 0)

    assert get_dimensions(5) is DIMENSIONLESS
    assert get_dimensions(5.0) is DIMENSIONLESS
    assert get_dimensions(np.array(5, dtype=np.int32)) is DIMENSIONLESS
    assert get_dimensions(np.array(5.0)) is DIMENSIONLESS
    assert get_dimensions(np.float32(5.0)) is DIMENSIONLESS
    assert get_dimensions(np.float64(5.0)) is DIMENSIONLESS
    assert is_scalar_type(5)
    assert is_scalar_type(5.0)
    assert is_scalar_type(np.array(5, dtype=np.int32))
    assert is_scalar_type(np.array(5.0))
    assert is_scalar_type(np.float32(5.0))
    assert is_scalar_type(np.float64(5.0))
    with pytest.raises(TypeError):
        get_dimensions("a string")
    # wrong number of indices
    with pytest.raises(TypeError):
        get_or_create_dimension([1, 2, 3, 4, 5, 6])
    # not a sequence
    with pytest.raises(TypeError):
        get_or_create_dimension(42)



def test_display():
    """
    Test displaying a quantity in different units
    """
    assert_equal(in_unit(3 * volt, mvolt), "3000. mV")
    assert_equal(in_unit(10 * mV, ohm * amp), "0.01 ohm A")
    with pytest.raises(DimensionMismatchError):
        in_unit(10 * nS, ohm)

    # A bit artificial...
    assert_equal(in_unit(10.0, Unit(10.0, scale=1)), "1.0")

def test_pickling():
    """
    Test pickling of units.
    """
    for q in [
        500 * mV,
        500 * mV / mV,
        np.arange(10) * mV,
        np.arange(12).reshape(4, 3) * mV / ms,
    ]:
        pickled = pickle.dumps(q)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, type(q))
        assert have_same_dimensions(unpickled, q)
        assert_equal(unpickled, q)



def test_dimension_singletons():
    # Make sure that Dimension objects are singletons, even when pickled
    volt_dim = get_or_create_dimension((2, 1, -3, -1, 0, 0, 0))
    assert volt.dim is volt_dim
    import pickle

    pickled_dim = pickle.dumps(volt_dim)
    unpickled_dim = pickle.loads(pickled_dim)
    assert unpickled_dim is volt_dim
    assert unpickled_dim is volt.dim



def test_str_repr():
    """
    Test that str representations do not raise any errors and that repr
    fullfills eval(repr(x)) == x. Also test generating LaTeX representations via sympy.
    """
    import sympy
    from numpy import array  # necessary for evaluating repr

    units_which_should_exist = [
        metre,
        meter,
        kilogram,
        kilogramme,
        second,
        amp,
        kelvin,
        mole,
        candle,
        radian,
        steradian,
        hertz,
        newton,
        pascal,
        joule,
        watt,
        coulomb,
        volt,
        farad,
        ohm,
        siemens,
        weber,
        tesla,
        henry,
        lumen,
        lux,
        becquerel,
        gray,
        sievert,
        katal,
        gram,
        gramme,
        molar,
        liter,
        litre,
    ]

    # scaled versions of all these units should exist (we just check farad as an example)
    some_scaled_units = [
        Yfarad,
        Zfarad,
        Efarad,
        Pfarad,
        Tfarad,
        Gfarad,
        Mfarad,
        kfarad,
        hfarad,
        dafarad,
        dfarad,
        cfarad,
        mfarad,
        ufarad,
        nfarad,
        pfarad,
        ffarad,
        afarad,
        zfarad,
        yfarad,
    ]

    # some powered units
    powered_units = [cmetre2, Yfarad3]

    # Combined units
    complex_units = [
        (kgram * metre2) / (amp * second3),
        5 * (kgram * metre2) / (amp * second3),
        metre * second**-1,
        10 * metre * second**-1,
        array([1, 2, 3]) * kmetre / second,
        np.ones(3) * nS / cm**2,
        # Made-up unit:
        Unit(
            1,
            dim=get_or_create_dimension(length=5, time=2),
            dispname="O",
            latexname=r"\Omega",
        ),
        8000 * umetre**3,
        [0.0001, 10000] * umetre**3,
        1 / metre,
        1 / (coulomb * metre**2),
        Unit(1) / second,
        3.0 * mM,
        5 * mole / liter,
        7 * liter / meter3,
        1 / second**2,
        volt**-2,
        (volt**2) ** -1,
        (1 / second) / meter,
        1 / (1 / second),
    ]

    unitless = [second / second, 5 * second / second, Unit(1)]

    for u in itertools.chain(
        units_which_should_exist,
        some_scaled_units,
        powered_units,
        complex_units,
        unitless,
    ):
        assert len(str(u)) > 0
        if not is_dimensionless(u):
            assert len(sympy.latex(u))
        assert get_dimensions(eval(repr(u))) == get_dimensions(u)
        np.allclose(np.asarray(eval(repr(u))), np.asarray(u))

    for ar in [np.arange(10000) * mV, np.arange(100).reshape(10, 10) * mV]:
        latex_str = sympy.latex(ar)
        assert 0 < len(latex_str) < 1000  # arbitrary threshold, but see #1425

    # test the `DIMENSIONLESS` object
    assert str(DIMENSIONLESS) == "1"
    assert repr(DIMENSIONLESS) == "Dimension()"

    # test DimensionMismatchError (only that it works without raising an error
    for error in [
        DimensionMismatchError("A description"),
        DimensionMismatchError("A description", DIMENSIONLESS),
        DimensionMismatchError("A description", DIMENSIONLESS, second.dim),
    ]:
        assert len(str(error))
        assert len(repr(error))



def test_format_quantity():
    # Avoid that the default f-string (or .format call) discards units when used without
    # a format spec
    q = 0.5 * ms
    assert f"{q}" == f"{q!s}" == str(q)
    assert f"{q:g}" == f"{float(q)}"



def test_slicing():
    # Slicing and indexing, setting items
    quantity = np.reshape(np.arange(6), (2, 3)) * mV
    assert_equal(quantity[:], quantity)
    assert_equal(quantity[0], np.asarray(quantity)[0] * volt)
    assert_equal(quantity[0:1], np.asarray(quantity)[0:1] * volt)
    assert_equal(quantity[0, 1], np.asarray(quantity)[0, 1] * volt)
    assert_equal(quantity[0:1, 1:], np.asarray(quantity)[0:1, 1:] * volt)
    bool_matrix = np.array([[True, False, False], [False, False, True]])
    assert_equal(quantity[bool_matrix], np.asarray(quantity)[bool_matrix] * volt)



def test_setting():
    quantity = np.reshape(np.arange(6), (2, 3)) * mV
    quantity[0, 1] = 10 * mV
    assert quantity[0, 1] == 10 * mV
    quantity[:, 1] = 20 * mV
    assert np.all(quantity[:, 1] == 20 * mV)
    quantity[1, :] = np.ones((1, 3)) * volt
    assert np.all(quantity[1, :] == 1 * volt)
    # Setting to zero should work without units as well
    quantity[1, 2] = 0
    assert quantity[1, 2] == 0 * mV

    def set_to_value(key, value):
        quantity[key] = value

    with pytest.raises(DimensionMismatchError):
        set_to_value(0, 1)
    with pytest.raises(DimensionMismatchError):
        set_to_value(0, 1 * second)
    with pytest.raises(DimensionMismatchError):
        set_to_value((slice(2), slice(3)), np.ones((2, 3)))



def test_multiplication_division():
    quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    q2 = 5 * second

    for q in quantities:
        # Scalars and array scalars
        assert_quantity(q / 3, np.asarray(q) / 3, volt)
        assert_quantity(3 / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * 3, np.asarray(q) * 3, volt)
        assert_quantity(3 * q, 3 * np.asarray(q), volt)
        assert_quantity(q / np.float64(3), np.asarray(q) / 3, volt)
        assert_quantity(np.float64(3) / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * np.float64(3), np.asarray(q) * 3, volt)
        assert_quantity(np.float64(3) * q, 3 * np.asarray(q), volt)
        assert_quantity(q / np.array(3), np.asarray(q) / 3, volt)
        assert_quantity(np.array(3) / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * np.array(3), np.asarray(q) * 3, volt)
        assert_quantity(np.array(3) * q, 3 * np.asarray(q), volt)

        # (unitless) arrays
        assert_quantity(q / np.array([3]), np.asarray(q) / 3, volt)
        assert_quantity(np.array([3]) / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * np.array([3]), np.asarray(q) * 3, volt)
        assert_quantity(np.array([3]) * q, 3 * np.asarray(q), volt)

        # arrays with units
        assert_quantity(q / q, np.asarray(q) / np.asarray(q), 1)
        assert_quantity(q * q, np.asarray(q) ** 2, volt**2)
        assert_quantity(q / q2, np.asarray(q) / np.asarray(q2), volt / second)
        assert_quantity(q2 / q, np.asarray(q2) / np.asarray(q), second / volt)
        assert_quantity(q * q2, np.asarray(q) * np.asarray(q2), volt * second)

        # using unsupported objects should fail
        with pytest.raises(TypeError):
            q / "string"
        with pytest.raises(TypeError):
            "string" / q
        with pytest.raises(TypeError):
            "string" * q
        with pytest.raises(TypeError):
            q * "string"



def test_addition_subtraction():
    quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    q2 = 5 * volt

    for q in quantities:
        # arrays with units
        assert_quantity(q + q, np.asarray(q) + np.asarray(q), volt)
        assert_quantity(q - q, 0, volt)
        assert_quantity(q + q2, np.asarray(q) + np.asarray(q2), volt)
        assert_quantity(q2 + q, np.asarray(q2) + np.asarray(q), volt)
        assert_quantity(q - q2, np.asarray(q) - np.asarray(q2), volt)
        assert_quantity(q2 - q, np.asarray(q2) - np.asarray(q), volt)

        # mismatching units
        with pytest.raises(DimensionMismatchError):
            q + 5 * second
        with pytest.raises(DimensionMismatchError):
            5 * second + q
        with pytest.raises(DimensionMismatchError):
            q - 5 * second
        with pytest.raises(DimensionMismatchError):
            5 * second - q

        # scalar
        with pytest.raises(DimensionMismatchError):
            q + 5
        with pytest.raises(DimensionMismatchError):
            5 + q
        with pytest.raises(DimensionMismatchError):
            q + np.float64(5)
        with pytest.raises(DimensionMismatchError):
            np.float64(5) + q
        with pytest.raises(DimensionMismatchError):
            q - 5
        with pytest.raises(DimensionMismatchError):
            5 - q
        with pytest.raises(DimensionMismatchError):
            q - np.float64(5)
        with pytest.raises(DimensionMismatchError):
            np.float64(5) - q

        # unitless array
        with pytest.raises(DimensionMismatchError):
            q + np.array([5])
        with pytest.raises(DimensionMismatchError):
            np.array([5]) + q
        with pytest.raises(DimensionMismatchError):
            q + np.array([5], dtype=np.float64)
        with pytest.raises(DimensionMismatchError):
            np.array([5], dtype=np.float64) + q
        with pytest.raises(DimensionMismatchError):
            q - np.array([5])
        with pytest.raises(DimensionMismatchError):
            np.array([5]) - q
        with pytest.raises(DimensionMismatchError):
            q - np.array([5], dtype=np.float64)
        with pytest.raises(DimensionMismatchError):
            np.array([5], dtype=np.float64) - q

        # Check that operations with 0 work
        assert_quantity(q + 0, np.asarray(q), volt)
        assert_quantity(0 + q, np.asarray(q), volt)
        assert_quantity(q - 0, np.asarray(q), volt)
        assert_quantity(0 - q, -np.asarray(q), volt)
        assert_quantity(q + np.float64(0), np.asarray(q), volt)
        assert_quantity(np.float64(0) + q, np.asarray(q), volt)
        assert_quantity(q - np.float64(0), np.asarray(q), volt)
        assert_quantity(np.float64(0) - q, -np.asarray(q), volt)

        # using unsupported objects should fail
        with pytest.raises(TypeError):
            "string" + q
        with pytest.raises(TypeError):
            q + "string"
        with pytest.raises(TypeError):
            q - "string"
        with pytest.raises(TypeError):
            "string" - q



def test_unary_operations():
    from operator import neg, pos

    for op in [neg, pos]:
        for x in [2, np.array([2]), np.array([1, 2])]:
            assert_quantity(op(x * kilogram), op(x), kilogram)



def test_binary_operations():
    """Test whether binary operations work when they should and raise
    DimensionMismatchErrors when they should.
    Does not test for the actual result.
    """
    from operator import add, eq, ge, gt, le, lt, ne, sub

    def assert_operations_work(a, b):
        try:
            # Test python builtins
            tryops = [add, sub, lt, le, gt, ge, eq, ne]
            for op in tryops:
                op(a, b)
                op(b, a)

            # Test equivalent numpy functions
            numpy_funcs = [
                np.add,
                np.subtract,
                np.less,
                np.less_equal,
                np.greater,
                np.greater_equal,
                np.equal,
                np.not_equal,
                np.maximum,
                np.minimum,
            ]
            for numpy_func in numpy_funcs:
                numpy_func(a, b)
                numpy_func(b, a)
        except DimensionMismatchError as ex:
            raise AssertionError(f"Operation raised unexpected exception: {ex}")

    def assert_operations_do_not_work(a, b):
        # Test python builtins
        tryops = [add, sub, lt, le, gt, ge, eq, ne]
        for op in tryops:
            with pytest.raises(DimensionMismatchError):
                op(a, b)
            with pytest.raises(DimensionMismatchError):
                op(b, a)

        # Test equivalent numpy functions
        numpy_funcs = [
            np.add,
            np.subtract,
            np.less,
            np.less_equal,
            np.greater,
            np.greater_equal,
            np.equal,
            np.not_equal,
            np.maximum,
            np.minimum,
        ]
        for numpy_func in numpy_funcs:
            with pytest.raises(DimensionMismatchError):
                numpy_func(a, b)
            with pytest.raises(DimensionMismatchError):
                numpy_func(b, a)

    #
    # Check that consistent units work
    #

    # unit arrays
    a = 1 * kilogram
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
        assert_operations_work(a, b)

    # dimensionless units and scalars
    a = 1
    for b in [
        2 * kilogram / kilogram,
        np.array([2]) * kilogram / kilogram,
        np.array([1, 2]) * kilogram / kilogram,
    ]:
        assert_operations_work(a, b)

    # dimensionless units and unitless arrays
    a = np.array([1])
    for b in [
        2 * kilogram / kilogram,
        np.array([2]) * kilogram / kilogram,
        np.array([1, 2]) * kilogram / kilogram,
    ]:
        assert_operations_work(a, b)

    #
    # Check that inconsistent units do not work
    #

    # unit arrays
    a = np.array([1]) * second
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
        assert_operations_do_not_work(a, b)

    # unitless array
    a = np.array([1])
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
        assert_operations_do_not_work(a, b)

    # scalar
    a = 1
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
        assert_operations_do_not_work(a, b)

    # Check that comparisons with inf/-inf always work
    values = [
        2 * kilogram / kilogram,
        2 * kilogram,
        np.array([2]) * kilogram,
        np.array([1, 2]) * kilogram,
    ]
    for value in values:
        assert np.all(value < np.inf)
        assert np.all(np.inf > value)
        assert np.all(value <= np.inf)
        assert np.all(np.inf >= value)
        assert np.all(value != np.inf)
        assert np.all(np.inf != value)
        assert np.all(value >= -np.inf)
        assert np.all(-np.inf <= value)
        assert np.all(value > -np.inf)
        assert np.all(-np.inf < value)



def test_power():
    """
    Test raising quantities to a power.
    """
    values = [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]
    for value in values:
        assert_quantity(value**3, np.asarray(value) ** 3, kilogram**3)
        # Test raising to a dimensionless quantity
        assert_quantity(value ** (3 * volt / volt), np.asarray(value) ** 3, kilogram**3)
        with pytest.raises(DimensionMismatchError):
            # FIXME: Not that if float(exponent) is a special value such as 1 or 2
            # numpy will actually use a ufunc such as identity or square, which will
            # not raise a DimensionMismatchError. This is a limitation of the current
            # implementation.
            value ** (2 * mV)
        with pytest.raises(TypeError):
            value ** np.array([2, 3])



def test_inplace_operations():
    q = np.arange(10) * volt
    q_orig = q.copy()
    q_ref = q

    q *= 2
    assert np.array_equal(q, 2 * q_orig)
    assert np.array_equal(q_ref, q)
    q /= 2
    assert np.array_equal(q, q_orig)
    assert np.array_equal(q_ref, q)
    q += 1 * volt
    assert np.array_equal(q, q_orig + 1 * volt)
    assert np.array_equal(q_ref, q)
    q -= 1 * volt
    assert np.array_equal(q, q_orig)
    assert np.array_equal(q_ref, q)
    q **= 2
    assert np.array_equal(q, q_orig**2)
    assert np.array_equal(q_ref, q)
    q **= 0.5
    assert np.array_equal(q, q_orig)
    assert np.array_equal(q_ref, q)

    def illegal_add(q2):
        q = np.arange(10) * volt
        q += q2

    with pytest.raises(DimensionMismatchError):
        illegal_add(1 * second)
    with pytest.raises(DimensionMismatchError):
        illegal_add(1)

    def illegal_sub(q2):
        q = np.arange(10) * volt
        q -= q2

    with pytest.raises(DimensionMismatchError):
        illegal_add(1 * second)
    with pytest.raises(DimensionMismatchError):
        illegal_add(1)

    def illegal_pow(q2):
        q = np.arange(10) * volt
        q **= q2

    with pytest.raises(DimensionMismatchError):
        illegal_pow(1 * mV)
    with pytest.raises(TypeError):
        illegal_pow(np.arange(10))

    # inplace operations with unsupported objects should fail
    for inplace_op in [
        q.__iadd__,
        q.__isub__,
        q.__imul__,
        q.__itruediv__,
        q.__ifloordiv__,
        q.__imod__,
        q.__ipow__,
    ]:
        try:
            result = inplace_op("string")
            # if it doesn't fail with an error, it should return NotImplemented
            assert result == NotImplemented
        except TypeError:
            pass  # raised on numpy >= 0.10

    # make sure that inplace operations do not work on units/dimensions at all
    for inplace_op in [
        volt.__iadd__,
        volt.__isub__,
        volt.__imul__,
        volt.__itruediv__,
        volt.__ifloordiv__,
        volt.__imod__,
        volt.__ipow__,
    ]:
        with pytest.raises(TypeError):
            inplace_op(volt)
    for inplace_op in [
        volt.dimensions.__imul__,
        volt.dimensions.__itruediv__,
        volt.dimensions.__ipow__,
    ]:
        with pytest.raises(TypeError):
            inplace_op(volt.dimensions)



def test_unit_discarding_functions():
    """
    Test functions that discard units.
    """

    values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(3, 4) * mV]
    for value in values:
        assert_equal(np.sign(value), np.sign(np.asarray(value)))
        assert_equal(np.zeros_like(value), 
                    Quantity(np.zeros_like(np.asarray(value)),
                            dim=value.dim))
        assert_equal(np.ones_like(value), 
                     Quantity(np.ones_like(np.asarray(value)),
                            dim=value.dim))
        # Calling non-zero on a 0d array is deprecated, don't test it:
        if value.ndim > 0:
            assert_equal(np.nonzero(value), np.nonzero(np.asarray(value)))

def test_special_case_numpy_functions():
    """
    Test a couple of functions/methods that need special treatment.
    """

    quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Check that function and method do the same thing
        assert_equal(np.ravel(quadratic_matrix), quadratic_matrix.ravel())
        # Check that function gives the same result as on unitless arrays
        assert_equal(
            np.asarray(np.ravel(quadratic_matrix)), np.ravel(np.asarray(quadratic_matrix))
        )
        # Check that the function gives the same results as the original numpy
        # function
        assert_equal(
            np.ravel(np.asarray(quadratic_matrix)), np.ravel(np.asarray(quadratic_matrix))
        )

    # Do the same checks for diagonal, trace and dot
    assert_equal(np.diagonal(quadratic_matrix), quadratic_matrix.diagonal())
    assert_equal(
        np.asarray(np.diagonal(quadratic_matrix)), np.diagonal(np.asarray(quadratic_matrix))
    )
    assert_equal(
        np.diagonal(np.asarray(quadratic_matrix)),
        np.asarray(quadratic_matrix).diagonal(),
    )

    assert_equal(np.trace(quadratic_matrix), quadratic_matrix.trace())
    assert_equal(
        np.asarray(np.trace(quadratic_matrix)), np.trace(np.asarray(quadratic_matrix))
    )
    assert_equal(
        np.trace(np.asarray(quadratic_matrix)), np.trace(np.asarray(quadratic_matrix))
    )

    assert_equal(
        quantity_with_dimensions(np.dot(quadratic_matrix, quadratic_matrix), 
                                quadratic_matrix.dim*quadratic_matrix.dim),
                                Quantity(quadratic_matrix.dot(quadratic_matrix),
                                         dim=quadratic_matrix.dim*quadratic_matrix.dim)
    )

    assert_equal(
        np.asarray(np.dot(quadratic_matrix, quadratic_matrix)),
        np.dot(np.asarray(quadratic_matrix), np.asarray(quadratic_matrix)),
    )
    assert_equal(
        np.dot(np.asarray(quadratic_matrix), np.asarray(quadratic_matrix)),
        np.dot(np.asarray(quadratic_matrix), np.asarray(quadratic_matrix)),
    )

    assert_equal(
        np.asarray(quadratic_matrix.prod()), np.asarray(quadratic_matrix).prod()
    )
    assert_equal(
        np.asarray(quadratic_matrix.prod(axis=0)),
        np.asarray(quadratic_matrix).prod(axis=0),
    )

    assert have_same_dimensions(quadratic_matrix, np.ravel(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix, np.trace(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix, np.diagonal(quadratic_matrix))
    assert have_same_dimensions(
        quadratic_matrix[0] ** 2, quantity_with_dimensions(np.dot(quadratic_matrix, quadratic_matrix), get_dimensions(quadratic_matrix)*get_dimensions(quadratic_matrix))
    )
    assert have_same_dimensions(
        quadratic_matrix.prod(axis=0), quadratic_matrix[0] ** quadratic_matrix.shape[0]
    )

    # check the where function
    # pure numpy array
    cond = [True, False, False]
    ar1 = np.array([1, 2, 3])
    ar2 = np.array([4, 5, 6])
    assert_equal(np.where(cond, ar1, ar2), np.where(cond, ar1, ar2))

    # dimensionless quantity
    assert_equal(
        np.where(cond, ar1, ar2), np.asarray(np.where(cond, ar1 * mV / mV, ar2 * mV / mV))
    )

    # quantity with dimensions
    ar1 = ar1 * mV
    ar2 = ar2 * mV
    assert_equal(
        np.where(cond, np.asarray(ar1), np.asarray(ar2)),
        np.asarray(np.where(cond, ar1, ar2)),
    )

    # Check some error cases
    with pytest.raises(ValueError):
        np.where(cond, ar1)
    with pytest.raises(TypeError):
        np.where(cond, ar1, ar1, ar2)

    # Check setasflat (for numpy < 1.7)
    if hasattr(Quantity, "setasflat"):
        a = np.arange(10) * mV
        b = np.ones(10).reshape(5, 2) * volt
        c = np.ones(10).reshape(5, 2) * second
        with pytest.raises(DimensionMismatchError):
            a.setasflat(c)
        a.setasflat(b)
        assert_equal(a.flatten(), b.flatten())

    # Check cumprod
    a = np.arange(1, 10) * mV / mV
    assert_equal(a.cumprod(), np.asarray(a).cumprod())
    with pytest.raises(TypeError):
        (np.arange(1, 5) * mV).cumprod()


# Functions that should not change units

def test_numpy_functions_same_dimensions():
    values = [np.array([1, 2]), np.ones((3, 3))]
    units = [volt, second, siemens, mV, kHz]

    # numpy functions
    keep_dim_funcs = [
        np.abs,
        np.cumsum,
        np.max,
        np.mean,
        np.min,
        np.negative,
        np.ptp,
        np.round,
        np.squeeze,
        np.std,
        np.sum,
        np.transpose,
    ]

    for value, unit in itertools.product(values, units):
        q_ar = value * unit
        for func in keep_dim_funcs:
            test_ar = func(q_ar)
            if not get_dimensions(test_ar) is q_ar.dim:
                raise AssertionError(
                    f"'{func.__name__}' failed on {q_ar!r} -- dim was "
                    f"{q_ar.dim}, is now {get_dimensions(test_ar)}."
                )

                # Python builtins should work on one-dimensional arrays
                value = np.arange(5)
                builtins = [abs, max, min, sum]
                for unit in units:
                    q_ar = value * unit
                for func in builtins:
                    test_ar = func(q_ar)
                if not get_dimensions(test_ar) is q_ar.dim:
                    raise AssertionError(
                        f"'{func.__name__}' failed on {q_ar!r} -- dim "
                        f"was {q_ar.dim}, is now "
                        f"{get_dimensions(test_ar)}"
                    )



def test_numpy_functions_indices():
    """
    Check numpy functions that return indices.
    """
    values = [np.array([-4, 3, -2, 1, 0]), np.ones((3, 3)), np.array([17])]
    units = [volt, second, siemens, mV, kHz]

    # numpy functions
    keep_dim_funcs = [np.argmin, np.argmax, np.nonzero]

    for value, unit in itertools.product(values, units):
        q_ar = value * unit
        for func in keep_dim_funcs:
            test_ar = func(q_ar)
            # Compare it to the result on the same value without units
            comparison_ar = func(value)
            assert_equal(
                test_ar,
                comparison_ar,
                (
                    "function %s returned an incorrect result when used on quantities "
                    % func.__name__
                ),
            )



def test_numpy_functions_dimensionless():
    """
    Test that numpy functions that should work on dimensionless quantities only
    work dimensionless arrays and return the correct result.
    """
    unitless_values = [3, np.array([-4, 3, -1, 2]), np.ones((3, 3))]
    unit_values = [3 * mV, np.array([-4, 3, -1, 2]) * mV, np.ones((3, 3)) * mV]
    with warnings.catch_warnings():
        # ignore division by 0 warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        for value in unitless_values:
            for ufunc in UFUNCS_DIMENSIONLESS:
                result_unitless = eval(f"np.{ufunc}(value)")
                result_array = eval(f"np.{ufunc}(np.array(value))")
                assert isinstance(
                    result_unitless, (np.ndarray, np.number)
                ) and not isinstance(result_unitless, Quantity)
                assert_equal(result_unitless, result_array)
            for ufunc in UFUNCS_DIMENSIONLESS_TWOARGS:
                result_unitless = eval(f"np.{ufunc}(value, value)")
                result_array = eval(f"np.{ufunc}(np.array(value), np.array(value))")
                assert isinstance(
                    result_unitless, (np.ndarray, np.number)
                ) and not isinstance(result_unitless, Quantity)
                assert_equal(result_unitless, result_array)

        for value, unitless_value in zip(unit_values, unitless_values):
            for ufunc in UFUNCS_DIMENSIONLESS:
                with pytest.raises(DimensionMismatchError):
                    eval(f"np.{ufunc}(value)", globals(), {"value": value})
            for ufunc in UFUNCS_DIMENSIONLESS_TWOARGS:
                with pytest.raises(DimensionMismatchError):
                    eval(
                        f"np.{ufunc}(value1, value2)",
                        globals(),
                        {"value1": value, "value2": unitless_value},
                    )
                with pytest.raises(DimensionMismatchError):
                    eval(
                        f"np.{ufunc}(value2, value1)",
                        globals(),
                        {"value1": value, "value2": unitless_value},
                    )
                with pytest.raises(DimensionMismatchError):
                    eval(f"np.{ufunc}(value, value)", globals(), {"value": value})



def test_numpy_functions_change_dimensions():
    """
    Test some numpy functions that change the dimensions of the quantity.
    """
    unit_values = [np.array([1, 2]) * mV, np.ones((3, 3)) * 2 * mV]
    for value in unit_values:
        assert_quantity(np.var(value), np.var(np.array(value)), volt**2)
        assert_quantity(np.square(value), np.square(np.array(value)), volt**2)
        assert_quantity(np.sqrt(value), np.sqrt(np.array(value)), volt**0.5)
        assert_quantity(
            np.reciprocal(value), np.reciprocal(np.array(value)), 1.0 / volt
        )



def test_numpy_functions_typeerror():
    """
    Assures that certain numpy functions raise a TypeError when called on
    quantities.
    """
    unitless_values = [
        3 * mV / mV,
        np.array([1, 2]) * mV / mV,
        np.ones((3, 3)) * mV / mV,
    ]
    unit_values = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    for value in unitless_values + unit_values:
        for ufunc in UFUNCS_INTEGERS:
            if ufunc == "invert":
                # only takes one argument
                with pytest.raises(TypeError):
                    eval(f"np.{ufunc}(value)", globals(), {"value": value})
            else:
                with pytest.raises(TypeError):
                    eval(f"np.{ufunc}(value, value)", globals(), {"value": value})



def test_numpy_functions_logical():
    """
    Assure that logical numpy functions work on all quantities and return
    unitless boolean arrays.
    """
    unit_values1 = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    unit_values2 = [3 * second, np.array([1, 2]) * second, np.ones((3, 3)) * second]
    for ufunc in UFUNCS_LOGICAL:
        for value1, value2 in zip(unit_values1, unit_values2):
            try:
                # one argument
                result_units = eval(f"np.{ufunc}(value1)")
                result_array = eval(f"np.{ufunc}(np.array(value1))")
            except (ValueError, TypeError):
                # two arguments
                result_units = eval(f"np.{ufunc}(value1, value2)")
                result_array = eval(f"np.{ufunc}(np.array(value1), np.array(value2))")
            assert not isinstance(result_units, Quantity)
            assert_equal(result_units, result_array)



def test_arange_linspace():
    # Make sure units are checked
    with pytest.raises(DimensionMismatchError):
        np.arange(1 * mV, 5)
    with pytest.raises(DimensionMismatchError):
        np.arange(1 * mV, 5 * mV)
    with pytest.raises(DimensionMismatchError):
        np.arange(1, 5 * mV)
    with pytest.raises(DimensionMismatchError):
        np.arange(1 * mV, 5 * ms)
    with pytest.raises(DimensionMismatchError):
        np.arange(1 * mV, 5 * mV, step=1 * ms)
    with pytest.raises(DimensionMismatchError):
        np.arange(1 * ms, 5 * mV)

def test_list():
    """
    Test converting to and from a list.
    """
    values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(4, 3) * mV]
    for value in values:
        l = value.tolist()
        from_list = Quantity(l)
        assert have_same_dimensions(from_list, value)
        assert_equal(from_list, value)



def test_check_units():
    """
    Test the check_units decorator
    """

    @check_units(v=volt)
    def a_function(v, x):
        """
        v has to have units of volt, x can have any (or no) unit.
        """
        pass

    # Try correct units
    a_function(3 * mV, 5 * second)
    a_function(5 * volt, "something")
    a_function([1, 2, 3] * volt, None)
    # lists that can be converted should also work
    a_function([1 * volt, 2 * volt, 3 * volt], None)
    # Strings and None are also allowed to pass
    a_function("a string", None)
    a_function(None, None)

    # Try incorrect units
    with pytest.raises(DimensionMismatchError):
        a_function(5 * second, None)
    with pytest.raises(DimensionMismatchError):
        a_function(5, None)
    with pytest.raises(TypeError):
        a_function(object(), None)
    with pytest.raises(TypeError):
        a_function([1, 2 * volt, 3], None)

    @check_units(result=second)
    def b_function(return_second):
        """
        Return a value in seconds if return_second is True, otherwise return
        a value in volt.
        """
        if return_second:
            return 5 * second
        else:
            return 3 * volt

    # Should work (returns second)
    b_function(True)
    # Should fail (returns volt)
    with pytest.raises(DimensionMismatchError):
        b_function(False)

    @check_units(a=bool, b=1, result=bool)
    def c_function(a, b):
        if a:
            return b > 0
        else:
            return b

    assert c_function(True, 1)
    assert not c_function(True, -1)
    with pytest.raises(TypeError):
        c_function(1, 1)
    with pytest.raises(TypeError):
        c_function(1 * mV, 1)
    with pytest.raises(TypeError):
        c_function(False, 1)



def test_get_unit():
    """
    Test get_unit
    """
    values = [
        (volt.dim, volt),
        (mV.dim, volt),
        ((amp / metre**2).dim, amp / metre**2),
    ]
    for dim, expected_unit in values:
        unit = get_unit(dim)
        assert isinstance(unit, Unit)
        assert unit == expected_unit
        assert float(unit) == 1.0



def test_get_best_unit():
    # get_best_unit should not check all values for long arrays, since it is
    # a function used for display purposes only. Instead, only the first and
    # last few values should matter (see github issue #966)
    long_ar = np.ones(10000) * siemens
    long_ar[:10] = 1 * nS
    long_ar[-10:] = 2 * nS
    values = [
        (np.arange(10) * mV, mV),
        ([0.001, 0.002, 0.003] * second, ms),
        (long_ar, nS),
    ]
    for ar, expected_unit in values:
        assert ar.get_best_unit() is expected_unit
        assert str(expected_unit) in ar.in_best_unit()



def test_switching_off_unit_checks():
    """
    Check switching off unit checks (used for external functions).
    """
    import QuantSI.fundamentalunits as fundamentalunits

    x = 3 * second
    y = 5 * volt
    with pytest.raises(DimensionMismatchError):
        x + y
    fundamentalunits.unit_checking = False
    # Now it should work
    assert np.asarray(x + y) == np.array(8)
    assert have_same_dimensions(x, y)
    assert x.has_same_dimensions(y)
    fundamentalunits.unit_checking = True



def test_fail_for_dimension_mismatch():
    """
    Test the fail_for_dimension_mismatch function.
    """
    # examples that should not raise an error
    dim1, dim2 = fail_for_dimension_mismatch(3)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt, 7)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt, 5 * volt)
    assert dim1 is volt.dim
    assert dim2 is volt.dim

    # examples that should raise an error
    with pytest.raises(DimensionMismatchError):
        fail_for_dimension_mismatch(6 * volt)
    with pytest.raises(DimensionMismatchError):
        fail_for_dimension_mismatch(6 * volt, 5 * second)



def test_deepcopy():
    d = {"x": 1 * second}
    from copy import deepcopy

    d_copy = deepcopy(d)
    assert d_copy["x"] == 1 * second
    d_copy["x"] += 1 * second
    assert d_copy["x"] == 2 * second
    assert d["x"] == 1 * second



def test_inplace_on_scalars():
    # We want "copy semantics" for in-place operations on scalar quantities
    # in the same way as for Python scalars
    for scalar in [3 * mV, 3 * mV / mV]:
        scalar_reference = scalar
        scalar_copy = Quantity(scalar, copy=True)
        scalar += scalar_copy
        assert_equal(scalar_copy, scalar_reference)
        scalar *= 1.5
        assert_equal(scalar_copy, scalar_reference)
        scalar /= 2
        assert_equal(scalar_copy, scalar_reference)

        # also check that it worked correctly for the scalar itself
        # assert_allclose(scalar, (scalar_copy + scalar_copy) * 1.5 / 2)

    # For arrays, it should use reference semantics
    for vector in [[3] * mV, [3] * mV / mV]:
        vector_reference = vector
        vector_copy = Quantity(vector, copy=True)
        vector += vector_copy
        assert_equal(vector, vector_reference)
        vector *= 1.5
        assert_equal(vector, vector_reference)
        vector /= 2
        assert_equal(vector, vector_reference)

        # also check that it worked correctly for the vector itself
        # assert_allclose(vector, (vector_copy + vector_copy) * 1.5 / 2)


def test_units_vs_quantities():
    # Unit objects should stay Unit objects under certain operations
    # (important e.g. in the unit definition of Equations, where only units but
    # not quantities are allowed)
    assert isinstance(meter**2, Unit)
    assert isinstance(meter**-1, Unit)
    assert isinstance(meter**0.5, Unit)
    assert isinstance(meter / second, Unit)
    assert isinstance(amp / meter**2, Unit)
    assert isinstance(1 / meter, Unit)
    assert isinstance(1.0 / meter, Unit)

    # Using the unconventional type(x) == y since we want to test that
    # e.g. meter**2 stays a Unit and does not become a Quantity however Unit
    # inherits from Quantity and therefore both would pass the isinstance test
    assert type(2 / meter) == Quantity
    assert type(2 * meter) == Quantity
    assert type(meter + meter) == Quantity
    assert type(meter - meter) == Quantity



def test_all_units_list():
    from QuantSI.allunits import all_units

    assert meter in all_units
    assert volt in all_units
    assert cm in all_units
    assert Hz in all_units
    assert all(isinstance(u, Unit) for u in all_units)



def test_constants():
    import QuantSI.constants as constants

    # Check that the expected names exist and have the correct dimensions
    assert constants.avogadro_constant.dim == (1 / mole).dim
    assert constants.boltzmann_constant.dim == (joule / kelvin).dim
    assert constants.electric_constant.dim == (farad / meter).dim
    assert constants.electron_mass.dim == kilogram.dim
    assert constants.elementary_charge.dim == coulomb.dim
    assert constants.faraday_constant.dim == (coulomb / mole).dim
    assert constants.gas_constant.dim == (joule / mole / kelvin).dim
    assert constants.magnetic_constant.dim == (newton / amp2).dim
    assert constants.molar_mass_constant.dim == (kilogram / mole).dim
    assert constants.zero_celsius.dim == kelvin.dim

    #Check the consistency between a few constants
    np.allclose(
        np.asarray(constants.gas_constant),
        np.asarray(constants.avogadro_constant * constants.boltzmann_constant),
    )
    np.allclose(
        np.asarray(constants.faraday_constant),
        np.asarray(constants.avogadro_constant * constants.elementary_charge),
    )


if __name__ == "__main__":
    test_construction()
    test_get_dimensions()
    test_display()
    test_power()
    test_pickling()
    test_str_repr()
    test_slicing()
    test_setting()
    test_multiplication_division()
    test_addition_subtraction()
    test_unary_operations()
    test_binary_operations()
    test_inplace_operations()
    test_unit_discarding_functions()
    test_special_case_numpy_functions()
    test_numpy_functions_same_dimensions()
    test_numpy_functions_indices()
    test_numpy_functions_dimensionless()
    test_numpy_functions_change_dimensions()
    test_numpy_functions_typeerror()
    test_numpy_functions_logical()
    test_arange_linspace()
    test_list()
    test_check_units()
    test_get_unit()
    test_get_best_unit()
    test_switching_off_unit_checks()
    test_fail_for_dimension_mismatch()
    test_deepcopy()
    test_inplace_on_scalars()
    test_units_vs_quantities()
    test_all_units_list()
    test_constants()
