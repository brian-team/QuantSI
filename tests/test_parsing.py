"""parse_quantity: a small, safe language for quantities in text."""

import numpy as np
import pytest

from QuantSI._core.errors import QuantityParseError
from QuantSI._core.parsing import MAX_LENGTH, parse_dimensions, parse_quantity
from QuantSI.allunits import (
    amp,
    cmetre,
    metre,
    msecond,
    mvolt,
    namp,
    second,
    siemens,
    usiemens,
    volt,
)
from QuantSI.fundamentalunits import DIMENSIONLESS
from QuantSI.stdunits import cm, mV


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3.5 * mV", 3.5 * mV),
        ("3.5*mvolt", 3.5 * mvolt),
        ("  -70 * mV ", -70 * mV),
        ("+3 * mV", 3 * mV),
        ("1e-9 * amp", 1e-9 * amp),
        ("0.5 * siemens / cm ** 2", 0.5 * siemens / cm**2),
        ("metre ** -2", metre**-2),
        ("(volt / second) * 2", (volt / second) * 2),
        ("2 * (3 * msecond)", 6 * msecond),
        ("metre ** (1 / 2)", metre**0.5),
    ],
)
def test_parse(text, expected):
    result = parse_quantity(text)
    assert result == expected
    assert result.dim is expected.dim  # the interned dimension


def test_dimensionless_results_are_plain_numbers():
    assert parse_quantity("3.5") == 3.5
    result = parse_quantity("3 * volt / volt")
    assert result == 3.0 and isinstance(result, float)
    # (volt / volt alone is a Unit: units combine into units, as in Python code)
    assert parse_quantity("volt / volt") == volt / volt
    assert parse_quantity("2 ** 10") == 1024.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("3.5 mV", "Not a valid expression"),  # no implicit multiplication
        ("", "Not a valid expression"),
        ("1; 2", "Not a valid expression"),
        ("sin(volt)", "Call is not allowed"),
        ("volt.dim", "Attribute is not allowed"),
        ("mvolt[0]", "Subscript is not allowed"),
        ("[1 * mV, 2 * mV]", "must be plain numbers"),
        ("array([1, 2], dtype=int)", "Call is not allowed"),
        ("zeros([1, 2])", "Call is not allowed"),
        ("'3 mV'", "Constant is not allowed"),
        ("True * volt", "Constant is not allowed"),
        ("lambda: 1", "Lambda is not allowed"),
        ("mv * 3", "'mv' is not a known unit"),
        ("metre ** metre", "exponent must be a plain number"),
        ("metre ** 65", "larger than 64"),
        ("((10 ** 64) ** 64) ** 64", "too large"),
    ],
)
def test_refused(text, message):
    with pytest.raises(QuantityParseError, match=message):
        parse_quantity(text)


def test_input_limits():
    with pytest.raises(QuantityParseError, match="longer than"):
        parse_quantity("1" * (MAX_LENGTH + 1))
    with pytest.raises(QuantityParseError, match="more than"):
        parse_quantity("-" * 199 + "1")  # short, but 400 nodes


def test_parse_errors_are_value_errors():
    with pytest.raises(ValueError):
        parse_quantity("3 mV")


def test_custom_namespace():
    assert parse_quantity("2 * stride", namespace={"stride": metre}) == 2 * metre
    with pytest.raises(QuantityParseError, match="not a known unit"):
        parse_quantity("mV", namespace={})


@pytest.mark.parametrize(
    ("text", "suggestion"),
    [("mv * 3", "Did you mean mV"), ("volts", "volt"), ("Siemens", "siemens")],
)
def test_unknown_names_get_suggestions(text, suggestion):
    with pytest.raises(QuantityParseError, match=suggestion):
        parse_quantity(text)


def test_no_suggestion_for_nonsense():
    with pytest.raises(QuantityParseError) as error:
        parse_quantity("xyzzy")
    assert "Did you mean" not in str(error.value)


def test_parse_dimensions():
    assert parse_dimensions("siemens / metre ** 2") is (siemens / metre**2).dim
    assert parse_dimensions("3 * mV") is volt.dim
    assert parse_dimensions("2.5") is DIMENSIONLESS
    with pytest.raises(QuantityParseError):
        parse_dimensions("volt.dim")


def test_lists_of_numbers():
    q = parse_quantity("[1.5, -2.5] * mV")
    np.testing.assert_array_equal(np.asarray(q), [0.0015, -0.0025])
    assert parse_quantity("array([[1, 2], [3, 4]]) * mvolt").shape == (2, 2)


# ---- Reading back what repr writes ------------------------------------------------
# repr writes numbers with NumPy's print precision: by default 8 digits after the
# decimal point, in the unit chosen for display. Values that print exactly read
# back exactly; with np.printoptions(precision=17), every value does.
ROUND_TRIP_VALUES = [3.0, -70.0, 0.025, 1e-9, 12345.0, 0.0]
ROUND_TRIP_VALUES += [
    np.array([1.5, -2.5]),
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    np.array([1e-3, 2.5, 1e3]),
]
# Built inside the test: combining units registers them for display, and doing
# that at import time would change the display for every other test.
ROUND_TRIP_UNITS = {
    "mvolt": lambda: mvolt,
    "namp": lambda: namp,
    "metre/second": lambda: metre / second,
    "usiemens/cmetre**2": lambda: usiemens / cmetre**2,
    "volt**2": lambda: volt**2,
    "siemens": lambda: siemens,
}


@pytest.mark.parametrize("unit", ROUND_TRIP_UNITS)
@pytest.mark.parametrize("value", ROUND_TRIP_VALUES, ids=lambda v: repr(v)[:20])
def test_repr_reads_back(value, unit):
    q = value * ROUND_TRIP_UNITS[unit]()
    back = parse_quantity(repr(q))
    assert back.dim is q.dim
    np.testing.assert_allclose(np.asarray(back), np.asarray(q), rtol=1e-14)


def test_repr_reads_back_exactly_with_full_precision():
    rng = np.random.default_rng(7)
    with np.printoptions(precision=17):
        for value in [
            *rng.normal(size=20) * 10.0 ** rng.integers(-12, 12, size=20),
            rng.normal(size=5),
        ]:
            q = value * mvolt
            np.testing.assert_array_equal(np.asarray(parse_quantity(repr(q))), np.asarray(q))


def test_abbreviated_repr_is_refused():
    with pytest.raises(QuantityParseError, match="abbreviated"):
        parse_quantity(repr(np.arange(2000.0) * mvolt))


def test_nan_does_not_read_back():
    # Documented: NaN and infinity are not part of the language.
    with pytest.raises(QuantityParseError, match="'nan' is not a known unit"):
        parse_quantity(repr(np.array([np.nan, 1.0]) * mvolt))


@pytest.mark.parametrize("value", [None, 3, [1, 2]])
def test_parse_quantity_requires_text(value):
    with pytest.raises(TypeError, match=f"Expected a string, got {type(value).__name__}"):
        parse_quantity(value)
