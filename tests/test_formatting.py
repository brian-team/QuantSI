"""The formatting core."""

import numpy as np
import pytest

from QuantSI._core.formatting import format_quantity
from QuantSI.allunits import msecond, mvolt, second, volt


def old_scalar_text(value, precision=None):
    """How scalars were written before: as a 1-element array, brackets removed."""
    text = np.array_str(np.array([value]), precision=precision)
    return text.replace("[", "").replace("]", "").strip()


SCALARS = [0.025123456, 3.0, 0.003, 1e-05, 1.5e-4, 123456789.0, 1e8, -70.0, 0.1 + 0.2, 2.5]
SCALARS += [1 / 3, 0.0, np.inf, -np.inf, np.nan, 7, np.float32(0.1), 1e-300]


@pytest.mark.parametrize("legacy", [False, "1.13"])
@pytest.mark.parametrize("precision", [None, 3])
@pytest.mark.parametrize("value", SCALARS, ids=repr)
def test_scalars_are_written_as_before(value, precision, legacy):
    # Also in NumPy's legacy print mode, where a 0-d array would be written with
    # the scalar's repr ("20.0" instead of "20.").
    stored = np.asarray((value * volt) / volt)  # what the core sees (7 * volt is a float)
    with np.printoptions(legacy=legacy):
        expected = old_scalar_text(stored, precision) + " V"
        assert format_quantity(value * volt, volt, precision=precision) == expected


def test_scalars_follow_numpy_print_options():
    with np.printoptions(precision=3):
        assert format_quantity(25.123456 * mvolt, mvolt) == "25.123 mV"


def test_python_code_evaluates_back():
    from QuantSI.allunits import mvolt as mvolt_  # noqa: F401  (used by eval)

    q = np.array([1.5, 2.5]) * mvolt
    text = format_quantity(q, mvolt, python_code=True)
    assert text == "array([1.5, 2.5]) * mvolt"
    np.testing.assert_array_equal(
        np.asarray(eval(text, {"array": np.array, "mvolt": mvolt})), np.asarray(q)
    )


@pytest.mark.parametrize(
    ("value", "spec", "expected"),
    [
        (3 * mvolt, ".2f", "0.00"),
        (3 * mvolt, "", "3. mV"),
        (0.025123456 * volt, ".3g", "0.0251"),
        (3 * mvolt, ">8.1f", "     0.0"),
    ],
)
def test_format_specifications(value, spec, expected):
    assert format(value, spec) == expected
    assert f"{value:{spec}}" == expected


def test_invalid_format_specification():
    with pytest.raises(ValueError):
        format(3 * mvolt, ".2q")


@pytest.mark.parametrize("value", [0.001 * second, 1 * msecond])
def test_scalar_formatting_does_not_depend_on_input_unit(value):
    assert f"{value:.2f}" == "0.00"
    assert f"{value:g}" == "0.001"
    assert f"{value}" == "1. ms"


@pytest.mark.parametrize("spec", [".1f", ".1e"])
def test_array_numeric_formatting_is_unsupported(spec):
    q = np.array([1.0, 2.5]) * mvolt
    with pytest.raises(TypeError, match="unsupported format string"):
        format(q, spec)
