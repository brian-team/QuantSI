"""The examples whose display output is stored in data/display_golden.txt.

render() turns each value into text in every way QuantSI can (str, repr, format,
in_unit, LaTeX, ...). test_display_golden.py runs it in a fresh Python process,
because registering units changes how values are displayed, and other tests
register units. If you change the output on purpose, regenerate the file with
``python tools/make_display_golden.py`` and check the diff.
"""

import numpy as np

import QuantSI  # noqa: F401  (registers the standard units)
from QuantSI.allunits import (
    farad,
    joule,
    kvolt,
    metre,
    msecond,
    mvolt,
    namp,
    newton,
    second,
    siemens,
    volt,
)
from QuantSI.fundamentalunits import Quantity, in_best_unit, in_unit


def values():
    """(case id, value) pairs, in a fixed order."""
    return [
        ("scalar_mV", 3 * mvolt),
        ("scalar_many_digits", 0.025123456 * volt),
        ("scalar_negative", -70 * mvolt),
        ("scalar_zero", 0 * mvolt),
        ("scalar_tiny", 1e-5 * volt),
        ("scalar_large", 1234.5 * msecond),
        ("scalar_inf", np.inf * mvolt),
        ("scalar_nan", np.nan * mvolt),
        ("scalar_int_times_unit", 7 * namp),
        ("array_1d", np.array([1.0, 2.0, 3.0]) * mvolt),
        ("array_mixed_magnitudes", np.array([1e-3, 2.5, 1000.0]) * volt),
        ("array_2d", np.arange(6.0).reshape(2, 3) * namp),
        ("array_empty", np.array([]) * volt),
        ("array_long", np.arange(2000.0) * msecond),
        ("array_float32", np.array([1.5, 2.5], dtype=np.float32) * metre),
        ("compound_registered", 3 * newton * metre),
        ("compound_unregistered", 2.0 * farad / metre**2),
        ("compound_density", 5 * siemens / metre**2),
        ("explicit_dimensionless", Quantity([1.0, 2.0])),
        ("unit_mvolt", mvolt),
        ("unit_power", volt**2),
        ("unit_quotient", metre / second),
        ("unit_compound_power", (metre / second) ** 2),
        ("energy", 3 * joule),
    ]


def renderings(value):
    """(name, function) pairs: the ways to turn one value into text."""
    return [
        ("str", lambda: str(value)),
        ("repr", lambda: repr(value)),
        ("format_empty", lambda: format(value, "")),
        ("format_2f", lambda: format(value, ".2f")),
        ("in_best_unit_p2", lambda: value.in_best_unit(precision=2)),
        ("in_best_unit_python", lambda: value.in_best_unit(python_code=True)),
        ("in_unit_volt", lambda: value.in_unit(volt)),
        ("in_unit_kvolt_p3", lambda: value.in_unit(kvolt, 3)),
        ("in_unit_volt_python", lambda: value.in_unit(volt, python_code=True)),
        ("repr_latex", lambda: value._repr_latex_()),
    ]


def render():
    lines = []
    for case, value in values():
        for name, func in renderings(value):
            try:
                out = repr(func())
            except Exception as exc:  # an error is a result too
                out = f"raises {type(exc).__name__}"
            lines.append(f"{case} | {name} | {out}")
        with np.printoptions(precision=3):
            lines.append(f"{case} | str_precision3 | {str(value)!r}")
    lines.append(f"function | in_unit | {in_unit(3 * volt, mvolt)!r}")
    lines.append(f"function | in_unit_p2 | {in_unit(123123 * msecond, second, 2)!r}")
    lines.append(f"function | in_best_unit | {in_best_unit(0.00123456 * volt)!r}")
    lines.append(f"function | in_best_unit_number | {in_best_unit(0.123456, 2)!r}")
    return "\n".join(lines) + "\n"
