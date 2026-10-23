"""The ufunc rule table, and the old name lists we keep for compatibility."""

import numpy as np
import pytest

from QuantSI._core import ufuncs as _ufuncs
from QuantSI._core.ufuncs import RULES, Rule
from QuantSI.fundamentalunits import Quantity

#: Which rules each old list stands for.
LEGACY_LISTS = {
    "UFUNCS_PRESERVE_DIMENSIONS": {Rule.PRESERVE},
    "UFUNCS_CHANGE_DIMENSIONS": {
        Rule.MULTIPLY,
        Rule.DIVIDE,
        Rule.SQRT,
        Rule.SQUARE,
        Rule.RECIPROCAL,
    },
    "UFUNCS_MATCHING_DIMENSIONS": {Rule.MATCH},
    "UFUNCS_COMPARISONS": {Rule.COMPARE},
    "UFUNCS_LOGICAL": {Rule.UNITLESS_RESULT},
    "UFUNCS_DIMENSIONLESS": {Rule.DIMENSIONLESS},
    "UFUNCS_DIMENSIONLESS_TWOARGS": {Rule.DIMENSIONLESS_BOTH},
    "UFUNCS_INTEGERS": {Rule.INTEGER_ONLY},
}


def test_rules_are_keyed_by_ufunc_objects():
    assert all(isinstance(ufunc, np.ufunc) for ufunc in RULES)
    assert all(isinstance(rule, Rule) for rule in RULES.values())


@pytest.mark.parametrize("list_name", sorted(LEGACY_LISTS))
def test_historical_lists_agree_with_the_rules(list_name):
    for name in getattr(_ufuncs, list_name):
        ufunc = getattr(np, name, None)
        if isinstance(ufunc, np.ufunc):  # some old names are not ufuncs
            assert RULES[ufunc] in LEGACY_LISTS[list_name], name


def test_aliases_share_one_rule():
    assert np.true_divide is np.divide and RULES[np.true_divide] is Rule.DIVIDE
    assert np.mod is np.remainder and RULES[np.mod] is Rule.MATCH


def test_every_numpy_ufunc_has_a_rule():
    # A new NumPy release that adds a ufunc fails this test until it gets a rule.
    # (Ufuncs outside the numpy namespace, like the numpy.strings implementations
    # and private helpers, never receive numbers; NumPy reports a missing loop.)
    public = {getattr(np, name) for name in dir(np) if isinstance(getattr(np, name), np.ufunc)}
    unclassified = sorted(u.__name__ for u in public if u not in RULES)
    assert not unclassified


def test_every_rule_has_a_handler():
    assert set(_ufuncs._HANDLER_FOR_RULE) == set(Rule)
    assert set(_ufuncs.UNSUPPORTED_REASONS) == {
        u for u, r in RULES.items() if r is Rule.UNSUPPORTED
    }


@pytest.mark.parametrize(
    ("expression", "expected_dim"),
    [
        (lambda q: np.fmax(q, q), "m"),
        (lambda q: np.fabs(-q), "m"),
        (lambda q: np.cbrt(q**3), "m"),
        (lambda q: np.copysign(q, -1), "m"),
        (lambda q: np.vecdot(q, q), "m2"),
        (lambda q: np.float_power(q, 2), "m2"),
    ],
)
def test_newly_classified_ufuncs(expression, expected_dim):
    from QuantSI.allunits import metre

    q = np.array([1.0, 8.0]) * metre
    expected = {"m": metre.dim, "m2": (metre**2).dim}[expected_dim]
    assert expression(q).dim is expected


@pytest.mark.parametrize("ufunc", [np.divmod, np.modf, np.heaviside, np.ldexp])
def test_unsupported_ufuncs_explain_themselves(ufunc):
    from QuantSI.allunits import metre

    q = np.array([1.0, 2.0]) * metre
    args = (q,) if ufunc.nin == 1 else (q, 2)
    with pytest.raises(TypeError, match="not supported for quantities"):
        ufunc(*args)


def test_angle_conversions_need_dimensionless_input():
    from QuantSI import DimensionMismatchError
    from QuantSI.allunits import metre

    assert np.degrees(np.pi) == 180
    with pytest.raises(DimensionMismatchError):
        np.degrees(3 * metre)


@pytest.mark.parametrize("ufunc", [np.sin, np.hypot])
def test_dimensionless_ufuncs_on_explicit_quantities(ufunc):
    values = np.array([0.1, 0.2])
    q = Quantity(values)
    arguments = (q,) if ufunc.nin == 1 else (q, q)
    plain_arguments = (values,) if ufunc.nin == 1 else (values, values)
    result = ufunc(*arguments)
    assert type(result) is np.ndarray
    np.testing.assert_allclose(result, ufunc(*plain_arguments))


def test_unrecognised_ufunc_returns_not_implemented():
    from QuantSI.allunits import metre

    custom_ufunc = np.frompyfunc(lambda x: x, 1, 1)
    with pytest.raises(TypeError, match="NotImplemented"):
        custom_ufunc(3 * metre)
