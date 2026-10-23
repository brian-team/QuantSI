"""The naming policy: which unit names exist (docs_sphinx/user/naming.rst)."""

import importlib

import pytest

from QuantSI import allunits, stdunits
from QuantSI._core.dimension import _siprefixes

PUBLIC_MODULES = ["QuantSI", "QuantSI.allunits", "QuantSI.stdunits", "QuantSI.fundamentalunits"]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_no_single_letter_names(module_name):
    # A user's variable called V, A or m must never be shadowed by a unit.
    module = importlib.import_module(module_name)
    assert [name for name in getattr(module, "__all__", []) if len(name) == 1] == []


def test_short_names_follow_the_rule():
    # A short name is the unit's own symbol (its display name, "cm^2" -> "cm2"),
    # and it is the very same object as the long name in allunits.
    for name in stdunits.__all__:
        unit = getattr(stdunits, name)
        assert getattr(allunits, unit.name) is unit
        assert unit.dispname.replace("^", "") == name


@pytest.mark.parametrize(
    "short, long",
    [
        ("mV", "mvolt"),
        ("uV", "uvolt"),
        ("nA", "namp"),
        ("ns", "nsecond"),
        ("GHz", "Ghertz"),
        ("cm2", "cmetre2"),
        ("pM", "pmolar"),
        ("mF", "mfarad"),
        ("pS", "psiemens"),
    ],
)
def test_short_names(short, long):
    assert getattr(stdunits, short) is getattr(allunits, long)


def test_every_prefix_of_the_rule_exists():
    # "uV exists because mV does": for each symbol, all prefixes of its family.
    for symbol, prefixes in [
        ("V", "munp"),
        ("A", "munp"),
        ("F", "munp"),
        ("S", "munp"),
        ("s", "munp"),
        ("M", "munp"),
    ]:
        base = getattr(stdunits, "m" + symbol)  # every family has the milli- unit
        for prefix in prefixes:
            unit = getattr(stdunits, prefix + symbol)
            assert unit.dim is base.dim
            assert unit.scale - base.scale == _siprefixes[prefix] - _siprefixes["m"]
