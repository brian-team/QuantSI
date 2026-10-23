"""Every name brian2.units offers today, QuantSI offers too.

``data/brian2_units_surface.json`` lists the names of Brian2's unit modules, at
the Brian2 commit recorded in the file (``tools/snapshot_brian2_surface.py``
writes it). Brian2's ``brian2.units`` will become a thin wrapper around QuantSI,
so each of these names has to exist in the QuantSI module it comes from, except
for the few left out on purpose (``REMOVED``), which Brian2 keeps providing
itself.
"""

import importlib
import json
import pathlib

import pytest

SURFACE = json.loads(
    (pathlib.Path(__file__).parent / "data" / "brian2_units_surface.json").read_text()
)

#: brian2 module -> the QuantSI modules that provide its names
SOURCES = {
    "brian2.units": ["QuantSI", "QuantSI.unitsafefunctions"],
    "brian2.units.fundamentalunits": ["QuantSI.fundamentalunits"],
    "brian2.units.allunits": ["QuantSI.allunits"],
    "brian2.units.stdunits": ["QuantSI.stdunits"],
    "brian2.units.constants": ["QuantSI.constants"],
    "brian2.units.unitsafefunctions": ["QuantSI.unitsafefunctions"],
}

#: Names that QuantSI does not provide, on purpose, and why.
REMOVED = {
    "unit_checking": "the undocumented switch to turn unit checks off was removed from QuantSI",
    "celsius": "removed from QuantSI; Brian2 keeps the placeholder that explains why",
    "_Celsius": "the class of that placeholder",
    "fundamentalunits_all": "Brian2's own bookkeeping in brian2/units/__init__.py",
    "stdunits_all": "Brian2's own bookkeeping in brian2/units/__init__.py",
    "unitsafefunctions_all": "Brian2's own bookkeeping in brian2/units/__init__.py",
}


@pytest.mark.parametrize("brian2_module", sorted(SOURCES))
def test_brian2_names_are_available(brian2_module):
    sources = [importlib.import_module(name) for name in SOURCES[brian2_module]]
    missing = [
        name
        for name in SURFACE[brian2_module]
        if name not in REMOVED and not any(hasattr(source, name) for source in sources)
    ]
    assert not missing, f"{brian2_module} names missing from QuantSI: {missing}"


def test_removed_names_are_still_removed():
    # If one comes back, it should be taken off REMOVED (and the reason dropped).
    import QuantSI.allunits
    import QuantSI.fundamentalunits

    assert not hasattr(QuantSI.fundamentalunits, "unit_checking")
    assert not hasattr(QuantSI.allunits, "celsius")
