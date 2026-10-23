"""Write tests/data/public_api.json: every name QuantSI's public modules provide.

test_public_api.py checks that these names stay importable. Only run this when
you change the public API on purpose, and look at the diff of the JSON file
before committing::

    python tools/snapshot_public_api.py
"""

import importlib
import inspect
import json
import pathlib

#: The public modules. Users import names from these.
MODULES = [
    "QuantSI",
    "QuantSI.fundamentalunits",
    "QuantSI.allunits",
    "QuantSI.stdunits",
    "QuantSI.constants",
    "QuantSI.unitsafefunctions",
]

#: Private names we still have to keep, because Brian2 imports them from
#: brian2.units.fundamentalunits.
PRIVATE_BUT_PRESENT = {
    "QuantSI.fundamentalunits": [
        "_di",
        "_dimensions",
        "_flatten",
        "_iclass_label",
        "_ilabel",
        "_short_str",
        "_siprefixes",
    ],
}

OUTPUT = pathlib.Path(__file__).parent.parent / "tests/data/public_api.json"


def is_own_public_name(module, name):
    """True for public names that QuantSI defines itself (not things like np)."""
    if name.startswith("_"):
        return False
    obj = getattr(module, name)
    if inspect.ismodule(obj):
        return False
    # Skip things QuantSI only imports, like numpy or sympy functions. Units and
    # dimensions report a QuantSI module through their class, and plain lists or
    # dicts have no __module__ at all, so both count as ours.
    owner = getattr(obj, "__module__", None)
    return owner is None or owner.startswith("QuantSI")


def snapshot():
    surface = {}
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        names = sorted(n for n in dir(module) if is_own_public_name(module, n))
        names += PRIVATE_BUT_PRESENT.get(module_name, [])
        surface[module_name] = sorted(names)
    return surface


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT}")
