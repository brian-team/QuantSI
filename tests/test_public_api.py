"""Names in QuantSI's public modules must never disappear.

data/public_api.json lists every name each public module had when it was
written (by tools/snapshot_public_api.py). Brian2 re-exports these modules, so
removing or renaming a name breaks Brian2 users too. If one of these tests
fails, a name went missing: bring it back instead of regenerating the snapshot.
"""

import importlib
import json
import pathlib
import pkgutil

import pytest

import QuantSI

SNAPSHOT = json.loads((pathlib.Path(__file__).parent / "data" / "public_api.json").read_text())

#: The modules users may import. Any other module must be private (a name
#: starting with an underscore, like QuantSI._core).
PUBLIC_MODULES = {
    "QuantSI",
    "QuantSI.allunits",
    "QuantSI.constants",
    "QuantSI.fundamentalunits",
    "QuantSI.stdunits",
    "QuantSI.unitsafefunctions",
}


@pytest.mark.parametrize("module_name", sorted(SNAPSHOT))
def test_snapshot_names_still_importable(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in SNAPSHOT[module_name] if not hasattr(module, name)]
    assert not missing, f"{module_name} lost names: {missing}"


@pytest.mark.parametrize("module_name", sorted(PUBLIC_MODULES))
def test_all_names_resolve(module_name):
    module = importlib.import_module(module_name)
    unresolved = [name for name in getattr(module, "__all__", []) if not hasattr(module, name)]
    assert not unresolved, f"{module_name}.__all__ lists missing names: {unresolved}"


def test_no_unexpected_public_modules():
    found = {"QuantSI"}
    for info in pkgutil.walk_packages(QuantSI.__path__, prefix="QuantSI."):
        name = info.name
        if any(part.startswith("_") for part in name.split(".")):
            continue
        found.add(name)
    assert found == PUBLIC_MODULES


def test_reexports_are_the_same_objects():
    # A name exported from several modules has to be the same object everywhere.
    # A copy would break checks like `dim1 is dim2`, and the shared unit registries.
    top = QuantSI
    for module_name in ("QuantSI.fundamentalunits", "QuantSI.allunits", "QuantSI.stdunits"):
        module = importlib.import_module(module_name)
        for name in SNAPSHOT["QuantSI"]:
            if hasattr(module, name):
                assert getattr(module, name) is getattr(top, name), f"{module_name}.{name}"
