"""Write the list of names Brian2's unit modules offer, used by ``test_brian2_surface.py``.

Brian2's ``brian2.units`` is to become a facade over QuantSI, so QuantSI must
offer every name those modules offer today. Run this with a checkout of Brian2
*before* that change, in an environment that has Brian2's dependencies::

    python tools/snapshot_brian2_surface.py path/to/brian2

The Brian2 commit is recorded in the file.
"""

import importlib
import inspect
import json
import pathlib
import subprocess
import sys

MODULES = [
    "brian2.units",
    "brian2.units.fundamentalunits",
    "brian2.units.allunits",
    "brian2.units.stdunits",
    "brian2.units.constants",
    "brian2.units.unitsafefunctions",
]

OUTPUT = pathlib.Path(__file__).parent.parent / "tests/data/brian2_units_surface.json"


def is_own_name(module, name):
    """Names that Brian2 itself provides (not incidental imports such as ``np``)."""
    if name.startswith("__"):
        return False
    obj = getattr(module, name)
    if inspect.ismodule(obj):
        return False
    owner = getattr(obj, "__module__", None)
    return owner is None or owner.startswith("brian2")


def snapshot(brian2_dir):
    sys.path.insert(0, str(brian2_dir))
    surface = {}
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        if not pathlib.Path(module.__file__).is_relative_to(brian2_dir):
            sys.exit(f"{module_name} was imported from {module.__file__}, not from {brian2_dir}")
        surface[module_name] = sorted(n for n in dir(module) if is_own_name(module, n))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=brian2_dir, capture_output=True, text=True, check=True
    ).stdout.strip()
    surface["brian2 commit"] = commit
    return surface


if __name__ == "__main__":
    surface = snapshot(pathlib.Path(sys.argv[1]).resolve())
    OUTPUT.write_text(json.dumps(surface, indent=1, sort_keys=True))
