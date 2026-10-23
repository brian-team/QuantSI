"""Make a Brian2 checkout use QuantSI: turn brian2/units/*.py into facades.

    python tools/brian2/make_facade.py path/to/brian2

Each module of ``brian2.units`` becomes a few lines that re-export the QuantSI
module of the same name; any other name (a private helper, a registry) is
looked up on the QuantSI module (PEP 562), so existing imports keep working.
``allunits`` keeps Brian2's ``celsius`` placeholder. This is what the companion
change in Brian2 looks like; CI uses it to run Brian2's test suite against
QuantSI.
"""

import pathlib
import sys

FORWARD = """

def __getattr__(name):
    # Names that are not star-exported (private helpers, registries, ...) are
    # looked up on the QuantSI module, so that existing imports keep working.
    try:
        return getattr(_quantsi_module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(dir(_quantsi_module)))
"""


def facade(module, *, exports="__all__", extra=""):
    lines = [
        f'"""Brian2\'s units are provided by QuantSI: this module re-exports ``QuantSI.{module}``."""',
        "",
        f"import QuantSI.{module} as _quantsi_module",
        f"from QuantSI.{module} import *  # noqa: F403",
    ]
    if exports == "__all__":
        lines.append(f"from QuantSI.{module} import __all__  # noqa: F401")
    return "\n".join(lines) + "\n" + extra + FORWARD


def main(brian2_root):
    units = pathlib.Path(brian2_root) / "brian2" / "units"
    for module in ("fundamentalunits", "stdunits", "unitsafefunctions"):
        (units / f"{module}.py").write_text(facade(module))
    (units / "constants.py").write_text(facade("constants", exports=None))
    original = (units / "allunits.py").read_text()
    start = original.index("class _Celsius")
    end = original.index("celsius = _Celsius()") + len("celsius = _Celsius()")
    celsius = original[start:end]
    extra = (
        "from QuantSI.allunits import __all__ as _quantsi_all\n\n\n"
        + celsius
        + "\n\n__all__ = [*_quantsi_all, 'celsius']\n"
    )
    (units / "allunits.py").write_text(facade("allunits", exports=None, extra=extra))
    print(f"brian2.units in {units} now re-exports QuantSI")


if __name__ == "__main__":
    main(sys.argv[1])
