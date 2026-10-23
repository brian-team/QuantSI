import contextlib
import importlib

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip the doctests of objects marked ``_do_not_run_doctests``.

    Such objects (e.g. Quantity.fill, the wrappers in unitsafefunctions) reuse a
    NumPy docstring, whose examples are about NumPy arrays. Brian2 uses the same
    marker.
    """

    # (Deselected rather than skipped: their docstrings have no line in our files,
    # which pytest needs to report a skip.)
    def marked(item):
        doctest = getattr(item, "dtest", None)
        return doctest is not None and getattr(
            _resolve(doctest.name), "_do_not_run_doctests", False
        )

    deselected = [item for item in items if marked(item)]
    if deselected:
        items[:] = [item for item in items if not marked(item)]
        config.hook.pytest_deselected(items=deselected)


def _resolve(dotted_name):
    """The object called ``dotted_name`` (module path, then attributes), or None."""
    parts = dotted_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute, None)
        return obj
    return None


@contextlib.contextmanager
def preserved_unit_registries():
    """Undo, on exit, every unit registered inside the ``with`` block.

    Units are registered globally, either explicitly with register_new_unit or
    implicitly whenever units are combined (``siemens / cm**2``), and that changes
    how quantities are displayed from then on.
    """
    from QuantSI.fundamentalunits import (
        additional_unit_register,
        standard_unit_register,
        user_unit_register,
    )

    registries = (standard_unit_register, user_unit_register, additional_unit_register)
    saved = [
        (dict(r.units), {dim: dict(units) for dim, units in r.units_for_dimensions.items()})
        for r in registries
    ]
    try:
        yield
    finally:
        for registry, (units, by_dimension) in zip(registries, saved, strict=True):
            registry.units.clear()
            registry.units.update(units)
            registry.units_for_dimensions.clear()
            registry.units_for_dimensions.update(by_dimension)
            registry._tables.clear()


@pytest.fixture(autouse=True)
def _isolated_unit_registries():
    """No test (or doctest) can change how quantities display in another one."""
    with preserved_unit_registries():
        yield
