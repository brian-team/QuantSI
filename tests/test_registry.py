"""Choosing the display unit: checked against the original algorithm."""

import itertools

import numpy as np
import pytest

from QuantSI.allunits import amp, farad, metre, second, siemens, volt
from QuantSI.fundamentalunits import UnitRegistry, standard_unit_register


def old_best_unit(registry, x):
    """UnitRegistry.__getitem__ as it was before it was made faster."""
    matching = registry.units_for_dimensions.get(x.dim, {})
    if len(matching) == 0:
        raise KeyError("Unit not found in registry.")
    matching_values = np.asarray(list(matching.keys()))
    print_opts = np.get_printoptions()
    edgeitems, threshold = print_opts["edgeitems"], print_opts["threshold"]
    if x.size > threshold:
        slices = []
        for shape in x.shape:
            if shape > 2 * edgeitems:
                slices.append((slice(0, edgeitems), slice(-edgeitems, None)))
            else:
                slices.append((slice(None),))
        values = np.asarray(x)
        x_flat = np.hstack([values[s].flatten() for s in itertools.product(*slices)])
    else:
        x_flat = np.asarray(x).flatten()
    floatreps = np.tile(np.abs(x_flat), (len(matching), 1)).T / matching_values
    floatreps[floatreps == 0] = np.nan
    if np.all(np.isnan(floatreps)):
        return matching[1.0]
    deviations = np.nansum((np.log10(floatreps) - 1) ** 2, axis=0)
    return list(matching.values())[deviations.argmin()]


def samples():
    rng = np.random.default_rng(3)
    scalars = list(10.0 ** rng.uniform(-26, 26, size=400)) + [0.0, -0.0, np.nan, np.inf, -np.inf]
    # Values exactly halfway (in log scale) between two prefixes: ties
    scalars += [10.0 ** (k + 0.5) for k in range(-12, 12)] + [-3.162e-4, 3.1622776601683795e-3]
    arrays = [rng.normal(size=n) * 10.0 ** rng.uniform(-15, 15) for n in (2, 5, 50, 1500)]
    arrays += [np.zeros(3), np.array([0.0, 1e-3, np.nan]), np.array([[1e-6, 2e-6], [3e-6, 4e-6]])]
    return scalars, arrays


@pytest.mark.parametrize("unit", [volt, amp, metre, second, farad, siemens])
def test_same_unit_as_the_original_algorithm(unit):
    scalars, arrays = samples()
    for value in scalars + arrays:
        q = value * unit
        assert standard_unit_register[q] is old_best_unit(standard_unit_register, q), value


def test_same_unit_with_other_print_options():
    scalars, arrays = samples()
    with np.printoptions(threshold=3, edgeitems=1):
        for value in arrays:
            q = value * volt
            assert standard_unit_register[q] is old_best_unit(standard_unit_register, q)


def test_registering_a_unit_updates_the_choice():
    registry = UnitRegistry()
    registry.add(volt)
    q = 0.002 * volt
    assert registry[q] is volt
    from QuantSI.allunits import mvolt

    registry.add(mvolt)
    assert registry[q] is mvolt


def test_registrations_are_undone_between_tests():
    from conftest import preserved_unit_registries
    from QuantSI.allunits import mmetre, pfarad
    from QuantSI.fundamentalunits import register_new_unit

    q = 2.0 * farad / metre**2
    before = str(q)
    with preserved_unit_registries():
        register_new_unit(pfarad / mmetre**2)
        assert str(q) != before
    assert str(q) == before


def test_old_best_unit_method_warns_and_forwards_registries():
    from numpy.exceptions import VisibleDeprecationWarning

    registry = UnitRegistry()
    registry.add(volt)
    q = 0.002 * volt
    with pytest.warns(VisibleDeprecationWarning, match="renamed to Quantity.get_best_unit"):
        assert q._get_best_unit(registry) is q.get_best_unit(registry) is volt
