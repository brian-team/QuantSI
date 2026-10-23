"""The unit registries, used to pick the unit a quantity is displayed in.

There are three of them, shared by the whole process, and they are searched in
this order: the standard units, units registered by the user, and extra
(compound) units. The goal is the unit in which a value reads best: ``3. mV``
rather than ``0.003 V``.

In this file: the ``UnitRegistry`` class, the three registries,
``register_new_unit``, ``get_unit`` (a unit for a dimension, made up if none is
registered) and ``get_unit_for_display`` (that unit's name as text).
"""

from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from .dimension import DIMENSIONLESS, Dimension

if TYPE_CHECKING:
    from .unit import Unit


class UnitRegistry:
    """
    Stores known units for printing in best units.

    All a user needs to do is to use the `register_new_unit`
    function.

    Default registries:

    The units module defines three registries, the standard units,
    user units, and additional units. Finding best units is done
    by first checking standard, then user, then additional. New
    user units are added by using the `register_new_unit` function.

    Standard units includes all the basic non-compound unit names
    built in to the module, including volt, amp, etc. Additional
    units defines some compound units like newton metre (Nm) etc.

    Methods
    -------
    add
    __getitem__
    """

    def __init__(self):
        self.units = collections.OrderedDict()
        self.units_for_dimensions = collections.defaultdict(dict)
        #: dim -> (scale of each matching unit as an array, the units), built on
        #: first use and dropped when a unit of that dimension is added.
        self._tables = {}

    def add(self, u):
        """Add a unit to the registry"""
        self.units[repr(u)] = u
        self.units_for_dimensions[u.dim][float(u)] = u
        self._tables.pop(u.dim, None)

    def _table(self, dim):
        table = self._tables.get(dim)
        if table is None:
            matching = self.units_for_dimensions[dim]
            table = self._tables[dim] = (np.asarray(list(matching)), list(matching.values()))
        return table

    def __getitem__(self, x):
        """Returns the best unit for quantity x

        The algorithm is to consider the value:

        m=abs(x/u)

        for all matching units u. We select the unit where this ratio is the
        closest to 10 (if it is an array with several values, we select the
        unit where the deviations from that are the smallest. More precisely,
        the unit that minimizes the sum of (log10(m)-1)**2 over all entries).
        """
        matching = self.units_for_dimensions.get(x.dim, {})
        if len(matching) == 0:
            raise KeyError("Unit not found in registry.")

        matching_values, matching_units = self._table(x.dim)
        if x.size == 1:  # most often, a single value (str(3 * mV))
            value = abs(np.asarray(x).item())
            if value == 0 or value != value:  # zero or NaN: any unit will do
                return matching[1.0]
            # The same arithmetic as below, for one value.
            deviations = (np.log10(value / matching_values) - 1) ** 2
            return matching_units[deviations.argmin()]

        print_opts = np.get_printoptions()
        edgeitems, threshold = print_opts["edgeitems"], print_opts["threshold"]
        if x.size > threshold:
            # Only care about optimizing the units for the values that will
            # actually be shown later
            # The code looks a bit complex, but should return the same numbers
            # that are shown by numpy's string conversion
            slices = []
            for shape in x.shape:
                if shape > 2 * edgeitems:
                    slices.append((slice(0, edgeitems), slice(-edgeitems, None)))
                else:
                    slices.append((slice(None),))
            values = np.asarray(x)  # plain numbers: the display unit is chosen by value
            x_flat = np.hstack(
                [values[use_slices].flatten() for use_slices in itertools.product(*slices)]
            )
        else:
            x_flat = np.asarray(x).flatten()
        floatreps = np.tile(np.abs(x_flat), (len(matching), 1)).T / matching_values
        # ignore zeros, they are well represented in any unit
        floatreps[floatreps == 0] = np.nan
        if np.all(np.isnan(floatreps)):
            return matching[1.0]  # all zeros, use the base unit

        deviations = np.nansum((np.log10(floatreps) - 1) ** 2, axis=0)
        return matching_units[deviations.argmin()]


#: `UnitRegistry` containing all the standard units (metre, kilogram, um2...)
standard_unit_register = UnitRegistry()
#: `UnitRegistry` containing additional units (newton*metre, farad / metre, ...)
additional_unit_register = UnitRegistry()
#: `UnitRegistry` containing all units defined by the user
user_unit_register = UnitRegistry()


def register_new_unit(u: Unit) -> None:
    """Register a new unit for automatic displaying of quantities

    Parameters
    ----------
    u : `Unit`
        The unit that should be registered.

    Examples
    --------
    >>> from QuantSI import *
    >>> 2.0 * farad / metre**2
    2. * metre ** -4 * kilogram ** -1 * second ** 4 * amp ** 2
    >>> register_new_unit(pfarad / mmetre**2)
    >>> 2.0 * farad / metre**2
    2000000. * pfarad / (mmetre ** 2)
    """
    user_unit_register.add(u)


def get_unit(d: Dimension) -> Unit:
    """
    Find an unscaled unit (e.g. `volt` but not `mvolt`) for a `Dimension`.

    Parameters
    ----------
    d : `Dimension`
        The dimension to find a unit for.

    Returns
    -------
    u : `Unit`
        A registered unscaled `Unit` for the dimensions ``d``, or a new `Unit`
        if no unit was found.
    """
    from .unit import Unit

    for unit_register in [
        standard_unit_register,
        user_unit_register,
        additional_unit_register,
    ]:
        if 1.0 in unit_register.units_for_dimensions[d]:
            return unit_register.units_for_dimensions[d][1.0]
    return Unit(1.0, dim=d)


def get_unit_for_display(d: Any) -> str:
    """
    Return a string representation of an appropriate unscaled unit or ``'1'``
    for a dimensionless quantity.

    Parameters
    ----------
    d : `Dimension` or int
        The dimension to find a unit for.

    Returns
    -------
    s : str
        A string representation of the respective unit or the string ``'1'``.
    """
    if (isinstance(d, int) and d == 1) or d is DIMENSIONLESS:
        return "1"
    else:
        return str(get_unit(d))
