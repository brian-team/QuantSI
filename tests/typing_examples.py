"""The types of QuantSI's API, as a type checker sees them.

mypy checks this file in CI (``mypy``, see pyproject.toml); pytest does not
collect it.

Arithmetic on quantities gives a Quantity, and a unit combined with a unit gives a
Unit. Two cases where the types are approximate: a result whose dimensions cancel,
like ``(3 * mV) / (1 * mV)``, is a plain number at run time (as everywhere in
QuantSI), and pyright (unlike mypy) types ``array * unit`` as a plain array, because
it uses the array's operator; ``unit * array`` gives a Quantity in both.
"""

from typing import assert_type

import numpy as np

from QuantSI import (
    Quantity,
    Unit,
    amp,
    get_dimensions,
    metre,
    mV,
    mvolt,
    parse_dimensions,
    second,
    uV,
    volt,
)
from QuantSI.fundamentalunits import Dimension, get_or_create_dimension

assert_type(volt, Unit)
assert_type(mvolt, Unit)
assert_type(uV, Unit)
assert_type(volt.dim, Dimension)
assert_type(get_dimensions(volt), Dimension)
assert_type(get_or_create_dimension(m=1), Dimension)
assert_type(parse_dimensions("3 * mV"), Dimension)
assert_type(Unit.create(volt.dim, "myvolt", "myV"), Unit)
assert_type(volt.in_unit(mvolt), str)
assert_type(volt.dim * volt.dim, Dimension)


def needs_a_quantity(q: Quantity) -> str:
    return q.in_best_unit()


needs_a_quantity(mvolt)  # a Unit is a Quantity

# Arithmetic on quantities gives quantities ...
assert_type(3 * mV, Quantity)
assert_type(mV * 3, Quantity)
assert_type(3 * mV + 4 * mV, Quantity)
assert_type((3 * mV) - (1 * mV), Quantity)
assert_type((3 * mV) / (1 * second), Quantity)
assert_type((3 * mV) ** 2, Quantity)
assert_type(-(3 * mV), Quantity)
assert_type(abs(3 * mV), Quantity)
assert_type(metre * np.ones(3), Quantity)
needs_a_quantity(3 * mV + 4 * mV)
assert_type((3 * mV + 4 * mV).in_unit(volt), str)

# ... and units combined with units give units.
assert_type(volt * amp, Unit)
assert_type(volt / amp, Unit)
assert_type(volt**2, Unit)
assert_type(3 * volt, Quantity)
