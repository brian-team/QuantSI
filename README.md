# QuantSI
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/brian-team/QuantSI/actions/workflows/pytest-ubuntu.yaml/badge.svg)](https://github.com/brian-team/QuantSI/actions/workflows/pytest-ubuntu.yaml)
[![Coverage Status](https://coveralls.io/repos/github/brian-team/QuantSI/badge.svg?branch=main)](https://coveralls.io/github/brian-team/QuantSI?branch=main)

**Pragmatic dimensional checking for NumPy-based SI calculations.**

QuantSI attaches a physical dimension to NumPy arrays and checks, operation by
operation, that the physics makes sense. It is the unit system of the
[Brian2](https://briansimulator.org) simulator, packaged so that it can also be
used on its own.

QuantSI requires Python 3.13 or newer and NumPy 2.3 or newer. Minimum supported
versions follow [SPEC-0](https://scientific-python.org/specs/spec-0000/).

```python
>>> import numpy as np
>>> from QuantSI import Mohm, mV, nA, parse_quantity
>>> v = np.array([-70.0, -55.0]) * mV
>>> v
array([-70., -55.]) * mvolt
>>> 2 * nA * 50 * Mohm
100. * mvolt
>>> f"{v.max() / mV:.1f} mV"
'-55.0 mV'
>>> v + 1 * nA
Traceback (most recent call last):
    ...
DimensionMismatchError: Cannot calculate [-70. -55.] mV add 1. nA, the units do not match (units are V and A).
>>> parse_quantity("-70 * mV")
-70. * mvolt

```

## What QuantSI does

- Stores every value in base SI units, with one shared `Dimension` object per
  kind of quantity (so checking "same kind?" is a single identity check).
- Checks arithmetic, comparisons and the supported NumPy functions, and raises
  `DimensionMismatchError` when dimensions do not match.
- Displays values in a sensible unit (`3. mV`), parses simple unit expressions
  (`"3.5 * mV"`), and provides ~2000 named units generated from one rule.

## What QuantSI deliberately does not do

- No offset units (degrees Celsius/Fahrenheit) and no user-defined base dimensions.
- No unit-conversion contexts, definition files or a general units language.
- Values never remember the unit they were entered in: `3 * mV` *is* `0.003 V`.

If you need any of these, have a look at [pint](https://pint.readthedocs.io) or
[astropy.units](https://docs.astropy.org/en/stable/units/).
