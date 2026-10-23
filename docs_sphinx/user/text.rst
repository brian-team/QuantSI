Quantities and text
===================

Writing quantities
------------------

A quantity is written in the unit in which its values read best::

    >>> from QuantSI import mV, ms, nA
    >>> print(3 * mV)
    3. mV
    >>> 3 * mV                 # repr: a Python expression that gives the value back
    3. * mvolt
    >>> (25.123456 * mV).in_unit(mV, 3)
    '25.123 mV'

The numbers are formatted by NumPy and follow its print options
(``np.set_printoptions``, ``np.printoptions``). An empty format specification
keeps the display unit. An explicit numeric format specification formats a
scalar's stored value in base SI units, without a unit suffix::

    >>> f"{3 * mV}"
    '3. mV'
    >>> f"{3 * mV:.2f}"
    '0.00'
    >>> f"{1 * ms:g}"
    '0.001'

QuantSI stores values in base SI units and does not remember the unit used to
create them. To format in a chosen unit, divide by that unit and append its name::

    >>> f"{(3 * mV) / mV:.2f} mV"
    '3.00 mV'

Non-empty numeric format specifications on arrays raise ``TypeError``, as they
do for NumPy arrays.

To get the bare value in base SI units, convert explicitly: ``float(q)``,
``np.asarray(q)``. Dividing by a unit (``q / mV``) gives the value in that unit.

Jupyter notebooks show quantities as LaTeX (``q._repr_latex_()``), and SymPy's
``latex()`` can print them as well.

Reading quantities
------------------

`parse_quantity` reads a quantity from text, and `parse_dimensions` its
dimensions::

    >>> from QuantSI import parse_quantity, parse_dimensions
    >>> parse_quantity("-70 * mV")
    -70. * mvolt
    >>> parse_quantity("[1, 2.5] * ms")
    array([1. , 2.5]) * msecond

The text is a small language, not Python. It allows exactly:

==================================  =========================================
numbers                             ``3``, ``-70``, ``1e-9``, ``2.5``
unit names                          every name in ``QuantSI.allunits`` and
                                    ``QuantSI.stdunits`` (``mV``, ``siemens``)
products and quotients              ``0.5 * siemens / cm ** 2``
powers, with a plain number         ``metre ** -2``, ``second ** 0.5``
signs and parentheses               ``-(3 * mV)``
lists of numbers                    ``[1, 2] * mV``, ``array([[1, 2], [3, 4]]) * nA``
==================================  =========================================

Everything else is refused with a ``QuantityParseError`` (a ``ValueError``),
including function calls, attributes and text without an operator (``"3.5 mV"``
needs to be ``"3.5 * mV"``). The text is never executed: it is parsed into a
syntax tree whose nodes are checked one by one, and its length, size and
exponents are limited. A name that is not a unit gets suggestions:
``'mv' is not a known unit. Did you mean mV?``

Like the same expression in Python, combining units in the text
(``siemens / cm ** 2``) makes that combination available for display.

Reading back what ``repr`` writes
---------------------------------

``parse_quantity(repr(q))`` gives back ``q``, with the same interned
dimensions, for scalars and for arrays that NumPy prints in full, up to the
precision NumPy prints with: by default 8 digits after the decimal point, in the
unit chosen for display. Print with ``np.printoptions(precision=17)`` for an exact
round trip of any value.

Not promised: arrays that NumPy abbreviates (``[1., 2., ..., 999.]``), values
containing NaN or infinity, and the array's dtype. Text is for people; to store
quantities, save ``np.asarray(q)`` together with ``q.dim``, or pickle them.
