"""The Unit class: a named, scaled quantity such as ``mvolt``.

A unit is a Quantity with a name (``mvolt``), a display name (``mV``), a LaTeX name
and a scale (the power of ten: -3 for milli). A new Unit registers itself for
display (in the user registry) unless ``Unit.automatically_register_units`` is
off; allunits.py turns it off while it builds its units with ``Unit.create``, and
then adds them all to the standard registry.

In this file: the Unit class, with its methods under these headings: creating
units, names, text, arithmetic, and comparison and hashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from .dimension import DIMENSIONLESS, Dimension, _siprefixes, is_dimensionless, is_scalar_type
from .quantity import Quantity
from .registry import register_new_unit
from .utils import _latex_number

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Unit(Quantity):
    r"""
     A physical unit.

     Normally, you do not need to worry about the implementation of
     units. They are derived from the `Quantity` object with
     some additional information (name and string representation).

     Basically, a unit is just a number with given dimensions, e.g.
     mvolt = 0.001 with the dimensions of voltage. The units module
     defines a large number of standard units, and you can also define
     your own (see below).

     The unit class also keeps track of various things that were used
     to define it so as to generate a nice string representation of it.
     See below.

     When creating scaled units, you can use the following prefixes:

      ======     ======  ==============
      Factor     Name    Prefix
      ======     ======  ==============
      10^24      yotta   Y
      10^21      zetta   Z
      10^18      exa     E
      10^15      peta    P
      10^12      tera    T
      10^9       giga    G
      10^6       mega    M
      10^3       kilo    k
      10^2       hecto   h
      10^1       deka    da
      1
      10^-1      deci    d
      10^-2      centi   c
      10^-3      milli   m
      10^-6      micro   u (\mu in SI)
      10^-9      nano    n
      10^-12     pico    p
      10^-15     femto   f
      10^-18     atto    a
      10^-21     zepto   z
      10^-24     yocto   y
      ======     ======  ==============

    **Defining your own**

     It can be useful to define your own units for printing
     purposes. So for example, to define the newton metre, you
     write

     >>> from QuantSI import *
     >>> from QuantSI.allunits import newton
     >>> Nm = newton * metre

     You can then do

     >>> (1 * Nm).in_unit(Nm)
     '1. N m'

     New "compound units", i.e. units that are composed of other units will be
     automatically registered and from then on used for display. For example,
     imagine you define total conductance for a membrane, and the total area of
     that membrane:

     >>> conductance = 10.0 * nS
     >>> area = 20000 * um**2

     If you now ask for the conductance density, you will get an "ugly" display
     in basic SI dimensions, as Brian does not know of a corresponding unit:

     >>> conductance / area
     0.5 * metre ** -4 * kilogram ** -1 * second ** 3 * amp ** 2

     By using an appropriate unit once, it will be registered and from then on
     used for display when appropriate:

     >>> usiemens / cm**2
     usiemens / (cmetre ** 2)
     >>> conductance / area  # same as before, but now Brian knows about uS/cm^2
     50. * usiemens / (cmetre ** 2)

     Note that user-defined units cannot override the standard units (`volt`,
     `second`, etc.) that are predefined by Brian. For example, the unit
     ``Nm`` has the dimensions "length²·mass/time²", and therefore the same
     dimensions as the standard unit `joule`. The latter will be used for display
     purposes:

     >>> 3 * joule
     3. * joule
     >>> 3 * Nm
     3. * joule

    """

    __slots__ = ["dim", "scale", "_dispname", "_name", "_latexname", "iscompound"]

    __array_priority__ = 100

    if TYPE_CHECKING:
        # Only type checkers see these (see Quantity): a unit combined with a unit
        # gives a unit (volt * amp, volt / amp, volt ** 2), anything else a Quantity.
        @overload  # type: ignore[override]
        def __mul__(self, other: Unit, /) -> Unit: ...
        @overload
        def __mul__(self, other: ArrayLike, /) -> Quantity: ...
        def __mul__(self, other: Any, /) -> Any: ...
        @overload  # type: ignore[override]
        def __truediv__(self, other: Unit, /) -> Unit: ...
        @overload
        def __truediv__(self, other: ArrayLike, /) -> Quantity: ...
        def __truediv__(self, other: Any, /) -> Any: ...
        def __pow__(self, other: ArrayLike, /) -> Unit: ...  # type: ignore[override, unused-ignore]  # NumPy 2.5's stubs need it, 2.2's don't

    #### Creating units ####
    automatically_register_units = True

    def __new__(
        cls,
        arr,
        dim=None,
        scale=0,
        name=None,
        dispname=None,
        latexname=None,
        iscompound=False,
        dtype=None,
        copy=False,
    ):
        if dim is None:
            dim = DIMENSIONLESS
        obj = super().__new__(cls, arr, dim=dim, dtype=dtype, copy=copy, force_quantity=True)
        return obj

    def __array_finalize__(self, orig):
        self.dim = getattr(orig, "dim", DIMENSIONLESS)
        self.scale = getattr(orig, "scale", 0)
        self._name = getattr(orig, "_name", "")
        self._dispname = getattr(orig, "_dispname", "")
        self._latexname = getattr(orig, "_latexname", "")
        self.iscompound = getattr(orig, "_iscompound", False)
        return self

    def __init__(
        self,
        value,
        dim=None,
        scale=0,
        name=None,
        dispname=None,
        latexname="",
        iscompound=False,
    ):
        if value != 10.0**scale:
            raise AssertionError(f"Unit value has to be 10**scale (scale={scale}, value={value})")
        if dim is None:
            dim = DIMENSIONLESS
        self.dim = dim  #: The Dimensions of this unit

        #: The scale for this unit (as the integer exponent of 10), i.e.
        #: a scale of 3 means 10^3, e.g. for a "k" prefix.
        self.scale = scale
        if name is None:
            if dim is DIMENSIONLESS:
                name = "Unit(1)"
            else:
                name = repr(dim)
        if dispname is None:
            if dim is DIMENSIONLESS:
                dispname = "1"
            else:
                dispname = str(dim)
        #: The full name of this unit.
        self._name = name
        #: The display name of this unit.
        self._dispname = dispname
        #: A LaTeX expression for the name of this unit.
        self._latexname = latexname
        #: Whether this unit is a combination of other units.
        self.iscompound = iscompound

        if Unit.automatically_register_units:
            register_new_unit(self)

    @staticmethod
    def create(
        dim: Dimension, name: str, dispname: str, latexname: str | None = None, scale: int = 0
    ) -> Unit:
        """
        Create a new named unit.

        Parameters
        ----------
        dim : `Dimension`
            The dimensions of the unit.
        name : `str`
            The full name of the unit, e.g. ``'volt'``
        dispname : `str`
            The display name, e.g. ``'V'``
        latexname : str, optional
            The name as a LaTeX expression (math mode is assumed, do not add
            $ signs or similar), e.g. ``'\\omega'``. If no `latexname` is
            specified, `dispname` will be used.
        scale : int, optional
            The scale of this unit as an exponent of 10, e.g. -3 for a unit that
            is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).

        Returns
        -------
        u : `Unit`
            The new unit.
        """
        name = str(name)
        dispname = str(dispname)
        if latexname is None:
            latexname = f"\\mathrm{{{dispname}}}"

        u = Unit(
            10.0**scale,
            dim=dim,
            scale=scale,
            name=name,
            dispname=dispname,
            latexname=latexname,
        )

        return u

    @staticmethod
    def create_scaled_unit(baseunit: Unit, scalefactor: str) -> Unit:
        """
        Create a scaled unit from a base unit.

        Parameters
        ----------
        baseunit : `Unit`
            The unit of which to create a scaled version, e.g. ``volt``,
            ``amp``.
        scalefactor : `str`
            The scaling factor, e.g. ``"m"`` for mvolt, mamp

        Returns
        -------
        u : `Unit`
            The new unit.
        """
        name = scalefactor + baseunit.name
        dispname = scalefactor + baseunit.dispname
        scale = _siprefixes[scalefactor] + baseunit.scale
        if scalefactor == "u":
            scalefactor = r"\mu"
        latexname = f"\\mathrm{{{scalefactor}}}{baseunit.latexname}"

        u = Unit(
            10.0**scale,
            dim=baseunit.dim,
            name=name,
            dispname=dispname,
            latexname=latexname,
            scale=scale,
        )

        return u

    #### Names ####
    name = property(fget=lambda self: self._name, doc="The name of the unit")

    dispname = property(
        fget=lambda self: self._dispname,
        doc="The display name of the unit",
    )

    latexname = property(
        fget=lambda self: self._latexname,
        doc="The LaTeX name of the unit",
    )

    #### Text ####
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.dispname

    def _latex(self, *args):
        return self.latexname

    def _repr_latex_(self):
        return f"${self._latex()}$"

    #### Arithmetic (units are constants: in-place operations are refused) ####
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        if ufunc.__name__ == "multiply":
            first, second = inputs
            if isinstance(first, Unit) and isinstance(second, Unit):
                name = f"{first.name} * {second.name}"
                dispname = f"{self.dispname} {second.dispname}"
                latexname = f"{first.latexname}\\,{second.latexname}"
                scale = first.scale + second.scale
                u = Unit(
                    10.0**scale,
                    dim=first.dim * second.dim,
                    name=name,
                    dispname=dispname,
                    latexname=latexname,
                    iscompound=True,
                    scale=scale,
                )
                return u
            else:
                # Not a unit-by-unit operation: the result is an ordinary Quantity.
                return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        elif ufunc.__name__ == "divide":
            first, second = inputs
            if isinstance(first, Unit) and isinstance(second, Unit):
                if first.iscompound:
                    dispname = f"({self.dispname})"
                    name = f"({self.name})"
                else:
                    dispname = self.dispname
                    name = self.name
                dispname += "/"
                name += " / "
                if second.iscompound:
                    dispname += f"({second.dispname})"
                    name += f"({second.name})"
                else:
                    dispname += second.dispname
                    name += second.name

                latexname = rf"\frac{{{first.latexname}}}{{{second.latexname}}}"
                scale = first.scale - second.scale
                u = Unit(
                    10.0**scale,
                    dim=first.dim / second.dim,
                    name=name,
                    dispname=dispname,
                    latexname=latexname,
                    scale=scale,
                    iscompound=True,
                )
                return u
            elif is_dimensionless(first) and np.array(first).shape == () and first == 1:
                return np.reciprocal(second)
            else:
                # Not a unit-by-unit operation: the result is an ordinary Quantity.
                return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        elif ufunc.__name__ == "power":
            first, second = inputs
            if is_scalar_type(second):
                if first.iscompound:
                    dispname = f"({first.dispname})"
                    name = f"({first.name})"
                    latexname = r"\left(%s\right)" % first.latexname
                else:
                    dispname = first.dispname
                    name = first.name
                    latexname = first.latexname
                dispname += f"^{str(second)}"
                name += f" ** {repr(second)}"
                latexname += "^{%s}" % _latex_number(second)
                scale = first.scale * second
                u = Unit(
                    10.0**scale,
                    dim=first.dim**second,
                    name=name,
                    dispname=dispname,
                    latexname=latexname,
                    scale=scale,
                    iscompound=True,
                )  # To avoid issues with units like (second ** -1) ** -1
                return u
            else:
                return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        elif ufunc.__name__ == "square":
            if self.iscompound:
                dispname = f"({self.dispname})"
                name = f"({self.name})"
                latexname = r"\left(%s\right)" % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += "^2"
            name += " ** 2"
            latexname += "^2"
            scale = self.scale * 2
            u = Unit(
                10.0**scale,
                dim=self.dim**2,
                name=name,
                dispname=dispname,
                latexname=latexname,
                scale=scale,
                iscompound=True,
            )
            return u
        elif ufunc.__name__ == "sqrt":
            if self.iscompound:
                dispname = f"({self.dispname})"
                name = f"({self.name})"
                latexname = r"\left(%s\right)" % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += "^0.5"
            name += " ** 0.5"
            latexname += "^0.5"
            scale = self.scale / 2
            u = Unit(
                10.0**scale,
                dim=self.dim**0.5,
                name=name,
                dispname=dispname,
                latexname=latexname,
                scale=scale,
                iscompound=True,
            )
            return u
        elif ufunc.__name__ == "reciprocal":
            if self.iscompound:
                dispname = f"({self.dispname})"
                name = f"({self.name})"
                latexname = r"\left(%s\right)" % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += "^-1"
            name += " ** -1"
            latexname += "^{-1}"
            scale = -self.scale
            u = Unit(
                10.0**scale,
                dim=self.dim**-1,
                name=name,
                dispname=dispname,
                latexname=latexname,
                scale=scale,
                iscompound=True,
            )
            return u
        else:
            # Treat the unit as a Quantity (e.g. meter + meter should not fail but give 2*meter)
            return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __iadd__(self, other):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    def __isub__(self, other):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    def __imul__(self, other):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    def __itruediv__(self, other):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    def __ifloordiv__(self, other):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    def __imod__(self, other):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    def __ipow__(self, other, modulo=None):  # type: ignore[misc]  # units are immutable
        raise TypeError("Units cannot be modified in-place")

    #### Comparison and hashing ####
    def __eq__(self, other):
        if isinstance(other, Unit):
            return other.dim is self.dim and other.scale == self.scale
        else:
            return Quantity.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.dim, self.scale))
