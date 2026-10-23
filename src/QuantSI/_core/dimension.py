"""Physical dimensions: the exponents of the seven SI base units.

A `Dimension` says what kind of quantity something is (a length, a voltage,
...), not how much of it there is. `get_or_create_dimension` makes sure there is
only one `Dimension` object for each combination of exponents, so checking
whether two things have the same dimensions is just ``dim1 is dim2``.

In this file, in order:

1. the SI base units and prefixes (``_di``, ``_ilabel``, ``_iclass_label``,
   ``_siprefixes``)
2. the ``Dimension`` class
3. ``DIMENSIONLESS`` and ``get_or_create_dimension``, which hands out the one
   ``Dimension`` object for each set of exponents
4. the caches that remember the results of ``Dimension`` arithmetic
5. helpers that work on any object: ``is_scalar_type``, ``get_dimensions``,
   ``is_dimensionless``, ``have_same_dimensions`` and
   ``fail_for_dimension_mismatch``
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from .errors import DimensionMismatchError
from .utils import _short_str, set_module

# The seven SI base dimensions (length, mass, time, current, temperature,
# amount of substance, luminosity) and various descriptions;
# each description maps to an index i, and the power of each dimension
# is stored in the variable dims[i]
_di = {
    "Length": 0,
    "length": 0,
    "metre": 0,
    "metres": 0,
    "meter": 0,
    "meters": 0,
    "m": 0,
    "Mass": 1,
    "mass": 1,
    "kilogram": 1,
    "kilograms": 1,
    "kg": 1,
    "Time": 2,
    "time": 2,
    "second": 2,
    "seconds": 2,
    "s": 2,
    "Electric Current": 3,
    "electric current": 3,
    "Current": 3,
    "current": 3,
    "ampere": 3,
    "amperes": 3,
    "A": 3,
    "Temperature": 4,
    "temperature": 4,
    "kelvin": 4,
    "kelvins": 4,
    "K": 4,
    "Quantity of Substance": 5,
    "Quantity of substance": 5,
    "quantity of substance": 5,
    "Substance": 5,
    "substance": 5,
    "mole": 5,
    "moles": 5,
    "mol": 5,
    "Luminosity": 6,
    "luminosity": 6,
    "candle": 6,
    "candles": 6,
    "cd": 6,
}

_ilabel = ["m", "kg", "s", "A", "K", "mol", "cd"]

# The same labels with the names used for constructing them in Python code
_iclass_label = ["metre", "kilogram", "second", "amp", "kelvin", "mole", "candle"]

# The SI prefixes, as powers of ten (milli is -3).
_siprefixes = {
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "": 0,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
}


class Dimension:
    """
    Stores the indices of the 7 basic SI unit dimension (length, mass, etc.).

    Provides a subset of arithmetic operations appropriate to dimensions:
    multiplication, division and powers, and equality testing.

    Parameters
    ----------
    dims : sequence of `float`
        The dimension indices of the 7 basic SI unit dimensions.

    Notes
    -----
    Users shouldn't use this class directly, it is used internally in Quantity
    and Unit. Even internally, never use ``Dimension(...)`` to create a new
    instance, use `get_or_create_dimension` instead. This function makes
    sure that only one Dimension instance exists for every combination of
    indices, allowing for a very fast dimensionality check with ``is``.
    """

    __slots__ = ["_dims"]

    __array_priority__ = 1000
    #### INITIALISATION ####

    def __init__(self, dims):
        self._dims = dims

    #### METHODS ####
    def get_dimension(self, d):
        """
        Return a specific dimension.

        Parameters
        ----------
        d : `str`
            A string identifying the SI basic unit dimension. Can be either a
            description like "length" or a basic unit like "m" or "metre".

        Returns
        -------
        dim : `float`
            The dimensionality of the dimension `d`.
        """
        return self._dims[_di[d]]

    @property
    def is_dimensionless(self):
        """
        Whether this Dimension is dimensionless.

        Notes
        -----
        Normally, instead one should check dimension for being identical to
        `DIMENSIONLESS`.
        """
        return all([x == 0 for x in self._dims])

    @property
    def dim(self):
        """
        Returns the `Dimension` object itself. This can be useful, because it
        allows to check for the dimension of an object by checking its ``dim``
        attribute -- this will return a `Dimension` object for `Quantity`,
        `Unit` and `Dimension`.
        """
        return self

    #### REPRESENTATION ####
    def _str_representation(self, python_code=False):
        """
        String representation in basic SI units, or ``"1"`` for dimensionless.
        Use ``python_code=False`` for display purposes and ``True`` for valid
        Python code.
        """

        if python_code:
            power_operator = " ** "
        else:
            power_operator = "^"

        parts = []
        for i in range(len(self._dims)):
            if self._dims[i]:
                if python_code:
                    s = _iclass_label[i]
                else:
                    s = _ilabel[i]
                if self._dims[i] != 1:
                    s += power_operator + str(self._dims[i])
                parts.append(s)
        if python_code:
            s = " * ".join(parts)
            if not len(s):
                return f"{self.__class__.__name__}()"
        else:
            s = " ".join(parts)
            if not len(s):
                return "1"
        return s.strip()

    def _latex(self, *args):
        parts = []
        for i in range(len(self._dims)):
            if self._dims[i]:
                s = _ilabel[i]
                if self._dims[i] != 1:
                    s += "^{%s}" % str(self._dims[i])
                parts.append(s)
        s = "\\,".join(parts)
        if not len(s):
            return "1"
        return s.strip()

    def _repr_latex(self):
        return f"${self._latex()}$"

    def __repr__(self):
        return self._str_representation(python_code=True)

    def __str__(self):
        return self._str_representation(python_code=False)

    #### ARITHMETIC ####
    # Note that none of the dimension arithmetic objects do sanity checking
    # on their inputs, although most will throw an exception if you pass the
    # wrong sort of input
    #
    # Every multiplication or division of quantities multiplies or divides their
    # dimensions, so the results are cached (see _remember below).
    def __mul__(self, value: Dimension) -> Dimension:
        key = (id(self), id(value))
        entry = _products.get(key)
        if entry is None:
            result = get_or_create_dimension([x + y for x, y in zip(self._dims, value._dims)])
            return _remember(_products, key, (self, value, result))
        return entry[-1]

    def __div__(self, value: Dimension) -> Dimension:
        key = (id(self), id(value))
        entry = _quotients.get(key)
        if entry is None:
            result = get_or_create_dimension([x - y for x, y in zip(self._dims, value._dims)])
            return _remember(_quotients, key, (self, value, result))
        return entry[-1]

    def __truediv__(self, value: Dimension) -> Dimension:
        return self.__div__(value)

    def __pow__(self, value: float | np.ndarray) -> Dimension:
        value = np.asarray(value)
        if value.size > 1:
            raise TypeError("Too many exponents")
        key = (id(self), value.item())
        entry = _powers.get(key)
        if entry is None:
            result = get_or_create_dimension([x * value for x in self._dims])
            return _remember(_powers, key, (self, result))
        return entry[-1]

    def __imul__(self, value):  # type: ignore[misc]  # dimensions are immutable
        raise TypeError("Dimension object is immutable")

    def __ifloordiv__(self, value):
        raise TypeError("Dimension object is immutable")

    def __itruediv__(self, value):  # type: ignore[misc]  # dimensions are immutable
        raise TypeError("Dimension object is immutable")

    def __ipow__(self, value):  # type: ignore[misc]  # dimensions are immutable
        raise TypeError("Dimension object is immutable")

    #### COMPARISON ####
    def __eq__(self, value):
        if value is self:  # interned: the common case
            return True
        try:
            other = value._dims
        except AttributeError:
            # Only compare equal to another Dimensions object
            return False
        # The test of np.allclose(self._dims, other), |a - b| <= atol + rtol * |b|,
        # written out for seven numbers: ~20x faster than calling NumPy.
        return all(
            a == b or abs(a - b) <= 1e-08 + 1e-05 * abs(b)
            for a, b in zip(self._dims, other, strict=True)
        )

    def __ne__(self, value):
        return not self.__eq__(value)

    def __hash__(self):
        return hash(self._dims)

    #### MAKE DIMENSION PICKABLE ####
    def __reduce__(self):
        # Make sure that unpickling Dimension objects does not bypass the singleton system
        return (get_or_create_dimension, (self._dims,))

    ### Dimension objects are singletons and deepcopy is therefore not necessary
    def __deepcopy__(self, memodict):
        return self


#: The singleton object for dimensionless Dimensions.
DIMENSIONLESS = Dimension((0, 0, 0, 0, 0, 0, 0))

#: The one Dimension object for each set of exponents; get_or_create_dimension
#: fills it.
_dimensions = {(0, 0, 0, 0, 0, 0, 0): DIMENSIONLESS}


@set_module("QuantSI.fundamentalunits")
def get_or_create_dimension(*args, **kwds) -> Dimension:
    """
    Create a new Dimension object or get a reference to an existing one.
    This function takes care of only creating new objects if they were not
    created before and otherwise returning a reference to an existing object.
    This allows to compare dimensions very efficiently using ``is``.

    Parameters
    ----------
    args : sequence of `float`
        A sequence with the indices of the 7 elements of an SI dimension.
    kwds : keyword arguments
        a sequence of ``keyword=value`` pairs where the keywords are the names of
        the SI dimensions, or the standard unit.

    Examples
    --------
    The following are all definitions of the dimensions of force

    >>> from QuantSI import *
    >>> get_or_create_dimension(length=1, mass=1, time=-2)
    metre * kilogram * second ** -2
    >>> get_or_create_dimension(m=1, kg=1, s=-2)
    metre * kilogram * second ** -2
    >>> get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
    metre * kilogram * second ** -2

    Notes
    -----
    The 7 units are (in order):

    Length, Mass, Time, Electric Current, Temperature,
    Quantity of Substance, Luminosity

    and can be referred to either by these names or their SI unit names,
    e.g. length, metre, and m all refer to the same thing here.
    """
    if len(args):
        # initialisation by list
        dims = args[0]
        try:
            if len(dims) != 7:
                raise TypeError()
        except TypeError:
            raise TypeError("Need a sequence of exactly 7 items")
    else:
        # initialisation by keywords
        dims = [0, 0, 0, 0, 0, 0, 0]
        for k in kwds:
            # _di stores the index of the dimension with name 'k'
            dims[_di[k]] = kwds[k]

    dims = tuple(dims)

    # check whether this Dimension object has already been created
    if dims in _dimensions:
        return _dimensions[dims]
    else:
        new_dim = Dimension(dims)
        _dimensions[dims] = new_dim
        return new_dim


#: Caches for Dimension arithmetic, keyed by the operands' ``id()``. An id is only
#: unique while its object is alive, so every entry also holds the operands
#: themselves: while the entry exists, no other object can get the same id.
#: (Interned dimensions live for the whole process anyway.)
_products: dict[tuple[int, int], tuple] = {}
_quotients: dict[tuple[int, int], tuple] = {}
_powers: dict[tuple[int, float], tuple] = {}
_MAX_CACHE_ENTRIES = 4096  # per cache; beyond that, results are computed every time


def _remember(cache, key, entry):
    """Store ``entry`` (operands..., result) unless the cache is full; return the result."""
    if len(cache) < _MAX_CACHE_ENTRIES:
        cache[key] = entry
    return entry[-1]


def is_scalar_type(obj: Any) -> bool:
    """
    Tells you if the object is a 1d number type.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    scalar : `bool`
        ``True`` if `obj` is a scalar that can be interpreted as a
        dimensionless `Quantity`.
    """
    try:
        return obj.ndim == 0 and is_dimensionless(obj)
    except AttributeError:
        return np.isscalar(obj) and not isinstance(obj, str)


def get_dimensions(obj: Any) -> Dimension:
    """
    Return the dimensions of any object that has them.

    Slightly more general than `Quantity.dimensions` because it will
    return `DIMENSIONLESS` if the object is of number type but not a `Quantity`
    (e.g. a `float` or `int`).

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    dim : `Dimension`
        The physical dimensions of the `obj`.
    """
    try:
        return obj.dim
    except AttributeError:
        # The following is not very pretty, but it will avoid the costly
        # isinstance check for the common types
        if type(obj) in [
            bool,
            int,
            float,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.ndarray,
            np.bool_,
        ] or isinstance(obj, (numbers.Number, np.number, np.ndarray)):
            return DIMENSIONLESS
        from .quantity import Quantity

        try:
            return Quantity(obj).dim
        except TypeError:
            raise TypeError(f"Object of type {type(obj)} does not have dimensions")


def is_dimensionless(obj: Any) -> bool:
    """
    Test if a value is dimensionless or not.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    dimensionless : `bool`
        ``True`` if `obj` is dimensionless.
    """
    return get_dimensions(obj) is DIMENSIONLESS


def have_same_dimensions(obj1: Any, obj2: Any) -> bool:
    """Test if two values have the same dimensions.

    Parameters
    ----------
    obj1, obj2 : {`Quantity`, array-like, number}
        The values of which to compare the dimensions.

    Returns
    -------
    same : `bool`
        ``True`` if `obj1` and `obj2` have the same dimensions.
    """
    # If dimensions are consistently created using get_or_create_dimensions,
    # the fast "is" comparison should always return the correct result.
    # To be safe, we also do an equals comparison in case it fails. This
    # should only add a small amount of unnecessary computation for cases in
    # which this function returns False which very likely leads to a
    # DimensionMismatchError anyway.
    dim1 = get_dimensions(obj1)
    dim2 = get_dimensions(obj2)
    return (dim1 is dim2) or (dim1 == dim2) or dim1 is None or dim2 is None


def fail_for_dimension_mismatch(
    obj1: Any,
    obj2: Any = None,
    error_message: str | None = None,
    **error_quantities: Any,
) -> tuple[Dimension, Dimension]:
    """
    Compare the dimensions of two objects.

    Parameters
    ----------
    obj1, obj2 : {array-like, `Quantity`}
        The object to compare. If `obj2` is ``None``, assume it to be
        dimensionless
    error_message : str, optional
        An error message that is used in the DimensionMismatchError
    error_quantities : dict mapping str to `Quantity`, optional
        Quantities in this dictionary will be converted using the `_short_str`
        helper method and inserted into the ``error_message`` (which should
        have placeholders with the corresponding names). The reason for doing
        this in a somewhat complicated way instead of directly including all the
        details in ``error_messsage`` is that converting large quantity arrays
        to strings can be rather costly and we don't want to do it if no error
        occured.

    Returns
    -------
    dim1, dim2 : `Dimension`, `Dimension`
        The physical dimensions of the two arguments (so that later code does
        not need to get the dimensions again).

    Raises
    ------
    DimensionMismatchError
        If the dimensions of `obj1` and `obj2` do not match (or, if `obj2` is
        ``None``, in case `obj1` is not dimensionsless).

    Notes
    -----
    Implements special checking for ``0``, treating it as having "any
    dimensions".
    """
    dim1 = get_dimensions(obj1)
    if obj2 is None:
        dim2 = DIMENSIONLESS
    else:
        dim2 = get_dimensions(obj2)

    if dim1 is not dim2 and not (dim1 is None or dim2 is None):
        # Special treatment for "0":
        # if it is not a Quantity, it has "any dimension".
        # This allows expressions like 3*mV + 0 to pass (useful in cases where
        # zero is treated as the neutral element, e.g. in the Python sum
        # builtin) or comparisons like 3 * mV == 0 to return False instead of
        # failing # with a DimensionMismatchError. Note that 3*mV == 0*second
        # is not allowed, though.
        if (dim1 is DIMENSIONLESS and np.all(obj1 == 0)) or (
            dim2 is DIMENSIONLESS and np.all(obj2 == 0)
        ):
            return dim1, dim2

        if error_message is None:
            error_message = "Dimension mismatch"
        else:
            error_quantities = {name: _short_str(q) for name, q in error_quantities.items()}
            error_message = error_message.format(**error_quantities)
        # If we are comparing an object to a specific unit, we don't want to
        # restate this unit (it is probably mentioned in the text already)
        from .unit import Unit

        if obj2 is None or isinstance(obj2, (Dimension, Unit)):
            raise DimensionMismatchError(error_message, dim1)
        else:
            raise DimensionMismatchError(error_message, dim1, dim2)
    else:
        return dim1, dim2
