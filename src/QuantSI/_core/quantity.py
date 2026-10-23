"""The Quantity class: a NumPy array that carries a physical dimension.

Values are always stored in base SI units, and ``.dim`` points to the shared
`Dimension` object. NumPy calls ``Quantity.__array_ufunc__`` for arithmetic, and
that's where dimensions are checked and combined.

In this file, in order:

1. ``wrap_function_keep_dimensions`` (used for a few methods, and by
   unitsafefunctions), ``quantity_with_dimensions`` (what pickles call to rebuild
   a Quantity) and ``_new_quantity`` (the fast way to wrap a NumPy result)
2. the Quantity class, with its methods under these headings: creating
   quantities, the NumPy hooks, dimensions, units and text, getting and setting
   items, comparisons, copying and pickling, and NumPy methods that need help
   with units
"""

from __future__ import annotations

import numbers
import operator
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from numpy.exceptions import VisibleDeprecationWarning

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

from .array_functions import (
    HANDLED,
    SUBCLASS_SAFE,
    UNIT_FREE,
    strip_units,
    unsupported_message,
    unsupported_reason,
)
from .dimension import (
    DIMENSIONLESS,
    Dimension,
    fail_for_dimension_mismatch,
    get_dimensions,
    get_or_create_dimension,
    is_scalar_type,
)
from .errors import DimensionMismatchError, QuantSIWarning
from .registry import (
    UnitRegistry,
    additional_unit_register,
    standard_unit_register,
    user_unit_register,
)
from .ufuncs import HANDLERS, check_method
from .utils import _flatten, numpy_docstring, set_module


def wrap_function_keep_dimensions(func):
    """
    Returns a new function that wraps the given function `func` so that it
    keeps the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling `func`, the output is a quantity
    with the original dimensions re-attached.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched, allowing to work functions like
    ``sum`` to work as expected with additional ``axis`` etc. arguments.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        return Quantity(func(np.asarray(x), *args, **kwds), dim=x.dim)

    f._arg_units = [None]
    f._return_unit = lambda u: u
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


@set_module("QuantSI.fundamentalunits")
def quantity_with_dimensions(floatval: object, dims: Dimension) -> Quantity:
    """
    Create a new `Quantity` with the given dimensions. Calls
    `get_or_create_dimensions` with the dimension tuple of the `dims`
    argument to make sure that unpickling (which calls this function) does not
    accidentally create new Dimension objects which should instead refer to
    existing ones.

    Parameters
    ----------
    floatval : `float`
        The floating point value of the quantity.
    dims : `Dimension`
        The physical dimensions of the quantity.

    Returns
    -------
    q : `Quantity`
        A quantity with the given dimensions.

    Examples
    --------
    >>> from QuantSI import *
    >>> quantity_with_dimensions(0.001, volt.dim)
    1. * mvolt

    See Also
    --------
    get_or_create_dimensions
    """
    return Quantity(floatval, get_or_create_dimension(dims._dims))


def _new_quantity(values, dim):
    """``Quantity(values, dim=dim)`` for values NumPy has just computed.

    ``Quantity.__new__`` checks that the data is numeric and works out a
    dimension when none is given. For the result of a NumPy operation on valid
    quantities neither check can fail, so this skips them and gives the same
    result. In particular, dimensionless results still become plain arrays or
    Python numbers.
    """
    values = np.asarray(values)
    if dim is DIMENSIONLESS:
        return values.item() if values.shape == () else values
    result = values.view(Quantity)
    result.dim = dim
    return result


class Quantity(np.ndarray):
    """
    A number with an associated physical dimension. In most cases, it is not
    necessary to create a Quantity object by hand, instead use multiplication
    and division of numbers with the constant unit names ``second``,
    ``kilogram``, etc.

    Notes
    -----
    The `Quantity` class defines arithmetic operations which check for
    consistency of dimensions and raise the DimensionMismatchError exception
    if they are inconsistent. It also defines default and other representations
    for a number for printing purposes.

    See the documentation on the Unit class for more details
    about the available unit names like mvolt, etc.

    *Casting rules*

    The rules that define the casting operations for
    Quantity object are:

    1. Quantity op Quantity = Quantity
       Performs dimension checking if appropriate
    2. (Scalar or Array) op Quantity = Quantity
       Assumes that the scalar or array is dimensionless

    There is one exception to the above rule, the number ``0`` is interpreted
    as having "any dimension".

    Examples
    --------
    >>> from QuantSI import *
    >>> I = 3 * amp  # I is a Quantity object
    >>> R = 2 * ohm  # same for R
    >>> I * R
    6. * volt
    >>> (I * R).in_unit(mvolt)
    '6000. mV'
    >>> (I * R) / mvolt
    6000.0
    >>> X = I + R  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Addition, dimensions were (A) (m^2 kg s^-3 A^-2)
    >>> Is = np.array([1, 2, 3]) * amp
    >>> Is * R
    array([2., 4., 6.]) * volt
    >>> np.asarray(Is * R)  # gets rid of units
    array([2., 4., 6.])

    See also
    --------
    Unit

    Attributes
    ----------
    dimensions
    is_dimensionless
    dim : Dimensions
        The physical dimensions of this quantity.

    Methods
    -------
    with_dimensions
    has_same_dimensions
    in_unit
    in_best_unit
    """

    __slots__ = ["dim"]
    dim: Dimension  #: the physical dimensions (shared, interned)

    __array_priority__ = 1000

    if TYPE_CHECKING:
        # Only type checkers see these; at run time NumPy's own operators run and call
        # __array_ufunc__. NumPy's type stubs say that arithmetic on any array gives a
        # plain ndarray, so without these `3 * mV` would not count as a Quantity.
        # (NumPy declares its own subclass np.matrix the same way.) A result whose
        # dimensions cancel, like (3 * mV) / (1 * mV), is a plain number at run time.
        def __add__(self, other: ArrayLike, /) -> Quantity: ...
        def __radd__(self, other: ArrayLike, /) -> Quantity: ...
        def __sub__(self, other: ArrayLike, /) -> Quantity: ...  # type: ignore[override]
        def __rsub__(self, other: ArrayLike, /) -> Quantity: ...  # type: ignore[override]
        def __mul__(self, other: ArrayLike, /) -> Quantity: ...
        def __rmul__(self, other: ArrayLike, /) -> Quantity: ...
        def __truediv__(self, other: ArrayLike, /) -> Quantity: ...  # type: ignore[override]
        def __rtruediv__(self, other: ArrayLike, /) -> Quantity: ...
        def __mod__(self, other: ArrayLike, /) -> Quantity: ...
        def __pow__(self, other: ArrayLike, /) -> Quantity: ...  # type: ignore[override, unused-ignore]  # NumPy 2.5's stubs need it, 2.2's don't
        def __neg__(self) -> Quantity: ...
        def __pos__(self) -> Quantity: ...
        def __abs__(self) -> Quantity: ...

    #### Creating quantities ####
    def __new__(cls, arr, dim=None, dtype=None, copy=False, force_quantity=False):
        # Do not create dimensionless quantities, use pure numpy arrays instead
        if dim is DIMENSIONLESS and not force_quantity:
            if copy:
                arr = np.array(arr, dtype=dtype)
            else:
                arr = np.asarray(arr, dtype=dtype)
            if arr.shape == ():
                # For scalar values, return a simple Python object instead of
                # a numpy scalar
                return arr.item()
            return arr

        # All np.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        if copy:
            subarr = np.array(arr, dtype=dtype).view(cls)
        else:
            subarr = np.asarray(arr, dtype=dtype).view(cls)
        # We only want numerical datatypes
        if not issubclass(np.dtype(subarr.dtype).type, (np.number, np.bool_)):
            raise TypeError("Quantities can only be created from numerical data.")

        # If a dimension is given, force this dimension
        if dim is not None:
            subarr.dim = dim
            return subarr

        # Use the given dimension or the dimension of the given array (if any)
        try:
            subarr.dim = arr.dim
        except AttributeError:
            if not isinstance(arr, (np.ndarray, np.number, numbers.Number)):
                # check whether it is an iterable containing Quantity objects
                try:
                    is_quantity = [isinstance(x, Quantity) for x in _flatten(arr)]
                except TypeError:
                    # Not iterable
                    is_quantity = [False]
                if len(is_quantity) == 0:
                    # Empty list
                    subarr.dim = DIMENSIONLESS
                elif all(is_quantity):
                    dims = [x.dim for x in _flatten(arr)]
                    one_dim = dims[0]
                    for d in dims:
                        if d != one_dim:
                            raise DimensionMismatchError(
                                "Mixing quantities with different dimensions is not allowed",
                                d,
                                one_dim,
                            )
                    subarr.dim = dims[0]
                elif any(is_quantity):
                    raise TypeError("Mixing quantities and non-quantities is not allowed.")

        return subarr

    def __array_finalize__(self, orig):
        self.dim = getattr(orig, "dim", DIMENSIONLESS)

    @staticmethod
    def with_dimensions(value, *args, **keywords):
        """
        Create a `Quantity` object with dim.

        Parameters
        ----------
        value : {array_like, number}
            The value of the dimension
        args : {`Dimension`, sequence of float}
            Either a single argument (a `Dimension`) or a sequence of 7 values.
        kwds
            Keywords defining the dim, see `Dimension` for details.

        Returns
        -------
        q : `Quantity`
            A `Quantity` object with the given dim

        Examples
        --------
        All of these define an equivalent `Quantity` object:

        >>> from QuantSI import *
        >>> Quantity.with_dimensions(2, get_or_create_dimension(length=1))
        2. * metre
        >>> Quantity.with_dimensions(2, length=1)
        2. * metre
        >>> 2 * metre
        2. * metre
        """
        if len(args) and isinstance(args[0], Dimension):
            dimensions = args[0]
        else:
            dimensions = get_or_create_dimension(*args, **keywords)
        return Quantity(value, dim=dimensions)

    #### NumPy hooks: ufuncs and other NumPy functions ####
    def __array_ufunc__(self, uf, method, *inputs, **kwargs):
        handler = HANDLERS.get(uf)
        if handler is None:
            return NotImplemented
        if method not in ("__call__", "reduce"):
            check_method(uf, method, inputs)
            if method == "at":  # modifies inputs[0] in place, returns None
                return uf.at(*map(np.asarray, inputs), **kwargs)
        if "out" in kwargs:
            return self._ufunc_with_output(handler, uf, method, inputs, kwargs)
        result, dim = handler(self, uf, method, inputs, kwargs)
        return result if dim is None else _new_quantity(result, dim)

    def _ufunc_with_output(self, handler, uf, method, inputs, kwargs):
        """Run a ufunc with ``out=``, keeping the output's dimensions correct.

        * A scalar (0-d) Quantity output behaves like a Python float: it is not
          written to, so ``x += y`` rebinds ``x`` to a new object.
        * A Quantity array output receives the values and is relabelled with the
          dimensions of the result (as in astropy), so ``q *= 2 * second`` works
          and the buffer never carries a wrong label. The output object itself is
          returned, as NumPy specifies.
        * A plain ndarray output receives the values in base SI units.
        * Units cannot be used as outputs: they are shared constants.
        """
        from .unit import Unit

        (out,) = kwargs["out"]
        if isinstance(out, Unit):
            raise TypeError("Units cannot be modified in-place")
        if isinstance(out, Quantity) and out.ndim == 0:
            del kwargs["out"]
            result, dim = handler(self, uf, method, inputs, kwargs)
            return result if dim is None else _new_quantity(result, dim)
        kwargs["out"] = (np.asarray(out),)  # the same buffer, without the Quantity
        result, dim = handler(self, uf, method, inputs, kwargs)
        if not isinstance(out, Quantity):
            return result if dim is None else _new_quantity(result, dim)
        out.dim = DIMENSIONLESS if dim is None else dim
        return out

    def __array_function__(self, func, types, args, kwargs):
        # NEP 18: called for NumPy functions such as np.concatenate; see array_functions.py.
        if func in SUBCLASS_SAFE:
            return super().__array_function__(func, types, args, kwargs)
        if not all(issubclass(t, np.ndarray) for t in types):
            return NotImplemented  # let another array type (dask, ...) handle it
        handler = HANDLED.get(func)
        if handler is not None:
            return handler(func, _new_quantity, *args, **kwargs)
        if func in UNIT_FREE:
            return func(*strip_units(args), **strip_units(kwargs))
        reason = unsupported_reason(func)
        if reason is not None:
            raise TypeError(unsupported_message(func, reason))
        # A function from a NumPy release newer than this QuantSI: run NumPy's own
        # implementation, so that upgrading NumPy does not break code, but say so.
        warn(
            f"numpy.{func.__name__} has not been reviewed for use with quantities; "
            "running NumPy's own implementation, whose result may have wrong "
            "dimensions.",
            QuantSIWarning,
            stacklevel=2,
        )
        return super().__array_function__(func, types, args, kwargs)

    #### Dimensions ####
    is_dimensionless = property(
        lambda self: self.dim.is_dimensionless,
        doc="Whether this is a dimensionless quantity.",
    )

    @property
    def dimensions(self):
        """
        The physical dimensions of this quantity.
        """
        return self.dim

    @dimensions.setter
    def dimensions(self, dim):
        self.dim = dim

    def has_same_dimensions(self, other: object) -> bool:
        """
        Return whether this object has the same dimensions as another.

        Parameters
        ----------
        other : {`Quantity`, array-like, number}
            The object to compare the dimensions against.

        Returns
        -------
        same : `bool`
            ``True`` if `other` has the same dimensions.
        """
        other_dim = get_dimensions(other)
        return (self.dim is other_dim) or (self.dim == other_dim)

    #### Units and text ####
    def in_unit(self, u: Quantity, precision: int | None = None, python_code: bool = False) -> str:
        """
        Represent the quantity in a given unit. If `python_code` is ``True``,
        this will return valid python code, i.e. a string like ``5.0 * um ** 2``
        instead of ``5.0 um^2``

        Parameters
        ----------
        u : {`Quantity`, `Unit`}
            The unit in which to show the quantity.
        precision : `int`, optional
            The number of digits of precision (in the given unit, see Examples).
            If no value is given, numpy's `get_printoptions` value is used.
        python_code : `bool`, optional
            Whether to return valid python code (``True``) or a human readable
            string (``False``, the default).

        Returns
        -------
        s : `str`
            String representation of the object in unit `u`.

        Examples
        --------
        >>> from QuantSI.allunits import *
        >>> from QuantSI.stdunits import *
        >>> x = 25.123456 * mV
        >>> x.in_unit(volt)
        '0.02512346 V'
        >>> x.in_unit(volt, 3)
        '0.025 V'
        >>> x.in_unit(mV, 3)
        '25.123 mV'

        See Also
        --------
        in_unit
        """

        from .formatting import format_quantity

        fail_for_dimension_mismatch(self, u, 'Non-matching unit for method "in_unit"')
        return format_quantity(self, u, precision=precision, python_code=python_code)

    def get_best_unit(self, *regs: UnitRegistry) -> Quantity:
        """
        Return the best unit for this `Quantity`.

        Parameters
        ----------
        regs : any number of `UnitRegistry` objects
            The registries that are searched for units. If none are provided, it
            will check the standard, user and additional unit registers in turn.

        Returns
        -------
            u : `Quantity` or `Unit`
                The best-fitting unit for the quantity `x`.
        """
        from .unit import Unit

        if self.is_dimensionless:
            return Unit(1)
        if len(regs):
            for r in regs:
                try:
                    return r[self]
                except KeyError:
                    pass
            return Quantity(1, self.dim)
        else:
            return self.get_best_unit(
                standard_unit_register, user_unit_register, additional_unit_register
            )

    def _get_best_unit(self, *regs):
        warn(
            "Quantity._get_best_unit has been renamed to Quantity.get_best_unit.",
            VisibleDeprecationWarning,
        )
        return self.get_best_unit(*regs)

    def in_best_unit(
        self, precision: int | None = None, python_code: bool = False, *regs: UnitRegistry
    ) -> str:
        """
        Represent the quantity in the "best" unit.

        Parameters
        ----------
        python_code : `bool`, optional
            If set to ``False`` (the default), will return a string like
            ``5.0 um^2`` which is not a valid Python expression. If set to
            ``True``, it will return ``5.0 * um ** 2`` instead.
        precision : `int`, optional
            The number of digits of precision (in the best unit, see
            Examples). If no value is given, numpy's
            `get_printoptions` value is used.
        regs : `UnitRegistry` objects
            The registries where to search for units. If none are given, the
            standard, user-defined and additional registries are searched in
            that order.

        Returns
        -------
        representation : `str`
            A string representation of this `Quantity`.

        Examples
        --------
        >>> from QuantSI.allunits import *

        >>> x = 0.00123456 * volt

        >>> x.in_best_unit()
        '1.23456 mV'

        >>> x.in_best_unit(3)
        '1.235 mV'

        See Also
        --------
        in_best_unit
        """
        from .formatting import format_quantity

        u = self.get_best_unit(*regs)
        return format_quantity(self, u, precision=precision, python_code=python_code)

    def __repr__(self):
        return self.in_best_unit(python_code=True)

    def __str__(self):
        return self.in_best_unit()

    def __format__(self, format_spec):
        """Keep units for an empty spec; otherwise format the raw SI value as NumPy does."""
        if format_spec == "":
            return str(self)
        return super().__format__(format_spec)

    def _latex(self, *args):
        """LaTeX for this quantity; SymPy's ``latex()`` calls this too (passing its
        printer, which is not needed). See `format_quantity_latex`."""
        from .formatting import format_quantity_latex

        return format_quantity_latex(self)

    def _repr_latex_(self):
        return f"${self._latex(None)}$"

    #### Getting and setting items ####
    def __getitem__(self, key):
        """Overwritten to assure that single elements (i.e., indexed with a
        single integer or a tuple of integers) retain their unit.
        """
        return _new_quantity(np.ndarray.__getitem__(self, key), self.dim)

    def item(self, *args):
        """Overwritten to assure that the returned element retains its unit."""
        return Quantity(np.ndarray.item(self, *args), self.dim)

    def __setitem__(self, key, value):
        fail_for_dimension_mismatch(self, value, "Inconsistent units in assignment")
        return super().__setitem__(key, value)

    def tolist(self):
        """
        Convert the array into a list.

        Returns
        -------
        l : list of `Quantity`
            A (possibly nested) list equivalent to the original array.
        """

        def replace_with_quantity(seq, dim):
            """
            Replace all the elements in the list with an equivalent `Quantity`
            with the given `dim`.
            """
            # No recursion needed for single values
            if not isinstance(seq, list):
                return Quantity(seq, dim)

            def top_replace(s):
                """
                Recursivley descend into the list.
                """
                for i in s:
                    if not isinstance(i, list):
                        yield Quantity(i, dim)
                    else:
                        yield type(i)(top_replace(i))

            return type(seq)(top_replace(seq))

        return replace_with_quantity(np.asarray(self).tolist(), self.dim)

    #### Comparisons ####
    def _comparison(self, other, operator_str, operation):
        is_scalar = is_scalar_type(other)
        if not is_scalar and not isinstance(other, np.ndarray):
            return NotImplemented
        if not is_scalar or not np.isinf(other):
            message = (
                "Cannot perform comparison {value1} %s {value2}, units do not match" % operator_str
            )
            fail_for_dimension_mismatch(self, other, message, value1=self, value2=other)
        return operation(np.asarray(self), np.asarray(other))

    def __lt__(self, other):
        return self._comparison(other, "<", operator.lt)

    def __le__(self, other):
        return self._comparison(other, "<=", operator.le)

    def __gt__(self, other):
        return self._comparison(other, ">", operator.gt)

    def __ge__(self, other):
        return self._comparison(other, ">=", operator.ge)

    def __eq__(self, other):
        return self._comparison(other, "==", operator.eq)

    def __ne__(self, other):
        return self._comparison(other, "!=", operator.ne)

    #### Copying and pickling ####
    def __reduce__(self):
        return quantity_with_dimensions, (np.asarray(self), self.dim)

    def __deepcopy__(self, memo):
        return Quantity(self, copy=True)

    #### NumPy methods that need help with units ####
    cumsum = wrap_function_keep_dimensions(np.ndarray.cumsum)

    trace = wrap_function_keep_dimensions(np.trace)

    # (Without these, ndarray's methods would return indices labelled with the
    # quantity's dimensions. NumPy functions such as np.argsort are handled in
    # array_functions.py.)
    def argmax(self, *args, **kwds):
        """Like `numpy.ndarray.argmax`; returns plain indices."""
        return np.asarray(self).argmax(*args, **kwds)

    def argmin(self, *args, **kwds):
        """Like `numpy.ndarray.argmin`; returns plain indices."""
        return np.asarray(self).argmin(*args, **kwds)

    def argsort(self, *args, **kwds):
        """Like `numpy.ndarray.argsort`; returns plain indices."""
        return np.asarray(self).argsort(*args, **kwds)

    def argpartition(self, *args, **kwds):
        """Like `numpy.ndarray.argpartition`; returns plain indices."""
        return np.asarray(self).argpartition(*args, **kwds)

    @numpy_docstring(np.ndarray.fill)
    def fill(self, values):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, values, "fill")
        super().fill(values)

    @numpy_docstring(np.ndarray.put)
    def put(self, indices, values, *args, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, values, "fill")
        super().put(indices, values, *args, **kwds)

    @numpy_docstring(np.ndarray.clip)
    def clip(self, a_min, a_max, *args, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, a_min, "clip")
        fail_for_dimension_mismatch(self, a_max, "clip")
        return Quantity(
            np.clip(
                np.asarray(self),
                np.asarray(a_min),
                np.asarray(a_max),
                *args,
                **kwds,
            ),
            self.dim,
        )

    @numpy_docstring(np.ndarray.dot)
    def dot(self, other, **kwds):  # pylint: disable=C0111
        return Quantity(
            np.array(self).dot(np.array(other), **kwds),
            self.dim * get_dimensions(other),
        )

    @numpy_docstring(np.ndarray.searchsorted)
    def searchsorted(self, v, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, v, "searchsorted")
        return super().searchsorted(np.asarray(v), **kwds)

    @numpy_docstring(np.ndarray.cumprod)
    def cumprod(self, *args, **kwds):  # pylint: disable=C0111
        if not self.is_dimensionless:
            raise TypeError(
                "cumprod over array elements on quantities with dimensions is not possible."
            )
        return Quantity(np.asarray(self).cumprod(*args, **kwds))
