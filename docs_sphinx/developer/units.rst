Units
=====  

Casting rules
-------------
QuantSI treats both scalars and arrays as dimensionless for unit checking
and make all operations involving quantities return a quantity.::

    >>> 1 + 1*second   # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Cannot calculate 1. s + 1, units do not match (units are second and 1).

    >>> np.array([1]) + 1*second   # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Cannot calculate 1. s + [1], units do not match (units are second and 1).

    >>> 1*second + 1*second
    2. * second
    >>> np.array([1])*second + 1*second
    array([ 2.]) * second

As one exception from this rule, a scalar or array ``0`` is considered as having
"any unit", i.e. ``0 + 1 * second`` will result in ``1 * second`` without a
dimension mismatch error and ``0 == 0 * mV`` will evaluate to ``True``. This
seems reasonable from a mathematical viewpoint and makes some sources of error
disappear. For example, the Python builtin ``sum`` (not numpy's version) adds
the value of the optional argument ``start``, which defaults to 0, to its
main argument. Without this exception, ``sum([1 * mV, 2 * mV])`` would therefore
raise an error.

The above rules also apply to all comparisons (e.g. ``==`` or ``<``) with one
further exception: ``inf`` and ``-inf`` also have "any unit", therefore an
expression like ``v <= inf`` will never raise an exception (and always return
``True``).

Functions and units
-------------------

ndarray methods
~~~~~~~~~~~~~~~
As far as the project has been able to test, all ndarray methods work well
with quantities by default, i.e. they check for the
correct units of their arguments and return quantities with units were
appropriate. The only two exceptions are ``cumsum`` and ``trace``, which are
overwritten using a thin function wrapper:

``wrap_function_keep_dimension``:
	Strips away the units before giving the array to the method of ``ndarray``,
	then reattaches the unit to the result

Numpy ufuncs
~~~~~~~~~~~~

All of the standard `numpy ufuncs`_ (functions that operate element-wise on numpy
arrays) are supported, meaning that they check for correct units and return
appropriate arrays. These functions are often called implicitly, for example
when using operators like ``<`` or ``**``.

*Math operations:*
	``add``, ``subtract``, ``multiply``, ``divide``, ``logaddexp``, ``logaddexp2``, 
        ``true_divide``, ``floor_divide``, ``negative``, ``power``, ``remainder``, ``mod``, 
        ``fmod``, ``absolute``, ``rint``, ``sign``, ``conj``, ``conjugate``, ``exp``, ``exp2``, 
        ``log``, ``log2``, ``log10``, ``expm1``, ``log1p``, ``sqrt``, ``square``, ``reciprocal``, 
        ``ones_like``
	
*Trigonometric functions:*
	``sin``, ``cos``, ``tan``, ``arcsin``, ``arccos``, ``arctan``, ``arctan2``, ``hypot``, 
        ``sinh``, ``cosh``, ``tanh``, ``arcsinh``, ``arccosh``, ``arctanh``, ``deg2rad``, ``rad2deg``

*Bitwise functions:*
	``bitwise_and``, ``bitwise_or``, ``bitwise_xor``, ``invert``, ``left_shift``, ``right_shift``

*Comparison functions:* 
	``greater``, ``greater_equal``, ``less``, ``less_equal``, ``not_equal``,
	``equal``, ``logical_and``, ``logical_or``, ``logical_xor``, ``logical_not``, ``maximum``, 
        ``minimum``
	
*Floating functions:*
	``isreal``, ``iscomplex``, ``isfinite``, ``isinf``, ``isnan``, ``floor``, ``ceil``, ``trunc``, 
        ``fmod``

Not taken care of yet: ``signbit``, ``copysign``, ``nextafter``, ``modf``, ``ldexp``, ``frexp``

**Notes**

* Everything involving ``log`` or ``exp``, as well as trigonometric functions
  only works on dimensionless array (for ``arctan2`` and ``hypot`` this is
  questionable, though)
* Unit arrays can only be raised to a scalar power, not to an array of
  exponents as this would lead to differing dimensions across entries. For
  simplicity, this is enforced even for dimensionless quantities.
* Bitwise functions never works on quantities (numpy will by itself throw a 
  ``TypeError`` because they are floats not integers).
* All comparisons only work for matching dimensions (with the exception of
  always allowing comparisons to 0) and return a pure boolean array.
* All logical functions treat quantities as boolean values in the same
  way as floats are treated as boolean: Any non-zero value is True.

.. _numpy ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html

Numpy functions
~~~~~~~~~~~~~~~
Many numpy functions are functional versions of ndarray methods (e.g. ``mean``,
``sum``, ``clip``). They therefore work automatically when called on quantities,
as numpy propagates the call to the respective method.

The only function in numpy that do not propagate its call to the
corresponding method (because it uses np.asarray instead of np.asanyarray,
which might actually be a bug in numpy) is ``trace``.
For this, a wrapped function in ``fundamentalunits.py`` is provided.
         
**numpy functions that work unchanged**

This includes all functional counterparts of the methods mentioned above (with
the exceptions mentioned above). Some other functions also work correctly, as
they are only using functions/methods that work with quantities:

* ``linspace``, ``diff``, ``digitize`` [1]_
* ``trim_zeros``, ``fliplr``, ``flipud``, ``roll``, ``rot90``, ``shuffle``
* ``corrcoeff`` [1]_

.. [1] But does not care about the units of its input.

**numpy functions that return a pure numpy array instead of quantities**

* ``arange``
* ``cov``
* ``random.permutation``
* ``histogram``, ``histogram2d``
* ``cross``, ``inner``, ``outer``
* ``where``

**numpy functions that do something wrong**

* ``insert``, ``delete`` (return a quantity array but without units)
* ``correlate`` (returns a quantity with wrong units)
* ``histogramdd`` (raises a ``DimensionMismatchError``)

**other unsupported functions**
Functions in ``numpy``'s subpackages such as ``linalg`` are not supported and will
either not work with units, or remove units from their inputs.

