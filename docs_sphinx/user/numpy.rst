Using NumPy functions with quantities
=====================================

Quantities are NumPy arrays, so NumPy's functions accept them. QuantSI makes sure
that the result has the right dimensions, or refuses the call when there is no
right answer. Every NumPy function that can see a quantity is in one of these
groups:

Functions that work as they are
    Reductions, statistics, rounding, reshaping, sorting, selecting by index:
    ``np.sum``, ``np.mean``, ``np.std`` (dimensions of the input), ``np.var``
    (squared), ``np.prod`` (to the power of the number of factors), ``np.diff``,
    ``np.sort``, ``np.reshape``, ``np.linspace``, ...

Functions QuantSI implements
    The function checks that its inputs have compatible dimensions and gives the
    result the dimensions it should have: ``np.concatenate`` and the other joining
    functions, ``np.where``, ``np.interp``, ``np.histogram`` (the bin edges have the
    data's dimensions, densities their inverse), products (``np.dot``,
    ``np.einsum``, ...), and linear algebra (``np.linalg.inv`` gives the inverse
    dimensions, ``np.linalg.det`` the dimensions to the power of the matrix size).

    For ``np.concatenate`` and ``np.stack``, an ``out=`` buffer must already have
    the result's dimensions. A buffer with different dimensions, including a
    plain array for a dimension-bearing result, raises ``DimensionMismatchError``
    before any values are written. A valid buffer is filled and returned without
    changing its dimensions.

Functions whose results have no units
    Indices (``np.argsort``, ``np.nonzero``), booleans and counts
    (``np.isin``, after checking the dimensions, and ``np.count_nonzero``), shapes
    and dtypes, text and files (``np.savetxt`` writes values in base SI units), and
    arrays created with ``like=``.

Functions that are refused
    A ``TypeError`` explains why: functions for dates, bits, polynomial
    coefficients and record arrays, and a few routines whose results would need
    several dimensions at once (``np.vander``, ``np.linalg.lstsq``). Apply such a
    function to ``np.asarray(x)`` (the values in base SI units) if dropping the
    units is what you want.

A function that a newer NumPy release added, and that QuantSI therefore does not
know yet, runs NumPy's own implementation with a ``QuantSIWarning``. Make such
warnings errors with ``warnings.simplefilter("error", QuantSIWarning)``.

Where units end
---------------

QuantSI can only check what NumPy hands to it. These calls never reach QuantSI:

* ``np.asarray(q)`` and ``np.array(q)`` return the values in base SI units
  (this is the way to drop units on purpose);
* ``np.arange`` with quantities as start, stop or step (NumPy converts them before
  any check);
* iterating over an array element by element in a C extension.

Changes from earlier versions
-----------------------------

Several functions used to return results with wrong or missing dimensions
without any warning. They now return the right dimensions or raise:

======================================  ==========================  ===========================
Call                                    Before                      Now
======================================  ==========================  ===========================
``np.concatenate([q_m, q_m])``          dimensionless               metres
``np.concatenate([q_m, q_s])``          dimensionless               ``DimensionMismatchError``
``np.where(mask, q_m, q_s)``            plain array                 ``DimensionMismatchError``
``np.dot(q_m, q_s)``                    plain number                metre * second
``np.linalg.inv(matrix_m)``             metres                      1 / metre
``np.argsort(q_m)``                     indices in metres           plain indices
``np.array_equal(1 * m, 1 * s)``        ``True``                    ``False``
``np.copy(q_m)``                        plain array                 metres
``np.polyval(p_m, x)``                  plain number                ``TypeError``
======================================  ==========================  ===========================
