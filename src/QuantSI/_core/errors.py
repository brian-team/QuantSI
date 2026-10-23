"""The exceptions and the warning that QuantSI raises.

In this file: ``DimensionMismatchError`` (dimensions that don't fit),
``QuantSIWarning`` (a NumPy function QuantSI hasn't reviewed yet) and
``QuantityParseError`` (text that ``parse_quantity`` can't read).
"""


class DimensionMismatchError(Exception):
    """
    Exception class for attempted operations with inconsistent dimensions.

    For example, ``3*mvolt + 2*amp`` raises this exception. The purpose of this
    class is to help catch errors based on incorrect units. The exception will
    print a representation of the dimensions of the two inconsistent objects
    that were operated on.

    Parameters
    ----------
    description : ``str``
        A description of the type of operation being performed, e.g. Addition,
        Multiplication, etc.
    dims : `Dimension`
        The physical dimensions of the objects involved in the operation, any
        number of them is possible
    """

    def __init__(self, description, *dims):
        # Call the base class constructor to make Exception pickable, see:
        # http://bugs.python.org/issue1692335
        Exception.__init__(self, description, *dims)
        self.dims = dims
        self.desc = description

    def __repr__(self):
        dims_repr = [repr(dim) for dim in self.dims]
        return f"{self.__class__.__name__}({self.desc!r}, {', '.join(dims_repr)})"

    def __str__(self):
        from .registry import get_unit_for_display

        s = self.desc
        if len(self.dims) == 0:
            pass
        elif len(self.dims) == 1:
            s += f" (unit is {get_unit_for_display(self.dims[0])}"
        elif len(self.dims) == 2:
            d1, d2 = self.dims
            s += f" (units are {get_unit_for_display(d1)} and {get_unit_for_display(d2)}"
        else:
            s += f" (units are {' '.join([f'({get_unit_for_display(d)})' for d in self.dims])}"
        if len(self.dims):
            s += ")."
        return s


class QuantSIWarning(UserWarning):
    """Warning about a possibly wrong result, e.g. from a NumPy function QuantSI
    has not reviewed yet. Turn it into an error with
    ``warnings.simplefilter("error", QuantSIWarning)``."""


class QuantityParseError(ValueError):
    """Text that `parse_quantity` cannot read as a quantity."""
