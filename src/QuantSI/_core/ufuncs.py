"""How each NumPy ufunc treats physical dimensions.

``RULES`` maps every ufunc QuantSI supports to a `Rule`. It is keyed by the ufunc
object itself rather than by its name: aliases such as ``np.divide`` and
``np.true_divide`` are the same object and cannot disagree, and a lookup is a
single dictionary access instead of scanning lists of names.

In this file, in order:

1. ``Rule``, ``RULES`` and ``UNSUPPORTED_REASONS``: what each ufunc does with
   dimensions
2. one handler function per rule, in the order of ``Rule``, and ``HANDLERS``,
   which maps each ufunc to its handler (``Quantity.__array_ufunc__`` looks it up
   there)
3. ``check_method``: when ``accumulate``, ``reduceat`` and ``at`` are allowed
4. the old ``UFUNCS_*`` name lists, kept only because other code imports them
"""

import enum

import numpy as np

from .dimension import DIMENSIONLESS, fail_for_dimension_mismatch, get_dimensions


class Rule(enum.Enum):
    """What a ufunc does with the dimensions of its arguments."""

    PRESERVE = "keeps the dimensions of its argument"
    MATCH = "needs arguments with matching dimensions, keeps them"
    COMPARE = "needs arguments with matching dimensions, returns plain booleans"
    UNITLESS_RESULT = "accepts any dimensions, returns a plain (unitless) result"
    MULTIPLY = "multiplies the dimensions"
    DIVIDE = "divides the dimensions"
    SQRT = "takes the square root of the dimensions"
    CBRT = "takes the cube root of the dimensions"
    SQUARE = "squares the dimensions"
    RECIPROCAL = "inverts the dimensions"
    POWER = "raises the dimensions to a dimensionless, scalar exponent"
    DIMENSIONLESS = "needs a dimensionless argument"
    DIMENSIONLESS_BOTH = "needs two dimensionless arguments"
    FIRST_ARGUMENT = "keeps the first argument's dimensions, uses only the second's sign"
    INTEGER_ONLY = "only works on integers, never on quantities"
    UNSUPPORTED = "is not supported for quantities"


#: The rule for each supported ufunc.
RULES = {
    np.absolute: Rule.PRESERVE,
    np.rint: Rule.PRESERVE,
    np.negative: Rule.PRESERVE,
    np.positive: Rule.PRESERVE,
    np.conjugate: Rule.PRESERVE,
    np.floor: Rule.PRESERVE,
    np.ceil: Rule.PRESERVE,
    np.trunc: Rule.PRESERVE,
    np.add: Rule.MATCH,
    np.subtract: Rule.MATCH,
    np.maximum: Rule.MATCH,
    np.minimum: Rule.MATCH,
    np.remainder: Rule.MATCH,
    np.fmod: Rule.MATCH,
    np.less: Rule.COMPARE,
    np.less_equal: Rule.COMPARE,
    np.greater: Rule.COMPARE,
    np.greater_equal: Rule.COMPARE,
    np.equal: Rule.COMPARE,
    np.not_equal: Rule.COMPARE,
    np.logical_and: Rule.UNITLESS_RESULT,
    np.logical_or: Rule.UNITLESS_RESULT,
    np.logical_xor: Rule.UNITLESS_RESULT,
    np.logical_not: Rule.UNITLESS_RESULT,
    np.isfinite: Rule.UNITLESS_RESULT,
    np.isinf: Rule.UNITLESS_RESULT,
    np.isnan: Rule.UNITLESS_RESULT,
    np.sign: Rule.UNITLESS_RESULT,
    np.multiply: Rule.MULTIPLY,
    np.matmul: Rule.MULTIPLY,
    np.divide: Rule.DIVIDE,
    np.floor_divide: Rule.DIVIDE,
    np.sqrt: Rule.SQRT,
    np.square: Rule.SQUARE,
    np.reciprocal: Rule.RECIPROCAL,
    np.power: Rule.POWER,
    np.sin: Rule.DIMENSIONLESS,
    np.sinh: Rule.DIMENSIONLESS,
    np.arcsin: Rule.DIMENSIONLESS,
    np.arcsinh: Rule.DIMENSIONLESS,
    np.cos: Rule.DIMENSIONLESS,
    np.cosh: Rule.DIMENSIONLESS,
    np.arccos: Rule.DIMENSIONLESS,
    np.arccosh: Rule.DIMENSIONLESS,
    np.tan: Rule.DIMENSIONLESS,
    np.tanh: Rule.DIMENSIONLESS,
    np.arctan: Rule.DIMENSIONLESS,
    np.arctanh: Rule.DIMENSIONLESS,
    np.log: Rule.DIMENSIONLESS,
    np.log2: Rule.DIMENSIONLESS,
    np.log10: Rule.DIMENSIONLESS,
    np.log1p: Rule.DIMENSIONLESS,
    np.exp: Rule.DIMENSIONLESS,
    np.exp2: Rule.DIMENSIONLESS,
    np.expm1: Rule.DIMENSIONLESS,
    np.logaddexp: Rule.DIMENSIONLESS_BOTH,
    np.logaddexp2: Rule.DIMENSIONLESS_BOTH,
    np.arctan2: Rule.DIMENSIONLESS_BOTH,
    np.hypot: Rule.DIMENSIONLESS_BOTH,
    np.bitwise_and: Rule.INTEGER_ONLY,
    np.bitwise_or: Rule.INTEGER_ONLY,
    np.bitwise_xor: Rule.INTEGER_ONLY,
    np.invert: Rule.INTEGER_ONLY,
    np.left_shift: Rule.INTEGER_ONLY,
    np.right_shift: Rule.INTEGER_ONLY,
    # The remaining ufuncs of the numpy namespace (see test_every_numpy_ufunc_has_a_rule)
    np.fabs: Rule.PRESERVE,
    np.spacing: Rule.PRESERVE,
    np.fmax: Rule.MATCH,
    np.fmin: Rule.MATCH,
    np.nextafter: Rule.MATCH,
    np.signbit: Rule.UNITLESS_RESULT,
    np.vecdot: Rule.MULTIPLY,
    np.matvec: Rule.MULTIPLY,
    np.vecmat: Rule.MULTIPLY,
    np.float_power: Rule.POWER,
    np.cbrt: Rule.CBRT,
    np.deg2rad: Rule.DIMENSIONLESS,
    np.degrees: Rule.DIMENSIONLESS,
    np.rad2deg: Rule.DIMENSIONLESS,
    np.radians: Rule.DIMENSIONLESS,
    np.copysign: Rule.FIRST_ARGUMENT,
    np.gcd: Rule.INTEGER_ONLY,
    np.lcm: Rule.INTEGER_ONLY,
    np.bitwise_count: Rule.INTEGER_ONLY,
    np.divmod: Rule.UNSUPPORTED,
    np.frexp: Rule.UNSUPPORTED,
    np.modf: Rule.UNSUPPORTED,
    np.ldexp: Rule.UNSUPPORTED,
    np.heaviside: Rule.UNSUPPORTED,
    np.isnat: Rule.UNSUPPORTED,
}

#: Why the UNSUPPORTED ufuncs are refused, for the error message.
UNSUPPORTED_REASONS = {
    np.divmod: "it has two outputs; use // and % separately",
    np.frexp: "it has two outputs",
    np.modf: "it has two outputs",
    np.ldexp: "use x * 2**n instead",
    np.heaviside: "its second argument's dimensions are ambiguous",
    np.isnat: "it only applies to dates and times",
}


# ------------------------------------------------------------------------------
# One handler per rule. A handler receives the Quantity whose __array_ufunc__
# NumPy called, the ufunc, the method ("__call__" or "reduce"), the inputs and
# the keyword arguments, and returns ``(result, dim)``: NumPy's result computed on
# plain arrays, and the dimensions to attach to it (``None`` for "return the result
# as it is"). Error messages are only built when a check fails.
# ------------------------------------------------------------------------------


def _dim(obj):
    """get_dimensions(obj), without the cost of an exception for plain numbers."""
    dim = getattr(obj, "dim", None)
    return get_dimensions(obj) if dim is None else dim


def _call(ufunc, method, inputs, kwargs):
    return getattr(ufunc, method)(*map(np.asarray, inputs), **kwargs)


#: ufunc methods that combine elements pairwise, like a plain call.
_ELEMENTWISE = {"__call__", "outer"}


def _preserve(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), self.dim


def _match(self, ufunc, method, inputs, kwargs):
    # Only checked for calls: for reductions, all elements share one dimension.
    if method in _ELEMENTWISE and _dim(inputs[0]) is not _dim(inputs[1]):
        fail_for_dimension_mismatch(
            inputs[0],
            inputs[1],
            error_message=(
                "Cannot calculate {val1} %s {val2}, the units do not match" % ufunc.__name__
            ),
            val1=inputs[0],
            val2=inputs[1],
        )
    return _call(ufunc, method, inputs, kwargs), self.dim


def _compare(self, ufunc, method, inputs, kwargs):
    result, _ = _match(self, ufunc, method, inputs, kwargs)
    return result, None


def _unitless_result(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), None


def _multiply(self, ufunc, method, inputs, kwargs):
    if method in _ELEMENTWISE:
        dim = _dim(inputs[0]) * _dim(inputs[1])
    else:  # reduce: a product of n factors of the same dimension has dimension**n
        dim = _dim(inputs[0]) ** _factors_per_result(inputs[0], kwargs)
    return _call(ufunc, method, inputs, kwargs), dim


def _factors_per_result(array, kwargs):
    """For a product reduction: how many elements are multiplied into each result.

    ``initial`` is a plain number and adds no factor. With a ``where`` mask the
    count can differ between results, but one Quantity has one dimension for all
    its elements, so such a reduction is refused.
    """
    ones = np.ones(np.shape(array), dtype=np.intp)
    axis = kwargs.get("axis", 0)  # ufunc.reduce's own default; ndarray.prod passes None
    where = kwargs.get("where", True)
    counts = np.add.reduce(ones, axis=axis, where=where)
    distinct = np.unique(counts)
    if distinct.size > 1:
        raise TypeError(
            "Cannot multiply quantities with a different number of factors per "
            "result (the 'where' mask selects different numbers of elements): the "
            "results would have different dimensions."
        )
    if distinct.size == 0:  # an empty result: no element, any dimension will do
        return 0
    return int(distinct[0])


def _divide(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), _dim(inputs[0]) / _dim(inputs[1])


def _sqrt(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), self.dim**0.5


def _cbrt(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), self.dim ** (1 / 3)


def _square(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), self.dim**2


def _reciprocal(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), _dim(inputs[0]) ** -1


def _power(self, ufunc, method, inputs, kwargs):
    exponent = inputs[1]
    if _dim(exponent) is not DIMENSIONLESS:
        fail_for_dimension_mismatch(
            exponent,
            error_message=(
                "The exponent for a power operation has to be dimensionless but was {value}"
            ),
            value=exponent,
        )
    if np.asarray(exponent).size != 1:
        raise TypeError("Only length-1 arrays can be used as an exponent for quantities.")
    dim = _dim(inputs[0]) ** np.asarray(exponent)
    return _call(ufunc, method, inputs, kwargs), dim


def _dimensionless(self, ufunc, method, inputs, kwargs):
    if _dim(inputs[0]) is not DIMENSIONLESS:
        fail_for_dimension_mismatch(
            inputs[0],
            error_message="%s expects a dimensionless argument but got {value}" % ufunc.__name__,
            value=inputs[0],
        )
    return getattr(ufunc, method)(np.asarray(inputs[0]), *inputs[1:], **kwargs), None


def _dimensionless_both(self, ufunc, method, inputs, kwargs):
    for position, value in (("first", inputs[0]), ("second", inputs[1])):
        if _dim(value) is not DIMENSIONLESS:
            fail_for_dimension_mismatch(
                value,
                error_message=(
                    f'Both arguments for "{ufunc.__name__}" should be dimensionless but '
                    f"{position} argument was {{value}}"
                ),
                value=value,
            )
    return _call(ufunc, method, inputs, kwargs), None


def _first_argument(self, ufunc, method, inputs, kwargs):
    return _call(ufunc, method, inputs, kwargs), _dim(inputs[0])


def _integer_only(self, ufunc, method, inputs, kwargs):
    raise TypeError(f"{ufunc.__name__} cannot be used on quantities.")


def _unsupported(self, ufunc, method, inputs, kwargs):
    raise TypeError(
        f"numpy.{ufunc.__name__} is not supported for quantities: "
        f"{UNSUPPORTED_REASONS[ufunc]}. Apply it to np.asarray(x) (the values in base "
        "SI units) if dropping the units is intended."
    )


_HANDLER_FOR_RULE = {
    Rule.PRESERVE: _preserve,
    Rule.MATCH: _match,
    Rule.COMPARE: _compare,
    Rule.UNITLESS_RESULT: _unitless_result,
    Rule.MULTIPLY: _multiply,
    Rule.DIVIDE: _divide,
    Rule.SQRT: _sqrt,
    Rule.CBRT: _cbrt,
    Rule.SQUARE: _square,
    Rule.RECIPROCAL: _reciprocal,
    Rule.POWER: _power,
    Rule.DIMENSIONLESS: _dimensionless,
    Rule.DIMENSIONLESS_BOTH: _dimensionless_both,
    Rule.FIRST_ARGUMENT: _first_argument,
    Rule.INTEGER_ONLY: _integer_only,
    Rule.UNSUPPORTED: _unsupported,
}

#: ufunc -> handler, built once from RULES.
HANDLERS = {ufunc: _HANDLER_FOR_RULE[rule] for ufunc, rule in RULES.items()}


# ------------------------------------------------------------------------------
# ufunc methods other than __call__, outer and reduce
# ------------------------------------------------------------------------------

#: Rules for which every partial result of ``accumulate``/``reduceat`` has the
#: input's dimensions (or none), so that one Quantity can hold them all.
_UNIFORM_PARTIAL_RESULTS = {Rule.MATCH, Rule.UNITLESS_RESULT}


def check_method(ufunc, method, inputs):
    """Refuse ufunc methods whose results a single Quantity cannot represent."""
    rule = RULES[ufunc]
    if method in ("accumulate", "reduceat"):
        if rule not in _UNIFORM_PARTIAL_RESULTS and _dim(inputs[0]) is not DIMENSIONLESS:
            raise TypeError(
                f"{ufunc.__name__}.{method} is not supported for quantities with "
                "dimensions: the partial results would have different dimensions."
            )
    elif method == "at":
        _check_at(ufunc, rule, inputs)


def _check_at(ufunc, rule, inputs):
    """``ufunc.at(target, indices[, operand])`` changes some elements in place.

    Allowed only if those elements keep the target's dimensions.
    """
    target, _, *operand = inputs
    target_dim = _dim(target)
    operand_dim = _dim(operand[0]) if operand else target_dim
    if rule is Rule.MATCH:
        if operand_dim is not target_dim:
            fail_for_dimension_mismatch(
                target,
                operand[0],
                error_message=(
                    "Cannot calculate {val1} %s {val2}, the units do not match" % ufunc.__name__
                ),
                val1=target,
                val2=operand[0],
            )
    elif rule is Rule.PRESERVE and not operand:
        pass
    elif rule in (Rule.MULTIPLY, Rule.DIVIDE) and operand_dim is DIMENSIONLESS:
        pass
    elif target_dim is not DIMENSIONLESS or operand_dim is not DIMENSIONLESS:
        raise TypeError(
            f"{ufunc.__name__}.at cannot be used here: it would change the dimensions "
            "of only some elements of a quantity."
        )


# ------------------------------------------------------------------------------
# The old name lists. Code outside QuantSI imports them (Brian2's test suite
# does), so they stay exactly as they were. Dispatching doesn't use them: RULES is
# what counts, and test_ufunc_rules.py checks that the two agree. Some names never
# matched on NumPy 2 (true_divide, mod, conj and isreal are aliases or not ufuncs,
# and dot is a function).
# ------------------------------------------------------------------------------

# Note: A list of numpy ufuncs can be found here:
# http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

#: ufuncs that work on all dimensions and preserve the dimensions, e.g. abs
UFUNCS_PRESERVE_DIMENSIONS = [
    "absolute",
    "rint",
    "negative",
    "positive",
    "conj",
    "conjugate",
    "floor",
    "ceil",
    "trunc",
]

#: ufuncs that work on all dimensions but change the dimensions, e.g. square
UFUNCS_CHANGE_DIMENSIONS = [
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "sqrt",
    "square",
    "reciprocal",
    "dot",
    "matmul",
]

#: ufuncs that work with matching dimensions, e.g. add
UFUNCS_MATCHING_DIMENSIONS = [
    "add",
    "subtract",
    "maximum",
    "minimum",
    "remainder",
    "mod",
    "fmod",
]

#: ufuncs that compare values, i.e. work only with matching dimensions but do
#: not result in a value with dimensions, e.g. equals
UFUNCS_COMPARISONS = [
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
]

#: Logical operations that work on all quantities and return boolean arrays
UFUNCS_LOGICAL = [
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "isreal",
    "iscomplex",
    "isfinite",
    "isinf",
    "isnan",
]

#: ufuncs that only work on dimensionless quantities
UFUNCS_DIMENSIONLESS = [
    "sin",
    "sinh",
    "arcsin",
    "arcsinh",
    "cos",
    "cosh",
    "arccos",
    "arccosh",
    "tan",
    "tanh",
    "arctan",
    "arctanh",
    "log",
    "log2",
    "log10",
    "log1p",
    "exp",
    "exp2",
    "expm1",
]

#: ufuncs that only work on two dimensionless quantities
UFUNCS_DIMENSIONLESS_TWOARGS = ["logaddexp", "logaddexp2", "arctan2", "hypot"]

#: ufuncs that only work on integers and therefore never on quantities
UFUNCS_INTEGERS = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
]
