"""Reading quantities from text, without ``eval``.

``parse_quantity("3.5 * mV")`` evaluates a deliberately small language: numbers,
names of units, ``*``, ``/``, ``**`` with a plain number as the exponent, ``+`` and
``-`` signs, parentheses, and lists of numbers, optionally written as
``array([...])`` so that the ``repr`` of an array quantity can be read back. The
text is turned into a Python syntax tree with ``ast.parse``, and the tree is
walked node by node. A node of any other kind (a function call, an attribute, a
subscript, a comprehension, ...) is refused before anything is evaluated, so
nothing in the text is ever executed.

In this file, in order: the limits and ``GRAMMAR``, ``default_namespace`` (all
unit names), ``parse_quantity`` and ``parse_dimensions``, and the evaluator
(``_evaluate`` and its helpers, including the "Did you mean ...?" suggestions).
"""

from __future__ import annotations

import ast
import difflib
import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .dimension import get_dimensions
from .errors import QuantityParseError

if TYPE_CHECKING:
    from .dimension import Dimension
    from .quantity import Quantity

MAX_LENGTH = 500  #: characters
MAX_NODES = 200  #: syntax-tree nodes
MAX_EXPONENT = 64  #: largest allowed |exponent|; 10 ** 64 is already absurd for a unit

GRAMMAR = (
    "numbers, unit names, *, /, ** with a number as exponent, + and - signs, "
    "parentheses, and lists of numbers"
)


@functools.cache
def default_namespace():
    """All unit names: those of QuantSI.allunits and QuantSI.stdunits."""
    from .. import allunits, stdunits

    names = {name: getattr(allunits, name) for name in allunits.__all__}
    names.update({name: getattr(stdunits, name) for name in stdunits.__all__})
    return names


def parse_quantity(
    text: str, namespace: Mapping[str, Any] | None = None
) -> Quantity | np.ndarray | float:
    """Read a quantity, written as an expression of numbers and units, from text.

    >>> from QuantSI import parse_quantity
    >>> parse_quantity("-70 * mV")
    -70. * mvolt
    >>> parse_quantity("2 * nA * 50 * Mohm")
    100. * mvolt

    ``namespace`` maps the names that may be used to their values; by default, all
    unit names. A dimensionless result is a plain number, as everywhere in QuantSI.
    Raises `QuantityParseError` for text outside the language (see ``GRAMMAR``).
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected a string, got {type(text).__name__}")
    if len(text) > MAX_LENGTH:
        raise QuantityParseError(f"Expression longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise QuantityParseError(f"Not a valid expression ({error.msg}): {text!r}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise QuantityParseError(f"Expression with more than {MAX_NODES} parts")
    if any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in nodes):
        raise QuantityParseError(
            "'...' marks an abbreviated array representation, which cannot be read "
            "back; print the array with np.printoptions(threshold=sys.maxsize)"
        )
    names = default_namespace() if namespace is None else namespace
    return _evaluate(tree.body, names)


def parse_dimensions(text: str, namespace: Mapping[str, Any] | None = None) -> Dimension:
    """The dimensions of a quantity written as text, as the interned `Dimension`.

    >>> from QuantSI import parse_dimensions
    >>> parse_dimensions("50 * mV")
    metre ** 2 * kilogram * second ** -3 * amp ** -1
    """
    return get_dimensions(parse_quantity(text, namespace))


def _suggest(name, names):
    """' Did you mean ...?' with up to three similar names (ignoring case)."""
    by_lower_case = {}
    for known in names:
        by_lower_case.setdefault(known.lower(), known)
    close = difflib.get_close_matches(name.lower(), by_lower_case, n=3)
    if not close:
        return ""
    return " Did you mean " + ", ".join(by_lower_case[c] for c in close) + "?"


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _evaluate(node, names):
    match node:
        case ast.Constant(value=value) if _is_number(value):
            return value
        case ast.Name(id=name):
            try:
                return names[name]
            except KeyError:
                raise QuantityParseError(
                    f"'{name}' is not a known unit.{_suggest(name, names)}"
                ) from None
        case ast.BinOp(left=left, op=ast.Mult(), right=right):
            return _evaluate(left, names) * _evaluate(right, names)
        case ast.BinOp(left=left, op=ast.Div(), right=right):
            return _evaluate(left, names) / _evaluate(right, names)
        case ast.BinOp(left=base, op=ast.Pow(), right=exponent):
            return _power(_evaluate(base, names), _evaluate(exponent, names))
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_evaluate(operand, names)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return +_evaluate(operand, names)
        case ast.List():
            return np.array(_numbers(node, names))
        case ast.Call(func=ast.Name(id="array"), args=[ast.List() as numbers], keywords=[]):
            return np.array(_numbers(numbers, names))  # as written by repr
        case _:
            raise QuantityParseError(
                f"{type(node).__name__} is not allowed in a unit expression (allowed: {GRAMMAR})"
            )


def _numbers(node, names):
    """The (nested) Python list of plain numbers written in a list node."""
    values = []
    for element in node.elts:
        value = (
            _numbers(element, names) if isinstance(element, ast.List) else _evaluate(element, names)
        )
        if not (_is_number(value) or isinstance(value, list)):
            raise QuantityParseError("The elements of a list must be plain numbers")
        values.append(value)
    return values


def _power(base, exponent):
    if not _is_number(exponent):
        raise QuantityParseError(f"The exponent must be a plain number, not {exponent!r}")
    if abs(exponent) > MAX_EXPONENT:
        raise QuantityParseError(f"The exponent {exponent} is larger than {MAX_EXPONENT}")
    if _is_number(base):
        # Floats, not Python's unbounded integers: ((10**64)**64)**64 must not
        # build a number with millions of digits.
        try:
            return float(base) ** exponent
        except OverflowError:
            raise QuantityParseError("The number is too large") from None
    return base**exponent
