"""NumPy functions on quantities: each function in a bucket has a case here.

Each case calls a NumPy function twice, on quantities and on the same values as
plain arrays, and checks the outcome against an expectation:

``"m"``, ``"m2"``, ``"m s"``, ...  a Quantity with these dimensions whose values
                                   (in base SI units) equal the plain call's result
``"plain"``                        no Quantity anywhere in the result
``"none"``                         the function returns None (it modifies in place)
``"TypeError"``, ...               the call raises this exception
a tuple of the above               the function returns a tuple
"""

import itertools
import pathlib
import types
import warnings

import numpy as np
import pytest

from QuantSI import DimensionMismatchError
from QuantSI._core import array_functions as _array_functions
from QuantSI.allunits import metre, second
from QuantSI.fundamentalunits import DIMENSIONLESS, Quantity

# Dimension arithmetic, not unit arithmetic: combining units (metre**2) would
# register display units while the tests are being collected.
M, S = metre.dim, second.dim
DIMS = {
    "m": M,
    "m2": M**2,
    "m3": M**3,
    "m4": M**4,
    "m s": M * S,
    "s": S,
    "1/m": M**-1,
    "s/m": S / M,
    "m/s": M / S,
    "m^0.5": M**0.5,
    "1/m2": M**-2,
    "1/(m s)": (M * S) ** -1,
    "1": DIMENSIONLESS,
}
EXCEPTIONS = {"TypeError": TypeError, "DimensionMismatchError": DimensionMismatchError}


def without_deprecations(function, *args):
    """Call a NumPy function that newer NumPy versions deprecate (e.g. np.fix in 2.5)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return function(*args)


def written(function, array, *args):
    """Call a function that writes into ``array`` in place; return the array."""
    assert function(array, *args) is None
    return array


def namespace(length, time, tmp_path):
    ns = types.SimpleNamespace(unit=length, second=time, tmp=tmp_path)
    ns.m = np.array([1.0, 2.0, 3.0, 4.0]) * length
    ns.m2 = np.arange(1.0, 7.0).reshape(2, 3) * length
    ns.s = np.array([1.0, 2.0, 4.0, 8.0]) * time
    ns.M = np.array([[2.0, 1.0], [1.0, 3.0]]) * length
    ns.d = np.array([0.1, 0.2, 0.3, 0.4])
    ns.i = np.array([0, 1, 1, 3])
    ns.b = np.array([True, False, True, False])
    return ns


#: function name (as reached from numpy) -> [(label, call, expectation), ...]
CASES = {
    # ---- SUBCLASS_SAFE ----------------------------------------------------------
    "all": [("1d", lambda ns: np.all(ns.m > 0 * ns.unit), "plain")],
    "any": [("1d", lambda ns: np.any(ns.m > 0 * ns.unit), "plain")],
    "amax": [("1d", lambda ns: np.amax(ns.m), "m")],
    "amin": [("1d", lambda ns: np.amin(ns.m), "m")],
    "max": [("1d", lambda ns: np.max(ns.m), "m")],
    "min": [("1d", lambda ns: np.min(ns.m), "m")],
    "sum": [("1d", lambda ns: np.sum(ns.m), "m"), ("axis", lambda ns: np.sum(ns.m2, axis=0), "m")],
    "prod": [("1d", lambda ns: np.prod(ns.m), "m4")],
    "mean": [("1d", lambda ns: np.mean(ns.m), "m")],
    "std": [("1d", lambda ns: np.std(ns.m), "m")],
    "var": [("1d", lambda ns: np.var(ns.m), "m2")],
    "median": [("1d", lambda ns: np.median(ns.m), "m")],
    "percentile": [("1d", lambda ns: np.percentile(ns.m, 50), "m")],
    "quantile": [("1d", lambda ns: np.quantile(ns.m, 0.5), "m")],
    "average": [
        ("1d", lambda ns: np.average(ns.m), "m"),
        ("weights", lambda ns: np.average(ns.m, weights=ns.d), "m"),
    ],
    "ptp": [("1d", lambda ns: np.ptp(ns.m), "m")],
    "nanmax": [("1d", lambda ns: np.nanmax(ns.m), "m")],
    "nanmin": [("1d", lambda ns: np.nanmin(ns.m), "m")],
    "nansum": [("1d", lambda ns: np.nansum(ns.m), "m")],
    "nanprod": [("1d", lambda ns: np.nanprod(ns.m), "m4")],
    "nanmean": [("1d", lambda ns: np.nanmean(ns.m), "m")],
    "nanstd": [("1d", lambda ns: np.nanstd(ns.m), "m")],
    "nanvar": [("1d", lambda ns: np.nanvar(ns.m), "m2")],
    "nanmedian": [("1d", lambda ns: np.nanmedian(ns.m), "m")],
    "nanpercentile": [("1d", lambda ns: np.nanpercentile(ns.m, 50), "m")],
    "nanquantile": [("1d", lambda ns: np.nanquantile(ns.m, 0.5), "m")],
    "trace": [("2d", lambda ns: np.trace(ns.M), "m")],
    "linalg.trace": [("2d", lambda ns: np.linalg.trace(ns.M), "m")],
    "cumsum": [("1d", lambda ns: np.cumsum(ns.m), "m")],
    "cumulative_sum": [("1d", lambda ns: np.cumulative_sum(ns.m), "m")],
    "nancumsum": [("1d", lambda ns: np.nancumsum(ns.m), "m")],
    "cumulative_prod": [
        ("with dimensions", lambda ns: np.cumulative_prod(ns.m * ns.m), "TypeError"),
    ],
    "diff": [("1d", lambda ns: np.diff(ns.m), "m")],
    "ediff1d": [("1d", lambda ns: np.ediff1d(ns.m), "m")],
    "trapezoid": [("1d", lambda ns: np.trapezoid(ns.m, ns.s), "m s")],
    "around": [("1d", lambda ns: np.around(ns.m, 1), "m")],
    "round": [("1d", lambda ns: np.round(ns.m, 1), "m")],
    "fix": [("1d", lambda ns: without_deprecations(np.fix, ns.m), "m")],
    "clip": [
        ("1d", lambda ns: np.clip(ns.m, 1 * ns.unit, 3 * ns.unit), "m"),
        (
            "mismatch",
            lambda ns: np.clip(ns.m, 1 * ns.second, 3 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "nan_to_num": [("1d", lambda ns: np.nan_to_num(ns.m), "m")],
    "real": [("1d", lambda ns: np.real(ns.m), "m")],
    "imag": [("1d", lambda ns: np.imag(ns.m), "m")],
    "real_if_close": [("1d", lambda ns: np.real_if_close(ns.m), "m")],
    "reshape": [("1d", lambda ns: np.reshape(ns.m, (2, 2)), "m")],
    "ravel": [("2d", lambda ns: np.ravel(ns.m2), "m")],
    "transpose": [("2d", lambda ns: np.transpose(ns.m2), "m")],
    "matrix_transpose": [("2d", lambda ns: np.matrix_transpose(ns.m2), "m")],
    "linalg.matrix_transpose": [("2d", lambda ns: np.linalg.matrix_transpose(ns.m2), "m")],
    "swapaxes": [("2d", lambda ns: np.swapaxes(ns.m2, 0, 1), "m")],
    "moveaxis": [("2d", lambda ns: np.moveaxis(ns.m2, 0, 1), "m")],
    "rollaxis": [("2d", lambda ns: np.rollaxis(ns.m2, 1), "m")],
    "squeeze": [("2d", lambda ns: np.squeeze(ns.m2[:1]), "m")],
    "expand_dims": [("1d", lambda ns: np.expand_dims(ns.m, 0), "m")],
    "atleast_1d": [("1d", lambda ns: np.atleast_1d(ns.m), "m")],
    "atleast_2d": [("1d", lambda ns: np.atleast_2d(ns.m), "m")],
    "atleast_3d": [("1d", lambda ns: np.atleast_3d(ns.m), "m")],
    "flip": [("1d", lambda ns: np.flip(ns.m), "m")],
    "fliplr": [("2d", lambda ns: np.fliplr(ns.m2), "m")],
    "flipud": [("2d", lambda ns: np.flipud(ns.m2), "m")],
    "rot90": [("2d", lambda ns: np.rot90(ns.m2), "m")],
    "roll": [("1d", lambda ns: np.roll(ns.m, 1), "m")],
    "repeat": [("1d", lambda ns: np.repeat(ns.m, 2), "m")],
    "tile": [("1d", lambda ns: np.tile(ns.m, 2), "m")],
    "diag": [("2d", lambda ns: np.diag(ns.m2), "m")],
    "diagflat": [("1d", lambda ns: np.diagflat(ns.m), "m")],
    "diagonal": [("2d", lambda ns: np.diagonal(ns.m2), "m")],
    "linalg.diagonal": [("2d", lambda ns: np.linalg.diagonal(ns.M), "m")],
    "take": [("1d", lambda ns: np.take(ns.m, [0, 1]), "m")],
    "take_along_axis": [("1d", lambda ns: np.take_along_axis(ns.m, np.array([0, 1]), 0), "m")],
    "compress": [("1d", lambda ns: np.compress(ns.b, ns.m), "m")],
    "extract": [("1d", lambda ns: np.extract(ns.b, ns.m), "m")],
    "delete": [("1d", lambda ns: np.delete(ns.m, 1), "m")],
    "trim_zeros": [("1d", lambda ns: np.trim_zeros(ns.m), "m")],
    "sort": [("1d", lambda ns: np.sort(ns.m), "m")],
    "partition": [("1d", lambda ns: np.partition(ns.m, 1), "m")],
    "searchsorted": [
        ("1d", lambda ns: np.searchsorted(ns.m, 2 * ns.unit), "plain"),
        ("mismatch", lambda ns: np.searchsorted(ns.m, 2 * ns.second), "DimensionMismatchError"),
    ],
    "put": [
        ("1d", lambda ns: np.put(ns.m.copy(), [0], 9 * ns.unit), "none"),
        ("mismatch", lambda ns: np.put(ns.m.copy(), [0], 9 * ns.second), "DimensionMismatchError"),
    ],
    "split": [("1d", lambda ns: np.split(ns.m, 2), ("m", "m"))],
    "array_split": [("1d", lambda ns: np.array_split(ns.m, 2), ("m", "m"))],
    "hsplit": [("1d", lambda ns: np.hsplit(ns.m, 2), ("m", "m"))],
    "vsplit": [("2d", lambda ns: np.vsplit(ns.m2, 2), ("m", "m"))],
    "dsplit": [("3d", lambda ns: np.dsplit(np.atleast_3d(ns.m2), 1), ("m",))],
    "unstack": [("2d", lambda ns: np.unstack(ns.m2), ("m", "m"))],
    "astype": [("1d", lambda ns: np.astype(ns.m, np.float32), "m")],
    "piecewise": [("1d", lambda ns: np.piecewise(ns.m, [ns.b], [lambda v: v]), "m")],
    "apply_along_axis": [("sum", lambda ns: np.apply_along_axis(np.sum, 0, ns.m2), "m")],
    "meshgrid": [("1d", lambda ns: np.meshgrid(ns.m, ns.s), ("m", "s"))],
    "linspace": [
        ("1d", lambda ns: np.linspace(1 * ns.unit, 4 * ns.unit, 4), "m"),
        (
            "mismatch",
            lambda ns: np.linspace(1 * ns.unit, 4 * ns.second, 4),
            "DimensionMismatchError",
        ),
    ],
    "unique": [("1d", lambda ns: np.unique(ns.m), "m")],
    "unique_values": [("1d", lambda ns: np.unique_values(ns.m), "m")],
    "unique_counts": [("1d", lambda ns: np.unique_counts(ns.m), ("m", "plain"))],
    "unique_inverse": [("1d", lambda ns: np.unique_inverse(ns.m), ("m", "plain"))],
    "empty_like": [("1d", lambda ns: np.empty_like(ns.m).shape, "plain")],
    "zeros_like": [("1d", lambda ns: np.zeros_like(ns.m), "m")],
    "ones_like": [("1d", lambda ns: np.ones_like(ns.m), "m")],
    "full_like": [("1d", lambda ns: np.full_like(ns.m, 1 * ns.unit), "m")],
    "kron": [("1d", lambda ns: np.kron(ns.m, ns.s), "m s")],
    "linalg.matmul": [("2d", lambda ns: np.linalg.matmul(ns.M, ns.M), "m2")],
    "linalg.matrix_power": [("2d", lambda ns: np.linalg.matrix_power(ns.M, 2), "m2")],
    "linalg.vecdot": [("1d", lambda ns: np.linalg.vecdot(ns.m, ns.s), "m s")],
    # ---- HANDLED: joining, stacking, reshaping -----------------------------------------
    "concatenate": [
        ("1d", lambda ns: np.concatenate([ns.m, ns.m]), "m"),
        ("zeros", lambda ns: np.concatenate([ns.m, np.zeros(2)]), "m"),
        ("mismatch", lambda ns: np.concatenate([ns.m, ns.s]), "DimensionMismatchError"),
        ("out", lambda ns: np.concatenate([ns.m, ns.m], out=np.zeros(8) * ns.unit), "m"),
    ],
    "stack": [
        ("1d", lambda ns: np.stack([ns.m, ns.m]), "m"),
        ("mismatch", lambda ns: np.stack([ns.m, ns.s]), "DimensionMismatchError"),
    ],
    "hstack": [
        ("1d", lambda ns: np.hstack([ns.m, ns.m]), "m"),
        ("mismatch", lambda ns: np.hstack([ns.m, ns.s]), "DimensionMismatchError"),
    ],
    "vstack": [("1d", lambda ns: np.vstack([ns.m, ns.m]), "m")],
    "dstack": [("1d", lambda ns: np.dstack([ns.m, ns.m]), "m")],
    "column_stack": [("1d", lambda ns: np.column_stack([ns.m, ns.m]), "m")],
    "block": [
        ("1d", lambda ns: np.block([ns.m, ns.m]), "m"),
        ("nested", lambda ns: np.block([[ns.M, ns.M], [ns.M, ns.M]]), "m"),
        ("mismatch", lambda ns: np.block([ns.m, ns.s]), "DimensionMismatchError"),
    ],
    "append": [
        ("1d", lambda ns: np.append(ns.m, ns.m), "m"),
        ("mismatch", lambda ns: np.append(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "insert": [
        ("1d", lambda ns: np.insert(ns.m, 1, 5 * ns.unit), "m"),
        ("mismatch", lambda ns: np.insert(ns.m, 1, 5 * ns.second), "DimensionMismatchError"),
    ],
    "pad": [
        ("zeros", lambda ns: np.pad(ns.m, 1), "m"),
        ("constant", lambda ns: np.pad(ns.m, 1, constant_values=7 * ns.unit), "m"),
        ("edge", lambda ns: np.pad(ns.m, 1, mode="edge"), "m"),
        (
            "mismatch",
            lambda ns: np.pad(ns.m, 1, constant_values=7 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "broadcast_arrays": [("1d", lambda ns: np.broadcast_arrays(ns.m, ns.s), ("m", "s"))],
    "broadcast_to": [("1d", lambda ns: np.broadcast_to(ns.m, (2, 4)), "m")],
    "copy": [("1d", lambda ns: np.copy(ns.m), "m")],
    "resize": [("1d", lambda ns: np.resize(ns.m, 6), "m")],
    "tril": [("2d", lambda ns: np.tril(ns.M), "m")],
    "triu": [("2d", lambda ns: np.triu(ns.M), "m")],
    "sort_complex": [("1d", lambda ns: np.sort_complex(ns.m), "m")],
    "lib.stride_tricks.sliding_window_view": [
        ("1d", lambda ns: np.lib.stride_tricks.sliding_window_view(ns.m, 2), "m")
    ],
    "unique_all": [("1d", lambda ns: np.unique_all(ns.m), ("m", "plain", "plain", "plain"))],
    # ---- HANDLED: selecting, writing, set operations ------------------------------------
    "where": [
        ("3 arguments", lambda ns: np.where(ns.b, ns.m, 0 * ns.unit), "m"),
        ("1 argument", lambda ns: np.where(ns.m > 2 * ns.unit), ("plain",)),
        ("mismatch", lambda ns: np.where(ns.b, ns.m, ns.s), "DimensionMismatchError"),
    ],
    "select": [
        ("1d", lambda ns: np.select([ns.b], [ns.m], 0 * ns.unit), "m"),
        ("mismatch", lambda ns: np.select([ns.b], [ns.m], 1 * ns.second), "DimensionMismatchError"),
    ],
    "choose": [
        ("1d", lambda ns: np.choose([0, 1, 0, 1], [ns.m, ns.m]), "m"),
        ("mismatch", lambda ns: np.choose([0, 1, 0, 1], [ns.m, ns.s]), "DimensionMismatchError"),
    ],
    "copyto": [
        ("quantity", lambda ns: written(np.copyto, ns.m.copy(), 2 * ns.unit), "m"),
        ("plain number", lambda ns: written(np.copyto, ns.m.copy(), 2.0), "m"),
        (
            "mismatch",
            lambda ns: written(np.copyto, ns.m.copy(), 2 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "place": [
        ("1d", lambda ns: written(np.place, ns.m.copy(), ns.b, 7 * ns.unit), "m"),
        (
            "mismatch",
            lambda ns: written(np.place, ns.m.copy(), ns.b, 7.0),
            "DimensionMismatchError",
        ),
    ],
    "putmask": [
        ("1d", lambda ns: written(np.putmask, ns.m.copy(), ns.b, 7 * ns.unit), "m"),
        (
            "mismatch",
            lambda ns: written(np.putmask, ns.m.copy(), ns.b, 7 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "put_along_axis": [
        (
            "1d",
            lambda ns: written(np.put_along_axis, ns.m.copy(), np.array([0]), 9 * ns.unit, 0),
            "m",
        ),
        (
            "mismatch",
            lambda ns: written(np.put_along_axis, ns.m.copy(), np.array([0]), 9.0, 0),
            "DimensionMismatchError",
        ),
    ],
    "fill_diagonal": [
        ("2d", lambda ns: written(np.fill_diagonal, ns.M.copy(), 0 * ns.unit), "m"),
        (
            "mismatch",
            lambda ns: written(np.fill_diagonal, ns.M.copy(), 1 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "isin": [
        ("1d", lambda ns: np.isin(ns.m, ns.m[:2]), "plain"),
        ("mismatch", lambda ns: np.isin(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "digitize": [
        ("1d", lambda ns: np.digitize(ns.m, ns.m), "plain"),
        ("mismatch", lambda ns: np.digitize(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "union1d": [
        ("1d", lambda ns: np.union1d(ns.m, ns.m), "m"),
        ("mismatch", lambda ns: np.union1d(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "setxor1d": [("1d", lambda ns: np.setxor1d(ns.m, ns.m[:2]), "m")],
    "setdiff1d": [
        ("1d", lambda ns: np.setdiff1d(ns.m, ns.m[:2]), "m"),
        ("mismatch", lambda ns: np.setdiff1d(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "intersect1d": [
        ("1d", lambda ns: np.intersect1d(ns.m, ns.m), "m"),
        (
            "indices",
            lambda ns: np.intersect1d(ns.m, ns.m, return_indices=True),
            ("m", "plain", "plain"),
        ),
    ],
    # ---- HANDLED: comparing and summarising ---------------------------------------------
    "allclose": [
        ("same", lambda ns: np.allclose(ns.m, ns.m), "plain"),
        ("atol", lambda ns: np.allclose(ns.m, ns.m + 1e-3 * ns.unit, atol=1e-2 * ns.unit), "plain"),
        ("mismatch", lambda ns: np.allclose(ns.m, ns.s), "DimensionMismatchError"),
        (
            "atol mismatch",
            lambda ns: np.allclose(ns.m, ns.m, atol=1 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "isclose": [
        ("same", lambda ns: np.isclose(ns.m, ns.m), "plain"),
        ("mismatch", lambda ns: np.isclose(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "array_equal": [("same", lambda ns: np.array_equal(ns.m, ns.m), "plain")],
    "array_equiv": [("same", lambda ns: np.array_equiv(ns.m, ns.m), "plain")],
    "cov": [
        ("1d", lambda ns: np.cov(ns.m), "m2"),
        ("mismatch", lambda ns: np.cov(ns.m, ns.s), "DimensionMismatchError"),
    ],
    "bincount": [
        ("counts", lambda ns: np.bincount(ns.i), "plain"),
        ("weights", lambda ns: np.bincount(ns.i, weights=ns.m), "m"),
        ("quantity input", lambda ns: np.bincount(ns.m), "DimensionMismatchError"),
    ],
    "histogram": [
        ("counts", lambda ns: np.histogram(ns.m, bins=2), ("plain", "m")),
        ("weights", lambda ns: np.histogram(ns.m, bins=2, weights=ns.s), ("s", "m")),
        ("density", lambda ns: np.histogram(ns.m, bins=2, density=True), ("1/m", "m")),
        ("bin edges", lambda ns: np.histogram(ns.m, bins=ns.m), ("plain", "m")),
        ("mismatch", lambda ns: np.histogram(ns.m, bins=ns.s), "DimensionMismatchError"),
        (
            "range",
            lambda ns: np.histogram(ns.m, bins=2, range=(0 * ns.unit, 5 * ns.unit)),
            ("plain", "m"),
        ),
        (
            "range mismatch",
            lambda ns: np.histogram(ns.m, range=(0 * ns.second, 5 * ns.second)),
            "DimensionMismatchError",
        ),
    ],
    "histogram_bin_edges": [("1d", lambda ns: np.histogram_bin_edges(ns.m, bins=2), "m")],
    "histogram2d": [
        ("counts", lambda ns: np.histogram2d(ns.m, ns.s, bins=2), ("plain", "m", "s")),
        (
            "bin edges per axis",
            lambda ns: np.histogram2d(ns.m, ns.s, bins=[ns.m, ns.s]),
            ("plain", "m", "s"),
        ),
        (
            "range per axis",
            lambda ns: np.histogram2d(
                ns.m, ns.s, bins=[2, 3], range=[(0 * ns.unit, 5 * ns.unit), None]
            ),
            ("plain", "m", "s"),
        ),
        (
            "bin edge mismatch",
            lambda ns: np.histogram2d(ns.m, ns.s, bins=[ns.s, ns.m]),
            "DimensionMismatchError",
        ),
        (
            "range mismatch",
            lambda ns: np.histogram2d(ns.m, ns.s, range=[(0 * ns.second, 5 * ns.second), None]),
            "DimensionMismatchError",
        ),
        (
            "density",
            lambda ns: np.histogram2d(ns.m, ns.s, bins=2, density=True),
            ("1/(m s)", "m", "s"),
        ),
    ],
    "histogramdd": [
        ("counts", lambda ns: np.histogramdd(ns.m2.T, bins=2), ("plain", ("m", "m"))),
        ("density", lambda ns: np.histogramdd(ns.m2.T, bins=2, density=True), ("1/m2", ("m", "m"))),
        (
            "separate axes",
            lambda ns: np.histogramdd((ns.m, ns.s), bins=2, weights=ns.m, density=True),
            ("1/(m s)", ("m", "s")),
        ),
    ],
    "interp": [
        ("1d", lambda ns: np.interp(ns.m, ns.m, ns.s), "s"),
        ("mismatch", lambda ns: np.interp(ns.m, ns.s, ns.s), "DimensionMismatchError"),
        (
            "left",
            lambda ns: np.interp(ns.m, ns.m, ns.s, left=1 * ns.unit),
            "DimensionMismatchError",
        ),
    ],
    "geomspace": [
        ("1d", lambda ns: np.geomspace(1 * ns.unit, 4 * ns.unit, 3), "m"),
        (
            "mismatch",
            lambda ns: np.geomspace(1 * ns.unit, 4 * ns.second, 3),
            "DimensionMismatchError",
        ),
    ],
    "gradient": [
        ("1d", lambda ns: np.gradient(ns.m), "m"),
        ("spacing", lambda ns: np.gradient(ns.m, ns.s), "m/s"),
        ("2d", lambda ns: np.gradient(ns.m2), ("m", "m")),
        (
            "spacing per axis",
            lambda ns: np.gradient(ns.m2, 2 * ns.unit, 3 * ns.second),
            ("plain", "m/s"),
        ),
    ],
    "cumprod": [
        ("with dimensions", lambda ns: np.cumprod(ns.M * ns.M), "TypeError"),
    ],
    "nancumprod": [
        ("with dimensions", lambda ns: np.nancumprod(ns.m * ns.m), "TypeError"),
    ],
    # ---- HANDLED: products and linear algebra -------------------------------------------
    "dot": [
        ("1d", lambda ns: np.dot(ns.m, ns.s), "m s"),
        ("2d", lambda ns: np.dot(ns.M, ns.M), "m2"),
        ("out", lambda ns: np.dot(ns.M, ns.M, out=np.zeros((2, 2)) * ns.second), "m2"),
    ],
    "inner": [("1d", lambda ns: np.inner(ns.m, ns.s), "m s")],
    "vdot": [("1d", lambda ns: np.vdot(ns.m, ns.s), "m s")],
    "outer": [("1d", lambda ns: np.outer(ns.m, ns.s), "m s")],
    "linalg.outer": [("1d", lambda ns: np.linalg.outer(ns.m, ns.s), "m s")],
    "tensordot": [("1d", lambda ns: np.tensordot(ns.m, ns.s, 1), "m s")],
    "linalg.tensordot": [("1d", lambda ns: np.linalg.tensordot(ns.m, ns.s, axes=1), "m s")],
    "einsum": [
        ("dot", lambda ns: np.einsum("i,i", ns.m, ns.s), "m s"),
        ("trace", lambda ns: np.einsum("ii", ns.M), "m"),
        ("matmul", lambda ns: np.einsum("ij,jk", ns.M, ns.M), "m2"),
        ("sublists", lambda ns: np.einsum(ns.M, [0, 1], ns.M, [1, 2]), "m2"),
    ],
    "cross": [("3d", lambda ns: np.cross(ns.m[:3], ns.s[:3]), "m s")],
    "linalg.cross": [("3d", lambda ns: np.linalg.cross(ns.m[:3], ns.s[:3]), "m s")],
    "convolve": [("1d", lambda ns: np.convolve(ns.m, ns.s), "m s")],
    "correlate": [("1d", lambda ns: np.correlate(ns.m, ns.s), "m s")],
    "linalg.multi_dot": [("2d", lambda ns: np.linalg.multi_dot([ns.M, ns.M, ns.M]), "m3")],
    "linalg.det": [("2d", lambda ns: np.linalg.det(ns.M), "m2")],
    "linalg.inv": [("2d", lambda ns: np.linalg.inv(ns.M), "1/m")],
    "linalg.pinv": [("2d", lambda ns: np.linalg.pinv(ns.M), "1/m")],
    "linalg.solve": [("2d", lambda ns: np.linalg.solve(ns.M, ns.s[:2]), "s/m")],
    "linalg.eig": [("2d", lambda ns: np.linalg.eig(ns.M), ("m", "plain"))],
    "linalg.eigh": [("2d", lambda ns: np.linalg.eigh(ns.M), ("m", "plain"))],
    "linalg.eigvals": [("2d", lambda ns: np.linalg.eigvals(ns.M), "m")],
    "linalg.eigvalsh": [("2d", lambda ns: np.linalg.eigvalsh(ns.M), "m")],
    "linalg.svd": [
        ("2d", lambda ns: np.linalg.svd(ns.M), ("plain", "m", "plain")),
        ("values only", lambda ns: np.linalg.svd(ns.M, compute_uv=False), "m"),
    ],
    "linalg.svdvals": [("2d", lambda ns: np.linalg.svdvals(ns.M), "m")],
    "linalg.qr": [
        ("2d", lambda ns: np.linalg.qr(ns.M), ("plain", "m")),
        ("r", lambda ns: np.linalg.qr(ns.M, mode="r"), "m"),
        ("raw", lambda ns: np.linalg.qr(ns.M, mode="raw"), "TypeError"),
    ],
    "linalg.cholesky": [("2d", lambda ns: np.linalg.cholesky(ns.M), "m^0.5")],
    "linalg.norm": [
        ("1d", lambda ns: np.linalg.norm(ns.m), "m"),
        ("count", lambda ns: np.linalg.norm(ns.m, 0), "plain"),
    ],
    "linalg.vector_norm": [
        ("1d", lambda ns: np.linalg.vector_norm(ns.m), "m"),
        ("count", lambda ns: np.linalg.vector_norm(ns.m, ord=0), "plain"),
    ],
    "linalg.matrix_norm": [("2d", lambda ns: np.linalg.matrix_norm(ns.M), "m")],
    # ---- HANDLED: functions of dimensionless numbers ------------------------------------
    "i0": [("with dimensions", lambda ns: np.i0(ns.m), "DimensionMismatchError")],
    "sinc": [("with dimensions", lambda ns: np.sinc(ns.m), "DimensionMismatchError")],
    "unwrap": [("with dimensions", lambda ns: np.unwrap(ns.m), "DimensionMismatchError")],
    "lib.scimath.arccos": [
        ("with dimensions", lambda ns: np.lib.scimath.arccos(ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.arcsin": [
        ("with dimensions", lambda ns: np.lib.scimath.arcsin(ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.arctanh": [
        ("with dimensions", lambda ns: np.lib.scimath.arctanh(ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.log": [
        ("with dimensions", lambda ns: np.lib.scimath.log(ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.log10": [
        ("with dimensions", lambda ns: np.lib.scimath.log10(ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.log2": [
        ("with dimensions", lambda ns: np.lib.scimath.log2(ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.logn": [
        ("with dimensions", lambda ns: np.lib.scimath.logn(2, ns.m), "DimensionMismatchError")
    ],
    "lib.scimath.sqrt": [("1d", lambda ns: np.lib.scimath.sqrt(ns.m), "m^0.5")],
    "lib.scimath.power": [
        ("1d", lambda ns: np.lib.scimath.power(ns.m, 2), "m2"),
        (
            "multiple exponents",
            lambda ns: np.lib.scimath.power(ns.m, [1, 2, 3, 4]),
            "TypeError",
        ),
        (
            "exponent with dimensions",
            lambda ns: np.lib.scimath.power(ns.m, ns.m),
            "DimensionMismatchError",
        ),
    ],
    # ---- UNIT_FREE ------------------------------------------------------------------
    "argmax": [
        ("1d", lambda ns: np.argmax(ns.m), "plain"),
        ("axis", lambda ns: np.argmax(ns.m2, axis=0), "plain"),
        ("method", lambda ns: ns.m2.argmax(axis=1), "plain"),
    ],
    "argmin": [
        ("1d", lambda ns: np.argmin(ns.m), "plain"),
        ("axis", lambda ns: np.argmin(ns.m2, axis=0), "plain"),
        ("method", lambda ns: ns.m2.argmin(axis=1), "plain"),
    ],
    "argsort": [
        ("1d", lambda ns: np.argsort(ns.m), "plain"),
        ("method", lambda ns: ns.m2.argsort(axis=0), "plain"),
    ],
    "argpartition": [
        ("1d", lambda ns: np.argpartition(ns.m, 1), "plain"),
        ("method", lambda ns: ns.m.argpartition(1), "plain"),
    ],
    "argwhere": [("1d", lambda ns: np.argwhere(ns.m), "plain")],
    "flatnonzero": [("1d", lambda ns: np.flatnonzero(ns.m), "plain")],
    "nonzero": [("1d", lambda ns: np.nonzero(ns.m), ("plain",))],
    "lexsort": [("1d", lambda ns: np.lexsort([ns.m]), "plain")],
    "nanargmax": [("1d", lambda ns: np.nanargmax(ns.m), "plain")],
    "nanargmin": [("1d", lambda ns: np.nanargmin(ns.m), "plain")],
    "ix_": [("1d", lambda ns: np.ix_(ns.i, ns.i), ("plain", "plain"))],
    "diag_indices_from": [("2d", lambda ns: np.diag_indices_from(ns.M), ("plain", "plain"))],
    "tril_indices_from": [("2d", lambda ns: np.tril_indices_from(ns.M), ("plain", "plain"))],
    "triu_indices_from": [("2d", lambda ns: np.triu_indices_from(ns.M), ("plain", "plain"))],
    "unravel_index": [("1d", lambda ns: np.unravel_index(ns.i, (5, 5)), ("plain", "plain"))],
    "ravel_multi_index": [("1d", lambda ns: np.ravel_multi_index((ns.i, ns.i), (5, 5)), "plain")],
    "count_nonzero": [("1d", lambda ns: np.count_nonzero(ns.m), "plain")],
    "isneginf": [("1d", lambda ns: np.isneginf(ns.m), "plain")],
    "isposinf": [("1d", lambda ns: np.isposinf(ns.m), "plain")],
    "iscomplex": [("1d", lambda ns: np.iscomplex(ns.m), "plain")],
    "isreal": [("1d", lambda ns: np.isreal(ns.m), "plain")],
    "iscomplexobj": [("1d", lambda ns: np.iscomplexobj(ns.m), "plain")],
    "isrealobj": [("1d", lambda ns: np.isrealobj(ns.m), "plain")],
    "may_share_memory": [("1d", lambda ns: np.may_share_memory(ns.m, ns.m), "plain")],
    "shares_memory": [("1d", lambda ns: np.shares_memory(ns.m, ns.m), "plain")],
    "corrcoef": [("1d", lambda ns: np.corrcoef(ns.m, ns.m), "plain")],
    "angle": [("1d", lambda ns: np.angle(ns.m), "plain")],
    "linalg.cond": [("2d", lambda ns: np.linalg.cond(ns.M), "plain")],
    "linalg.matrix_rank": [("2d", lambda ns: np.linalg.matrix_rank(ns.M), "plain")],
    "ndim": [("1d", lambda ns: np.ndim(ns.m), "plain")],
    "size": [("1d", lambda ns: np.size(ns.m), "plain")],
    "shape": [("1d", lambda ns: np.shape(ns.m), "plain")],
    "can_cast": [("1d", lambda ns: np.can_cast(ns.m, np.float64), "plain")],
    "common_type": [("1d", lambda ns: np.common_type(ns.m), "plain")],
    "min_scalar_type": [("1d", lambda ns: np.min_scalar_type(ns.m), "plain")],
    "result_type": [("1d", lambda ns: np.result_type(ns.m), "plain")],
    "einsum_path": [("dot", lambda ns: np.einsum_path("i,i", ns.m, ns.s), "plain")],
    "array2string": [("1d", lambda ns: np.array2string(ns.m), "plain")],
    "array_repr": [("1d", lambda ns: np.array_repr(ns.m), "plain")],
    "array_str": [("1d", lambda ns: np.array_str(ns.m), "plain")],
    "save": [("1d", lambda ns: np.save(ns.tmp / "a.npy", ns.m), "none")],
    "savetxt": [("1d", lambda ns: np.savetxt(ns.tmp / "a.txt", ns.m), "none")],
    "savez": [("1d", lambda ns: np.savez(ns.tmp / "a.npz", ns.m), "none")],
    "savez_compressed": [("1d", lambda ns: np.savez_compressed(ns.tmp / "b.npz", ns.m), "none")],
    "arange": [("like", lambda ns: np.arange(3, like=ns.m), "plain")],
    "array": [("like", lambda ns: np.array([1.0], like=ns.m), "plain")],
    "asarray": [("like", lambda ns: np.asarray([1.0], like=ns.m), "plain")],
    "asanyarray": [("like", lambda ns: np.asanyarray([1.0], like=ns.m), "plain")],
    "ascontiguousarray": [("like", lambda ns: np.ascontiguousarray([1.0], like=ns.m), "plain")],
    "asfortranarray": [("like", lambda ns: np.asfortranarray([1.0], like=ns.m), "plain")],
    "empty": [("like", lambda ns: np.empty(3, like=ns.m).shape, "plain")],
    "zeros": [("like", lambda ns: np.zeros(2, like=ns.m), "plain")],
    "ones": [("like", lambda ns: np.ones(2, like=ns.m), "plain")],
    "full": [("like", lambda ns: np.full(2, 1.0, like=ns.m), "plain")],
    "eye": [("like", lambda ns: np.eye(2, like=ns.m), "plain")],
    "identity": [("like", lambda ns: np.identity(2, like=ns.m), "plain")],
    "tri": [("like", lambda ns: np.tri(2, like=ns.m), "plain")],
    "require": [("like", lambda ns: np.require([1.0], like=ns.m), "plain")],
    "frombuffer": [("like", lambda ns: np.frombuffer(b"\0" * 8, like=ns.m), "plain")],
    "fromfile": [
        ("like", lambda ns: np.fromfile(__file__, dtype=np.uint8, count=1, like=ns.m), "plain")
    ],
    "fromfunction": [("like", lambda ns: np.fromfunction(lambda a: a, (2,), like=ns.m), "plain")],
    "fromiter": [("like", lambda ns: np.fromiter([1.0], float, like=ns.m), "plain")],
    "fromstring": [("like", lambda ns: np.fromstring("1 2", sep=" ", like=ns.m), "plain")],
    "loadtxt": [("like", lambda ns: np.loadtxt(["1 2"], like=ns.m), "plain")],
    "genfromtxt": [("like", lambda ns: np.genfromtxt(["1 2"], like=ns.m), "plain")],
    # ---- UNSUPPORTED ----------------------------------------------------------------
    "busday_count": [("1d", lambda ns: np.busday_count(ns.m, ns.m), "TypeError")],
    "busday_offset": [("1d", lambda ns: np.busday_offset(ns.m, ns.m), "TypeError")],
    "is_busday": [("1d", lambda ns: np.is_busday(ns.m), "TypeError")],
    "datetime_as_string": [("1d", lambda ns: np.datetime_as_string(ns.m), "TypeError")],
    "packbits": [("1d", lambda ns: np.packbits(ns.m), "TypeError")],
    "unpackbits": [("1d", lambda ns: np.unpackbits(ns.m), "TypeError")],
    "poly": [("1d", lambda ns: np.poly(ns.m), "TypeError")],
    "polyadd": [("1d", lambda ns: np.polyadd(ns.m, ns.m), "TypeError")],
    "polyder": [("1d", lambda ns: np.polyder(ns.m), "TypeError")],
    "polydiv": [("1d", lambda ns: np.polydiv(ns.m, ns.m), "TypeError")],
    "polyfit": [("1d", lambda ns: np.polyfit(ns.d, ns.m, 1), "TypeError")],
    "polyint": [("1d", lambda ns: np.polyint(ns.m), "TypeError")],
    "polymul": [("1d", lambda ns: np.polymul(ns.m, ns.d), "TypeError")],
    "polysub": [("1d", lambda ns: np.polysub(ns.m, ns.m), "TypeError")],
    "polyval": [("1d", lambda ns: np.polyval(ns.m, 2), "TypeError")],
    "roots": [("1d", lambda ns: np.roots(ns.m), "TypeError")],
    "vander": [("1d", lambda ns: np.vander(ns.m, 3), "TypeError")],
    "apply_over_axes": [("1d", lambda ns: np.apply_over_axes(np.sum, ns.m2, [0]), "TypeError")],
    "logspace": [("1d", lambda ns: np.logspace(0 * ns.unit, 1 * ns.unit, 3), "TypeError")],
    "linalg.lstsq": [("2d", lambda ns: np.linalg.lstsq(ns.M, ns.m[:2]), "TypeError")],
    "linalg.tensorinv": [("2d", lambda ns: np.linalg.tensorinv(ns.M, 1), "TypeError")],
    "linalg.tensorsolve": [("2d", lambda ns: np.linalg.tensorsolve(ns.M, ns.m[:2]), "TypeError")],
    "linalg.slogdet": [("2d", lambda ns: np.linalg.slogdet(ns.M), "TypeError")],
}


if hasattr(np, "in1d"):  # NumPy < 2.4
    CASES["in1d"] = [
        ("1d", lambda ns: without_deprecations(np.in1d, ns.m, ns.m[:2]), "plain"),
        (
            "mismatch",
            lambda ns: without_deprecations(np.in1d, ns.m, ns.s),
            "DimensionMismatchError",
        ),
    ]


def numpy_function(name):
    module = np
    for part in name.split(".")[:-1]:
        module = getattr(module, part)
    return getattr(module, name.split(".")[-1])


def all_cases():
    for name, cases in CASES.items():
        for label, call, expected in cases:
            yield pytest.param(call, expected, id=f"{name}-{label}")


def check(result, plain, expected):
    if isinstance(expected, tuple):
        assert isinstance(result, (tuple, list)) and len(result) == len(expected)
        for item, plain_item, expected_item in zip(result, plain, expected, strict=True):
            check(item, plain_item, expected_item)
    elif expected == "none":
        assert result is None
    elif expected == "plain":
        assert not isinstance(result, Quantity)
        if isinstance(result, np.ndarray) and result.dtype.kind in "biufc":
            np.testing.assert_allclose(result, plain)
    else:
        assert isinstance(result, Quantity), f"expected a Quantity, got {type(result)}"
        assert result.dim is DIMS[expected], f"dimensions {result.dim}, expected {expected}"
        np.testing.assert_allclose(np.asarray(result), np.asarray(plain))


@pytest.mark.parametrize(("call", "expected"), list(all_cases()))
def test_numpy_function(call, expected, tmp_path):
    quantities = namespace(metre, second, tmp_path)
    if expected in EXCEPTIONS:
        with pytest.raises(EXCEPTIONS[expected]):
            call(quantities)
        return
    result = call(quantities)
    plain = call(namespace(1.0, 1.0, pathlib.Path(tmp_path)))
    check(result, plain, expected)


def test_every_classified_function_has_a_case():
    classified = (
        _array_functions.SUBCLASS_SAFE
        | _array_functions.UNIT_FREE
        | set(_array_functions.UNSUPPORTED)
        | set(_array_functions.HANDLED)
    )
    covered = {numpy_function(name) for name in CASES}
    assert not {f.__name__ for f in classified - covered}


def test_buckets_do_not_overlap():
    buckets = [
        _array_functions.SUBCLASS_SAFE,
        _array_functions.UNIT_FREE,
        set(_array_functions.UNSUPPORTED),
        set(_array_functions.HANDLED),
    ]
    for i, first in enumerate(buckets):
        for second_bucket in buckets[i + 1 :]:
            assert not first & second_bucket


def test_quantities_of_different_kinds_are_never_equal():
    assert not np.array_equal(np.ones(3) * metre, np.ones(3) * second)
    assert not np.array_equiv(np.ones(3) * metre, np.ones(3) * second)


def test_dimensionless_functions_accept_dimensionless_quantities():
    x = Quantity([0.1, 0.2])  # an explicitly constructed, dimensionless Quantity
    np.testing.assert_allclose(np.sinc(x), np.sinc([0.1, 0.2]))


@pytest.mark.parametrize("function", [np.cumprod, np.nancumprod, np.cumulative_prod])
def test_dimensionless_cumulative_products(function):
    values = np.array([2.0, 3.0, 4.0])
    result = function(Quantity(values))
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, function(values))


def test_concatenate_dimensionless_quantities():
    result = np.concatenate([Quantity([1, 2]), np.array([3, 4])])
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, [1, 2, 3, 4])


@pytest.mark.parametrize("function", [np.concatenate, np.stack, np.block])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_zero_arrays_do_not_hide_mixed_dimensions(function, order):
    arrays = [np.zeros(1), np.array([1.0]) * metre, np.array([2.0]) * second]
    with pytest.raises(DimensionMismatchError, match="same dimensions"):
        function([arrays[i] for i in order])


@pytest.mark.parametrize("function", [np.concatenate, np.stack, np.block])
@pytest.mark.parametrize("zero_position", range(3))
def test_zero_arrays_can_join_matching_dimensions(function, zero_position):
    arrays = [np.array([1.0]) * metre, np.array([2.0]) * metre]
    arrays.insert(zero_position, np.zeros(1))
    result = function(arrays)
    assert result.dim is metre.dim
    np.testing.assert_array_equal(np.asarray(result), function([np.asarray(a) for a in arrays]))


@pytest.mark.parametrize("function", [np.concatenate, np.stack])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("positional_out", [False, True])
def test_joining_into_matching_output(function, dtype, axis, positional_out):
    arrays = [np.arange(4.0).reshape(2, 2) * metre, np.ones((2, 2)) * metre]
    expected = function([np.asarray(a) for a in arrays], axis=axis, dtype=dtype)
    out = Quantity(np.zeros_like(expected), dim=metre.dim)
    if positional_out:
        result = function(arrays, axis, out)
    else:
        result = function(arrays, axis=axis, out=out)
    assert result is out
    assert out.dim is metre.dim
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out), expected)
    with pytest.raises(TypeError):
        function(arrays, axis=axis, out=out, dtype=dtype)


@pytest.mark.parametrize("function", [np.concatenate, np.stack])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output_dim", [DIMENSIONLESS, S])
@pytest.mark.parametrize("fill_value", [0, 7])
def test_joining_rejects_incompatible_output_before_writing(function, dtype, output_dim, fill_value):
    arrays = [np.array([1.0, 2.0]) * metre, np.array([3.0, 4.0]) * metre]
    shape = (4,) if function is np.concatenate else (2, 2)
    out = Quantity(np.full(shape, fill_value, dtype=dtype), dim=output_dim)
    before = np.asarray(out).copy()
    with pytest.raises(DimensionMismatchError, match="out"):
        function(arrays, out=out)
    assert getattr(out, "dim", DIMENSIONLESS) is output_dim
    np.testing.assert_array_equal(np.asarray(out), before)


@pytest.mark.parametrize("function", [np.concatenate, np.stack])
def test_joining_dimensionless_values_into_plain_output(function):
    arrays = [Quantity([1, 2]), np.array([3, 4])]
    expected = function([np.asarray(a) for a in arrays])
    out = np.empty_like(expected)
    assert function(arrays, out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_where_with_quantity_condition_returns_plain_indices():
    (indices,) = np.where(Quantity([True, False, True]))
    assert type(indices) is np.ndarray
    np.testing.assert_array_equal(indices, [0, 2])


def test_record_array_functions_are_refused():
    from numpy.lib import recfunctions

    with pytest.raises(TypeError, match="structured"):
        recfunctions.structured_to_unstructured(np.ones(3) * metre)


def public_numpy_function(function):
    """NumPy's list of overridable functions contains internal dispatchers for the
    like= argument (np.eye's, np.ones', ...); NumPy passes the public function to
    __array_function__, so that is what has to be classified."""
    public = getattr(np, function.__name__, None)
    if function.__module__ == "numpy" and public is not None and public is not function:
        return public
    return function


def test_every_numpy_function_is_classified():
    # A NumPy release that adds a function fails this test until the function is
    # put into a bucket (with a case above). Until then, users get a QuantSIWarning.
    from numpy.testing.overrides import get_overridable_numpy_array_functions

    classified = (
        _array_functions.SUBCLASS_SAFE
        | _array_functions.UNIT_FREE
        | set(_array_functions.UNSUPPORTED)
        | set(_array_functions.HANDLED)
    )
    unclassified = sorted(
        f"{function.__module__}.{function.__name__}"
        for function in map(public_numpy_function, get_overridable_numpy_array_functions())
        if function not in classified
        and function.__module__ not in _array_functions.UNSUPPORTED_MODULES
    )
    assert not unclassified


def test_unknown_functions_run_with_a_warning(monkeypatch):
    from QuantSI import QuantSIWarning

    # Pretend np.sum had not been reviewed.
    monkeypatch.setattr(
        _array_functions, "SUBCLASS_SAFE", _array_functions.SUBCLASS_SAFE - {np.sum}
    )
    from QuantSI._core import quantity as _quantity

    monkeypatch.setattr(_quantity, "SUBCLASS_SAFE", _array_functions.SUBCLASS_SAFE)
    with pytest.warns(QuantSIWarning, match="numpy.sum has not been reviewed"):
        result = np.sum(np.ones(3) * metre)
    assert result == 3 * metre


def test_other_array_types_get_their_turn():
    # NEP 18: with an unknown array type among the arguments, return NotImplemented
    # so that NumPy can ask the other type (here it has no answer either).
    class Other:
        def __array_function__(self, func, types, args, kwargs):
            return NotImplemented

    with pytest.raises(TypeError, match="no implementation found"):
        np.concatenate([np.ones(2) * metre, Other()])
