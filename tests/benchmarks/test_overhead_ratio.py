"""Check that QuantSI's overhead compared with plain NumPy doesn't grow.

Timings on CI runners are noisy, but timing a QuantSI operation and the same
operation on plain NumPy arrays back to back, in the same process, gives a ratio
that is much more stable. Each limit is about twice the ratio measured when it
was set, so noise doesn't fail the build, but a real slowdown does (for example
work that is now done on every call instead of once). When a change makes
QuantSI faster, lower the limit in the same commit.

These tests don't run by default; run them with ``pytest -m perf_guard``.
"""

import timeit

import numpy as np
import pytest

from QuantSI.allunits import mvolt, ohm, volt

pytestmark = pytest.mark.perf_guard

#: (case, QuantSI statement, plain NumPy statement, maximum ratio)
CASES = [
    ("array_add", "A + B", "a + b", 8),
    ("array_multiply", "A * R", "a * b", 9),
    ("array_compare", "A < B", "a < b", 5),
    ("array_sum", "np.sum(A)", "np.sum(a)", 6),
    ("array_index", "A[10]", "a[10]", 19),
    ("scalar_add", "X + Y", "x + y", 12),
    ("scalar_multiply", "X * Y", "x * y", 14),
    ("number_times_unit", "3 * mvolt", "3 * x0", 12),
    ("str_scalar", "str(X)", "str(x)", 80),
]


@pytest.fixture(scope="module")
def namespace():
    a, b = np.linspace(1, 2, 1000), np.linspace(2, 3, 1000)
    return {
        "np": np,
        "mvolt": mvolt,
        "a": a,
        "b": b,
        "A": a * volt,
        "B": b * volt,
        "R": b * ohm,
        "x": np.asarray(3.0),
        "y": np.asarray(5.0),
        "x0": np.asarray(0.001),
        "X": 3 * mvolt,
        "Y": 5 * mvolt,
    }


def best_time(statement, namespace):
    """Seconds per run of ``statement``: the best of 7 repeats."""
    timer = timeit.Timer(statement, globals=namespace)
    number, _ = timer.autorange()
    return min(timer.repeat(7, number)) / number


@pytest.mark.parametrize(("case", "quantity", "plain", "limit"), CASES, ids=[c[0] for c in CASES])
def test_overhead_ratio(namespace, case, quantity, plain, limit):
    ratio = best_time(quantity, namespace) / best_time(plain, namespace)
    assert ratio < limit, f"{case}: {ratio:.1f}x plain NumPy (limit {limit}x)"
