"""Benchmarks for the operations every QuantSI user pays for.

In a normal test run each benchmark runs just once, as a plain test
(``--benchmark-disable`` in pyproject.toml), so they keep working. To measure::

    pytest tests/benchmarks --benchmark-enable --benchmark-only

To compare with an earlier run, save it with ``--benchmark-autosave`` and pass
``--benchmark-compare`` next time. The numbers only mean something when you
compare runs on the same machine.
"""

import pickle

import numpy as np
import pytest

from QuantSI import check_units
from QuantSI.allunits import amp, mvolt, ohm, volt
from QuantSI.fundamentalunits import Quantity, get_or_create_dimension


@pytest.fixture(scope="module")
def arrays():
    # Fixed seed, so every run times the same data.
    rng = np.random.default_rng(42)
    a, b = rng.random(1000), rng.random(1000)
    return {"a": a, "b": b, "A": a * volt, "B": b * volt, "R": b * ohm}


# Scalars ---------------------------------------------------------------------


def test_number_times_unit(benchmark):
    benchmark(lambda: 3 * mvolt)


def test_scalar_add(benchmark):
    x, y = 3 * mvolt, 5 * mvolt
    benchmark(lambda: x + y)


def test_scalar_multiply(benchmark):
    x, y = 3 * mvolt, 2 * ohm
    benchmark(lambda: x * y)


def test_scalar_compare(benchmark):
    x, y = 3 * mvolt, 5 * mvolt
    benchmark(lambda: x < y)


# Arrays ----------------------------------------------------------------------


def test_array_add(benchmark, arrays):
    A, B = arrays["A"], arrays["B"]
    benchmark(lambda: A + B)


def test_array_multiply(benchmark, arrays):
    A, R = arrays["A"], arrays["R"]
    benchmark(lambda: A * R)


def test_array_compare(benchmark, arrays):
    A, B = arrays["A"], arrays["B"]
    benchmark(lambda: A < B)


def test_array_sum(benchmark, arrays):
    A = arrays["A"]
    benchmark(lambda: np.sum(A))


def test_array_index_scalar(benchmark, arrays):
    A = arrays["A"]
    benchmark(lambda: A[10])


def test_array_slice(benchmark, arrays):
    A = arrays["A"]
    benchmark(lambda: A[10:20])


def test_strip_units(benchmark, arrays):
    A = arrays["A"]
    benchmark(lambda: np.asarray(A))


# Machinery -------------------------------------------------------------------


def test_dimension_identity(benchmark):
    d1, d2 = volt.dim, mvolt.dim
    benchmark(lambda: d1 is d2)


def test_dimension_equality(benchmark):
    d1, d2 = volt.dim, amp.dim
    benchmark(lambda: d1 == d2)


def test_dimension_multiply(benchmark):
    d1, d2 = volt.dim, amp.dim
    benchmark(lambda: d1 * d2)


def test_get_or_create_dimension(benchmark):
    benchmark(lambda: get_or_create_dimension(m=1, kg=1, s=-2))


def test_quantity_constructor(benchmark, arrays):
    a, dim = arrays["a"], volt.dim
    benchmark(lambda: Quantity(a, dim=dim))


def test_check_units_call(benchmark):
    @check_units(v=volt, i=amp, result=ohm)
    def resistance(v, i):
        return v / i

    v, i = 3 * mvolt, 2 * amp
    benchmark(lambda: resistance(v, i))


def test_pickle_roundtrip(benchmark, arrays):
    A = arrays["A"]
    benchmark(lambda: pickle.loads(pickle.dumps(A)))


# Display ---------------------------------------------------------------------


def test_str_scalar(benchmark):
    x = 3 * mvolt
    benchmark(lambda: str(x))


def test_repr_array(benchmark, arrays):
    A = arrays["A"]
    benchmark(lambda: repr(A))
