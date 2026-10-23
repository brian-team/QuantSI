"""Dimension behaviour, including equality checked against the old implementation."""

import copy

import numpy as np
import pytest

from QuantSI.allunits import amp, metre, volt
from QuantSI.fundamentalunits import DIMENSIONLESS, get_or_create_dimension


def old_eq(dim1, dim2):
    """Dimension.__eq__ as it was before it was rewritten for speed."""
    try:
        return np.allclose(dim1._dims, dim2._dims)
    except AttributeError:
        return False


def dimension_pairs():
    rng = np.random.default_rng(1)
    pairs = [(volt.dim, volt.dim), (volt.dim, amp.dim), (DIMENSIONLESS, metre.dim)]
    # Floating-point drift: equal but not identical tuples.
    pairs.append((get_or_create_dimension(length=0.1 + 0.2), get_or_create_dimension(length=0.3)))
    for _ in range(300):
        base = rng.integers(-4, 5, size=7).astype(float) * rng.choice([1, 0.5, 1 / 3])
        # Offsets around the tolerance boundary of np.allclose, absolute and relative.
        offset = rng.choice([0, 1e-9, 1e-8, 2e-8, 1e-6, 1e-5, 1e-4]) * rng.choice([-1, 1])
        other = base + offset * rng.integers(0, 2, size=7)
        pairs.append((get_or_create_dimension(tuple(base)), get_or_create_dimension(tuple(other))))
    return pairs


@pytest.mark.parametrize(("dim1", "dim2"), dimension_pairs())
def test_equality_matches_old_implementation(dim1, dim2):
    assert (dim1 == dim2) == old_eq(dim1, dim2)
    assert (dim1 != dim2) == (not old_eq(dim1, dim2))


def test_equality_with_other_types():
    assert volt.dim != (2, 1, -3, -1, 0, 0, 0)
    assert volt.dim != volt
    assert not (volt.dim == None)  # noqa: E711


def test_interning_makes_equal_dimensions_identical():
    assert get_or_create_dimension(m=2, kg=1, s=-3, A=-1) is volt.dim


def uncached_product(dim1, dim2):
    return get_or_create_dimension([x + y for x, y in zip(dim1._dims, dim2._dims, strict=True)])


@pytest.mark.parametrize(("dim1", "dim2"), dimension_pairs()[:60])
def test_cached_arithmetic_matches_direct_computation(dim1, dim2):
    for _ in range(2):  # the second round is answered from the caches
        assert dim1 * dim2 is uncached_product(dim1, dim2)
        assert dim1 / dim2 is get_or_create_dimension(
            [x - y for x, y in zip(dim1._dims, dim2._dims, strict=True)]
        )
        assert dim1**2 is get_or_create_dimension([2 * x for x in dim1._dims])
        assert dim1**0.5 is get_or_create_dimension([0.5 * x for x in dim1._dims])


def test_cache_entries_keep_their_operands_alive():
    from QuantSI._core import dimension as _dimension

    dim1 = get_or_create_dimension(m=7, kg=-3)
    product = dim1 * volt.dim
    entry = _dimension._products[(id(dim1), id(volt.dim))]
    assert entry[0] is dim1 and entry[1] is volt.dim and entry[2] is product


def test_full_cache_still_computes(monkeypatch):
    from QuantSI._core import dimension as _dimension

    monkeypatch.setattr(_dimension, "_MAX_CACHE_ENTRIES", 0)
    monkeypatch.setattr(_dimension, "_products", {})
    dim1 = get_or_create_dimension(m=5)
    assert dim1 * volt.dim is uncached_product(dim1, volt.dim)
    assert _dimension._products == {}


def test_arithmetic_with_a_non_dimension_fails():
    with pytest.raises(AttributeError):
        volt.dim * 3


def test_dimension_power_rejects_multiple_exponents():
    with pytest.raises(TypeError, match="Too many exponents"):
        metre.dim ** np.array([1, 2])


def test_deepcopy_preserves_dimension_singletons():
    assert copy.deepcopy(metre.dim) is metre.dim
