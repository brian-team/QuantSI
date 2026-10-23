"""Pickles must keep working in both directions.

- Old pickles: data/pickles.json holds pickles made with QuantSI before we moved
  any code (by tools/make_pickle_fixtures.py; don't regenerate it). They must
  still load, and give back the shared Dimension objects.
- New pickles: a pickle stores the import path of the function that rebuilds the
  object. New pickles must keep using the old paths, so that older QuantSI
  versions and Brian2 can still read them.
"""

import base64
import importlib
import json
import pathlib
import pickle

import numpy as np
import pytest

from QuantSI import DimensionMismatchError, mV, volt
from QuantSI.fundamentalunits import (
    DIMENSIONLESS,
    Quantity,
    get_or_create_dimension,
    quantity_with_dimensions,
)

FIXTURES = json.loads((pathlib.Path(__file__).parent / "data" / "pickles.json").read_text())


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_old_pickles_still_load(case):
    fixture = FIXTURES[case]
    expected = fixture["expected"]
    obj = pickle.loads(base64.b64decode(fixture["pickle"]))
    if expected["kind"] == "error":
        assert isinstance(obj, DimensionMismatchError)
        assert obj.desc == expected["desc"]
        assert [d._dims for d in obj.dims] == [tuple(d) for d in expected["dims"]]
    elif expected["kind"] == "dimension":
        assert obj is get_or_create_dimension(expected["dims"])  # still the shared object
    else:
        assert isinstance(obj, Quantity)
        assert obj.dim is get_or_create_dimension(expected["dims"])
        assert np.asarray(obj).dtype.str == expected["dtype"]
        np.testing.assert_array_equal(np.asarray(obj), expected["values"])


#: The functions that pickles refer to, and the module path they must record.
HISTORICAL_RECONSTRUCTORS = {
    quantity_with_dimensions: "QuantSI.fundamentalunits",
    get_or_create_dimension: "QuantSI.fundamentalunits",
}


@pytest.mark.parametrize(
    "obj",
    [3 * mV, np.arange(3.0) * volt, volt.dim, DIMENSIONLESS],
    ids=["scalar", "array", "dimension", "dimensionless"],
)
def test_new_pickles_name_the_historical_paths(obj):
    reconstructor = obj.__reduce__()[0]
    assert reconstructor in HISTORICAL_RECONSTRUCTORS
    module_path = HISTORICAL_RECONSTRUCTORS[reconstructor]
    # pickle saves the function's __module__ and __qualname__ ...
    assert reconstructor.__module__ == module_path
    # ... and imports it from there again when loading.
    module = importlib.import_module(module_path)
    assert getattr(module, reconstructor.__qualname__) is reconstructor
    for protocol in (2, 4, 5):
        assert pickle.loads(pickle.dumps(obj, protocol=protocol)) is not None
