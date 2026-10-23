"""Write the pickle fixtures for test_pickle_compat.py.

This was run once, with QuantSI as it was before any code moved. Don't run it
again: the fixtures are only useful because they come from the old code. To
cover more objects, add entries to the JSON file instead of rewriting it.
"""

import base64
import json
import pathlib
import pickle

import numpy as np

from QuantSI import DimensionMismatchError, ms, mV, nA, volt
from QuantSI.allunits import metre, mvolt
from QuantSI.fundamentalunits import DIMENSIONLESS

OUTPUT = pathlib.Path(__file__).parent.parent / "tests/data/pickles.json"

CASES = {
    "scalar_quantity": 3 * mV,
    "array_quantity": np.array([1.0, 2.0, 3.0]) * volt,
    "array_2d_quantity": np.arange(6.0).reshape(2, 3) * nA,
    "float32_quantity": np.array([1.0, 2.0], dtype=np.float32) * metre,
    "integer_times_unit": np.arange(3) * ms,
    "unit": mvolt,
    "dimension": volt.dim,
    "dimensionless": DIMENSIONLESS,
    "error": DimensionMismatchError("Addition", volt.dim, nA.dim),
}


def describe(obj):
    if isinstance(obj, DimensionMismatchError):
        return {"kind": "error", "desc": obj.desc, "dims": [list(d._dims) for d in obj.dims]}
    if hasattr(obj, "_dims"):  # a Dimension
        return {"kind": "dimension", "dims": list(obj._dims)}
    arr = np.asarray(obj)
    return {
        "kind": "quantity",
        "dims": list(obj.dim._dims),
        "values": arr.tolist(),
        "dtype": arr.dtype.str,
    }


def main():
    fixtures = {}
    for name, obj in CASES.items():
        for protocol in (2, 4, 5):
            data = pickle.dumps(obj, protocol=protocol)
            fixtures[f"{name}-p{protocol}"] = {
                "pickle": base64.b64encode(data).decode("ascii"),
                "expected": describe(obj),
            }
    OUTPUT.write_text(json.dumps(fixtures, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(fixtures)} fixtures to {OUTPUT}")


if __name__ == "__main__":
    main()
