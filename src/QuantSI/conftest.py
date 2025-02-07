import pytest

def pytest_collection_modifyitems(config, items):
    # List of function names whose doctests should be skipped
    functions_to_skip = {
        "QuantSI.fundamentalunits.Quantity.fill",
        "QuantSI.fundamentalunits.Quantity.trace",
    }

    for item in items:
        function_name = item.location[2]  # The third element often contains the name in doctests
        # Skip specific functions' doctests
        if any(fn in function_name for fn in functions_to_skip):
            item.add_marker(pytest.mark.skip(reason="Skipping doctest for specific function due to known documentation issues"))

