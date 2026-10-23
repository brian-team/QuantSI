"""Generated files must be exactly what their generators produce."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[1]
GENERATOR = ROOT / "tools" / "generate_units.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_units", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not GENERATOR.exists(), reason="generator not available (installed package)")
def test_allunits_is_up_to_date():
    generator = load_generator()
    assert generator.OUTPUT.read_text() == generator.generate(), (
        "allunits.py differs from what tools/generate_units.py generates; "
        "run `python tools/generate_units.py` and commit the result"
    )


@pytest.mark.skipif(not GENERATOR.exists(), reason="generator not available (installed package)")
def test_stdunits_is_up_to_date():
    generator = load_generator()
    assert generator.STDUNITS_OUTPUT.read_text() == generator.generate_stdunits(), (
        "stdunits.py differs from what tools/generate_units.py generates; "
        "run `python tools/generate_units.py` and commit the result"
    )


@pytest.mark.skipif(not GENERATOR.exists(), reason="generator not available (installed package)")
def test_init_unit_names_are_up_to_date():
    generator = load_generator()
    current = generator.INIT.read_text()
    assert generator.generate_init(current) == current, (
        "the unit names in __init__.py differ from what tools/generate_units.py "
        "generates; run `python tools/generate_units.py` and commit the result"
    )
