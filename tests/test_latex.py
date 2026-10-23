"""LaTeX output, without SymPy.

QuantSI used to import SymPy (about 80% of its import time) just to print the
exponent in names like ``metre ** 3``. It now writes LaTeX itself; these tests
pin down that the output is the same and that importing QuantSI no longer
imports SymPy. SymPy itself can still print quantities through the ``_latex``
printer hooks.
"""

import subprocess
import sys

import numpy as np
import pytest

from QuantSI import mV, volt
from QuantSI._core.utils import _latex_number
from QuantSI.allunits import metre, second


def test_importing_quantsi_does_not_import_sympy():
    code = (
        "import sys, QuantSI\n"
        "from QuantSI.allunits import metre, mvolt\n"
        "(3 * mvolt)._repr_latex_(); (metre ** 3)._repr_latex_(); mvolt.dim._repr_latex()\n"
        "print('sympy' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


#: Exponents as they appear in unit names, and a few awkward floats.
NUMBERS = [3, 2, -1, 0, 0.5, 1.5, -0.5, 2.0, 1 / 3, 0.1, 0.25, -2.5, 1e-5, 1e-4, 1e14, 1e15, 1e-20]
NUMBERS += [12345.678, float("inf"), float("-inf"), float("nan"), -0.0, np.float64(0.5)]


@pytest.mark.parametrize("value", NUMBERS, ids=repr)
def test_latex_number_matches_sympy(value):
    sympy = pytest.importorskip("sympy")
    assert _latex_number(value) == sympy.latex(value)


def test_numpy_integers_print_as_plain_digits():
    # SymPy prints these as \mathtt{\text{3}}; plain digits are what we want.
    assert _latex_number(np.int64(3)) == "3"


def test_unit_power_latex():
    assert (metre**3).latexname == r"\mathrm{m}^{3}"
    assert (second**-0.5).latexname == r"\mathrm{s}^{-0.5}"


def test_sympy_can_still_print_quantities():
    sympy = pytest.importorskip("sympy")
    q = 3 * mV
    assert f"${sympy.latex(q)}$" == q._repr_latex_()
    assert sympy.latex(volt) == volt._latex()
    assert sympy.latex(volt.dim) == volt.dim._latex()


def test_unit_notebook_representation():
    assert metre._repr_latex_() == r"$\mathrm{m}$"


def test_latex_rejects_arrays_with_more_than_two_axes():
    with pytest.raises(NotImplementedError, match="3-d matrix"):
        (np.ones((2, 2, 2)) * metre)._repr_latex_()
