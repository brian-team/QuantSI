import builtins

import pytest

from brian2units.utils.stringtools import SpellChecker

@pytest.mark.codegen_independent
def test_spell_check():
    checker = SpellChecker(["vm", "alpha", "beta"])
    assert checker.suggest("Vm") == {"vm"}
    assert checker.suggest("alphas") == {"alpha"}
    assert checker.suggest("bta") == {"beta"}
    assert checker.suggest("gamma") == set()


if __name__ == "__main__":
    test_environment()
    test_spell_check()
