"""The examples in the user documentation must keep working."""

import doctest
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[1]
USER_DOCS = ROOT / "docs_sphinx" / "user"


@pytest.mark.skipif(
    not USER_DOCS.exists(), reason="documentation not available (installed package)"
)
@pytest.mark.parametrize("page", ["text.rst"])
def test_documentation_examples(page):
    result = doctest.testfile(
        str(USER_DOCS / page), module_relative=False, optionflags=doctest.ELLIPSIS
    )
    assert result.failed == 0


@pytest.mark.skipif(not (ROOT / "README.md").exists(), reason="README not available")
def test_readme_examples():
    result = doctest.testfile(
        str(ROOT / "README.md"),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL,
    )
    assert result.failed == 0
