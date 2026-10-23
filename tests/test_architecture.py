"""The private modules in QuantSI/_core are layered: top-level imports only point down.

At the top of a file, a module in _core may only import the modules listed *before*
it in ``LAYERS``. Importing a module from a higher layer is fine inside a function
(on a rare path, for example to build an error message), and keeps the imports free
of cycles. Private modules never import the public modules that re-export them, and
all private code lives in _core. See docs_sphinx/developer/architecture.rst.
"""

import ast
import pathlib

import pytest

import QuantSI

PACKAGE = pathlib.Path(QuantSI.__file__).parent
CORE = PACKAGE / "_core"

#: Bottom-up. Add new private modules here, at the lowest layer that works.
LAYERS = [
    "utils",
    "errors",
    "dimension",
    "registry",
    "ufuncs",
    "array_functions",
    "quantity",
    "unit",
    "formatting",
    "parsing",
    "decorators",
    "wrappers",
]


def module_level_imports(path):
    """QuantSI modules imported by top-level statements of ``path`` (a module in _core)."""
    imported = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            # "from .x import y" names x; "from . import x" names x
            imported.update([node.module] if node.module else [a.name for a in node.names])
        elif isinstance(node, ast.ImportFrom) and node.level > 1:
            # "from .. import allunits": a public module
            imported.update(f"QuantSI.{node.module or a.name}" for a in node.names)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("QuantSI"):
            imported.add(node.module.removeprefix("QuantSI._core."))
    return imported


def test_every_private_module_has_a_layer():
    private = {p.stem for p in CORE.glob("*.py")} - {"__init__"}
    assert private == set(LAYERS)


def test_private_code_lives_in_core():
    # _version.py is written by the build, from the git history.
    assert {p.stem for p in PACKAGE.glob("_*.py")} == {"__init__", "_version"}


@pytest.mark.parametrize("module", LAYERS)
def test_imports_only_point_down(module):
    allowed = set(LAYERS[: LAYERS.index(module)])
    imported = module_level_imports(CORE / f"{module}.py")
    assert imported <= allowed, f"{module} imports {sorted(imported - allowed)} at module level"
