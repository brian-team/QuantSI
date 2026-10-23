"""Rewrite tests/data/display_golden.txt from the current display code.

Only run this when you change the output on purpose, and check the diff.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # the tests are not part of the installed package

from tests.display_cases import render  # noqa: E402

OUTPUT = ROOT / "tests/data/display_golden.txt"

if __name__ == "__main__":
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT}")
