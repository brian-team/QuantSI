"""How long ``import QuantSI`` takes in a fresh Python process."""

import subprocess
import sys


def _import_in_fresh_interpreter():
    subprocess.run([sys.executable, "-c", "import QuantSI"], check=True)


def test_import_time(benchmark):
    # Each round starts a new Python process, so just do 5 rounds instead of
    # letting pytest-benchmark work out how many runs it needs.
    benchmark.pedantic(_import_in_fresh_interpreter, rounds=5, iterations=1)
