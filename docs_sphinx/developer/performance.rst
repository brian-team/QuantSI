Performance
===========

QuantSI adds a fixed cost to every operation (checking and combining
dimensions in Python) on top of NumPy's own work. For large arrays that cost is
negligible; for scalars and small arrays it dominates, so that is what the
benchmarks measure.

Benchmarks
----------

``tests/benchmarks`` contains pytest-benchmark benchmarks of the
operations every user pays for. Normal test runs execute each of them once, so
they keep working; to measure::

    pytest tests/benchmarks --benchmark-enable --benchmark-only

Absolute timings on CI machines vary too much to be compared between runs, so
CI checks *ratios* instead: ``test_overhead_ratio.py`` (run with
``pytest -m perf_guard``) times each operation and its plain-NumPy counterpart
back to back and fails if QuantSI's overhead grows past a limit. When a change
makes QuantSI faster, lower the limit in the same commit.

What changed, and why
---------------------

Minimum times on one machine (Apple M-series, Python 3.13, NumPy 2.5), before
and after the work that introduced this page:

====================================  =========  =========  =======
operation                             before     after      speed-up
====================================  =========  =========  =======
``import QuantSI`` (new interpreter)  178 ms     55 ms      3.3x
``A + B`` (1000 values)               1.87 us    1.08 us    1.7x
``A * R`` (1000 values)               2.12 us    1.21 us    1.8x
``3 * mvolt``                         3.88 us    1.79 us    2.2x
``x + y`` (scalars)                   1.71 us    0.92 us    1.9x
``A[10]``                             0.50 us    0.28 us    1.8x
``str(3 * mV)``                       37 us      18 us      2.1x
``dim1 == dim2``                      7.9 us     0.28 us    28x
``dim1 * dim2``                       0.37 us    0.12 us    3x
call of a ``check_units`` function    6.1 us     3.6 us     1.7x
====================================  =========  =========  =======

Where the time went:

* **Importing SymPy** (about 80% of the import time) to print unit exponents in
  LaTeX. QuantSI now writes LaTeX itself.
* **Dispatching ufuncs** by scanning lists of names, concatenating some of them
  and formatting error messages on every call. Rules are now looked up by ufunc
  object, one handler per rule, and messages are only built on failure.
* **Validating results** that NumPy had just computed from valid quantities.
* **Recomputing** dimension products, the candidate display units, and
  ``check_units``' argument names on every call; these are now worked out once.
* **Dimension equality** called ``np.allclose`` on two 7-tuples.

Each of these changes is tested against the old implementation (equality, unit
choice, LaTeX output), to check that being faster doesn't change the result.
