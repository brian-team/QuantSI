# Running Brian2 on this QuantSI

Brian2's `brian2.units` is to become a thin facade over QuantSI. The CI workflow
`.github/workflows/brian2.yaml` checks out Brian2's main branch and

1. turns `brian2/units/*.py` into facades over QuantSI (`make_facade.py`);
2. applies `brian2-companion.patch`, the changes Brian2 needs alongside:
   * `spatialneuron/morphology.py` multiplied the result of `np.hstack` by
     `meter` because `np.hstack` used to drop units; it no longer does;
   * `test_switching_off_unit_checks` tested `fundamentalunits.unit_checking`,
     which QuantSI does not have;
3. runs Brian2's test suite (no code generation, and the numpy target).

To do the same locally, from a Brian2 checkout with this QuantSI installed:

    python path/to/QuantSI/tools/brian2/make_facade.py .
    git apply path/to/QuantSI/tools/brian2/brian2-companion.patch
    python -c "import brian2; brian2.test('numpy', test_codegen_independent=True)"
