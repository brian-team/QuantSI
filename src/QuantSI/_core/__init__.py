"""QuantSI's internals.

Nothing in this folder is public: import from ``QuantSI``,
``QuantSI.fundamentalunits`` and the other public modules instead, which re-export
what users need. The modules here are layered, lowest first; at the top of a file
a module only imports the ones listed above it (tests/test_architecture.py checks
this):

- ``utils``: small helpers (``set_module``, ``numpy_docstring``, ``_latex_number``)
- ``errors``: the exceptions and ``QuantSIWarning``
- ``dimension``: the ``Dimension`` class and the helpers that read dimensions
- ``registry``: the unit registries that choose the display unit
- ``ufuncs``: the rule for every NumPy ufunc
- ``array_functions``: the rule for every other NumPy function
- ``quantity``: the ``Quantity`` class
- ``unit``: the ``Unit`` class
- ``formatting``: quantity to text
- ``parsing``: text to quantity, without ``eval``
- ``decorators``: ``check_units``
- ``wrappers``: the ``wrap_function_*`` helpers behind ``unitsafefunctions``
"""
