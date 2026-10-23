How the code is organised
=========================

QuantSI is a NumPy array subclass (``Quantity``) that carries a physical
dimension, plus the code that keeps that dimension right through calculations.
The code is split into small private modules with one job each, and a few public
modules re-export what users need, under names that don't change.

Public modules
--------------

Users import from these, and names never disappear from them
(``tests/test_public_api.py`` checks them against ``tests/data/public_api.json``):

``QuantSI``
    The curated top-level namespace: the classes, the functions and the
    commonly used units.
``QuantSI.fundamentalunits``
    The historical home of the core classes. It now only re-exports them, but it
    stays forever: pickles store the paths ``QuantSI.fundamentalunits.quantity_with_dimensions``
    and ``QuantSI.fundamentalunits.get_or_create_dimension``, and Brian2's
    ``brian2.units.fundamentalunits`` is a facade over it.
``QuantSI.allunits``, ``QuantSI.stdunits``, ``QuantSI.constants``
    The unit and constant catalogue.

Private modules
---------------

All of QuantSI's internals live in one folder, ``src/QuantSI/_core/``. The
underscore in the folder's name marks everything inside as private, so the files
themselves don't need one (NumPy organises its internals the same way, in
``numpy._core``). Each file's docstring ends with a short list of what's in it.

They are listed from the bottom up: at the top of a file, a module may only
import the modules listed *above* it (``tests/test_architecture.py`` checks
this). When a module needs one listed below it, for example to build an error
message, it imports it inside the function that needs it, so the import only
happens on that rare path.

``_core.utils``
    Small helpers (``set_module``, list flattening, short array strings).
``_core.errors``
    The exception classes.
``_core.dimension``
    ``Dimension``, the intern cache and ``DIMENSIONLESS``, plus the functions
    that get and compare the dimensions of arbitrary objects.
``_core.registry``
    The three unit registries used to pick a display unit.
``_core.ufuncs``
    The rule for every NumPy ufunc (``RULES``) and one handler per rule.
``_core.array_functions``
    How every other NumPy function treats quantities (``__array_function__``):
    the four buckets and QuantSI's implementations.
``_core.quantity``
    The ``Quantity`` class.
``_core.unit``
    The ``Unit`` class.
``_core.formatting``
    Turning quantities into text: ``format_quantity`` (``str``, ``repr``,
    ``in_unit``, ``in_best_unit``) and ``format_quantity_latex``. Explicit numeric
    format specifications use NumPy's formatting of the raw SI value.
``_core.parsing``
    Reading quantities from text (``parse_quantity``) without ``eval``.
``_core.decorators``
    ``check_units``.
``_core.wrappers``
    The ``wrap_function_*`` helpers that ``unitsafefunctions`` is built from.

``QuantSI.unitsafefunctions`` (public, from Brian2) holds unit-aware versions of
``sin``, ``exp``, ``arange``, ... with the metadata Brian2's code generation
reads. The tests live in ``tests/`` at the top of the repository, outside the package, so
they aren't installed with QuantSI; ``conftest.py`` next to them sets up the tests
and the doctests. ``tools/`` holds the code generator for the unit modules
(``generate_units.py``), the scripts that write the test fixtures, and the
Brian2 integration used in CI (``tools/brian2``).

Rules that keep the package correct
-----------------------------------

* **One shared object.** There is one ``Dimension`` object per set of exponents,
  and one copy of each registry, for the whole process. Don't copy them, and don't
  call ``Dimension(...)`` directly: use ``get_or_create_dimension``. A second copy
  would break every ``dim1 is dim2`` check.
* **Re-export explicitly.** Compatibility namespaces use ``from ._x import name as name``
  so that linters and type checkers know the import is on purpose.
* **Pickle paths are public API.** Functions that pickles refer to keep their
  old ``__module__`` through ``_core.utils.set_module``. ``tests/test_pickle_compat.py``
  checks this with pickles written before the code moved.
* **Metadata names are an interface.** ``check_units`` sets ``_arg_units``,
  ``_return_unit`` and a few more attributes on the functions it wraps, and
  Brian2's code generation reads them.
* **Types.** QuantSI ships ``py.typed``. Its functions and classes are annotated
  and checked with mypy in CI, together with ``tests/typing_examples.py``, which
  states the types users see. NumPy's type stubs say that arithmetic on any
  array gives a plain ``ndarray``, so ``Quantity`` and ``Unit`` declare their
  operators again inside ``if TYPE_CHECKING:``, which only type checkers read:
  ``3 * mV`` is a ``Quantity`` and ``volt * amp`` a ``Unit``. NumPy declares its own
  subclass ``np.matrix`` the same way, and since nothing runs, it costs nothing at
  run time. The types are approximate in two places: a result whose dimensions
  cancel is a plain number at run time, and pyright types ``array * unit`` as a
  plain array.
