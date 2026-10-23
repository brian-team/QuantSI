Unit names
==========

Which names can I import?
-------------------------

Every unit has a full name made of an SI prefix and the unit's name:
``mvolt``, ``kohm``, ``usiemens``, ``cmetre2``. ``QuantSI.allunits`` contains
all of them: every unit with every SI prefix, and the squares and cubes
(``umetre2``, ``mmetre3``).

From the package itself, ``from QuantSI import ...`` gives:

* every unit, without prefix and with the prefixes ``p n u m k M G T`` (and
  ``c`` for ``metre``/``meter``), for example ``newton``, ``knewton``, ``mvolt``,
  ``Gohm``. Units that are never prefixed (``kilogram``, ``kelvin``) come
  without prefix only;
* every short name.

Short names
-----------

Short names are written like the unit's symbol: ``<prefix><symbol>``. They exist
for a fixed list of units and prefixes (``QuantSI.stdunits``):

==============  ==============  ==============================================
unit            symbol          prefixes
==============  ==============  ==============================================
volt            ``V``           ``m u n p`` (``mV``, ``uV``, ``nV``, ``pV``)
amp             ``A``           ``m u n p``
farad           ``F``           ``m u n p``
siemens         ``S``           ``m u n p``
second          ``s``           ``m u n p`` (``ms``, ``us``, ``ns``, ``ps``)
molar           ``M``           ``m u n p``
hertz           ``Hz``          none, ``k M G`` (``Hz``, ``kHz``, ``MHz``, ``GHz``)
metre           ``m``           ``c m u``, and squares and cubes (``cm2``, ``um3``)
==============  ==============  ==============================================

So if ``mV`` exists, so do ``uV``, ``nV`` and ``pV``. ``u`` stands for micro.

No single letters
-----------------

There are deliberately no names of a single letter (no ``V``, ``A``, ``s``, ``m``
or ``M``), so that importing QuantSI's names never replaces variables such as
``V`` or ``m`` in your own code. Write ``volt`` or ``metre`` instead.

These rules are applied by ``tools/generate_units.py``, which generates the
modules, and checked by the test suite. Names are only ever added: a name that
can be imported today will stay importable.
