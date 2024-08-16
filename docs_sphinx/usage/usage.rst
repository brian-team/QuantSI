Usage
===== 

In this section, we will compare the usage of unitSI with another packages.
The comparation will be made with Pint and ,,,

From Pint usage documentation we have the next example:

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> # UnitRegistry() uses the default list of units and prefixes
    >>> distance = 24.0 * ureg.meter
    >>> distance
    <Quantity(24.0, 'meter')>
    >>> print(distance)
    24.0 meter

We can do the same thing using unitSI but in a more simple way:

    >>> from unitSI import meter
    >>> distance = 24.0 * meter
    >>> distance
    24.0 * meter
    >>> print(distance)
    24.0 meter

