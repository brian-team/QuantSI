Usage comparison
================

In this section, we will compare the usage of QuantSI with another packages.
The comparation will be made with Pint and Astropy's units module.


From Pint usage documentation we have the next example:

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> distance = 24.0 * ureg.meter
    >>> distance
    <Quantity(24.0, 'meter')>
    >>> print(distance)
    24.0 meter

Astropy has a similar usage:

    >>> from astropy import units as u
    >>> distance = 24.0 * u.meter
    >>> distance
    <Quantity 24.0 m>
    >>> print(distance)
    24.0 m

We can do the same thing using QuantSI but with some modifications using the Quantity class:

    >>> from QuantSI import meter
    >>> distance = 24.0 * meter
    >>> distance
    24.0 * meter
    >>> print(distance)
    24.0 meter
    >>> type(distance)
    <class 'QuantSI.fundamentalunits.Quantity'>

Registration of new units can always be done:

For Pint:

    >>> ureg.define('furlong = 220 * yard = fur')
    >>> 1 * ureg.furlong
    <Quantity(220, 'yard')>

For Astropy:
    
    >>> u.add_enabled_units(['furlong'])
    >>> u.def_unit('furlong', 220 * u.yard)
    >>> 1 * u.furlong
    <Quantity 220.0 yard>

For QuantSI:

    >>> from QuantSI import *
    >>> from QuantSI.allunits import newton
    >>> Nm = newton*meter
    >>> (1*Nm).in_unit(Nm)
    '1. N m'


Using Numpy functions with units:

For Pint:
    
    >>> import numpy as np
    >>> arr=[1,2,3]*ureg.meter
    >>> np.sum(arr)
    <Quantity(6, 'meter')>

For Astropy:
    
    >>> import numpy as np
    >>> arr=[1,2,3]*u.meter
    >>> np.sum(arr)
    <Quantity 6.0 m>

For QuantSI:

    >>> import numpy as np
    >>> arr=[1,2,3]*meter
    >>> np.sum(arr)
    6.0 * meter

Another Numpy functions like numpy.allclose() can be used with QuantSI but users need to pay attention to assign units to the function arguments as well:

    >>> import numpy as np
    >>> arr=[1,2,3]*si.meter
    >>> np.allclose(arr,arr,atol=1*si.meter)
    True

Using Numpy trigonometric functions with units:

For Pint:

    >>> import numpy as np
    >>> np.sin(90 * ureg.degree)
    <Quantity(1.0, 'dimensionless')>
    >>> print(np.sin(90 * ureg.degree))
    1.0 dimensionless

For Astropy:
    
    >>> import numpy as np
    >>> np.sin(90 * u.degree)
    <Quantity 1.0>
    >>> print(np.sin(90 * u.degree))
    1.0


QuantSI does not have a degree unit because the package assign units to numerical values and the difference between the object created stands in the unit. 
Furthermore with this approach we only use radians so the conversion to degrees is necessary:

    >>> import numpy as np
    >>> from QuantSI import radian
    >>> degree = 90*radian*np.pi/180
    >>> np.sin(degree)
    np.float64(1.0)
    >>> print(90*radian)
    90.0

Converting from units to standard classes in python or numpy arrays:

For Pint:

    >>> distance = 24.0 * ureg.meter
    >>> distance_float = distance.magnitude
    >>> print(distance_float)
    24.0
    # To convert to a numpy array
    >>> distance_arr = [10,20,30] * ureg.meter
    >>> distnace_arr_numpy = distance_arr.magnitude
    >>> print(distance_arr_numpy)
    array([10, 20, 30])

For Astropy:

    >>> distance = 24.0 * u.meter
    >>> distance_float = distance.value
    >>> print(distance_float)
    24.0
    # To convert to a numpy array
    >>> distance_arr = [10,20,30] * u.meter
    >>> distnace_arr_numpy = distance_arr.value
    >>> print(distance_arr_numpy)
    array([10, 20, 30])

For QuantSI:
    
    >>> distance = 24.0 * si.meter
    >>> distance_float = float(distance)
    # Can convert to another types as well
    >>> print(distance_float)
    24.0
    # To convert to a numpy array
    >>> distance_arr = [10,20,30] * si.meter
    >>> distnace_arr_numpy = np.asarray(distance_arr)
    # np.array function can also be used
    >>> print(distance_arr_numpy)
    array([10, 20, 30])


From a developer perspective, QuantSI and units from Astropy have a similar approach in their code. Both packages implement their functionality by 
inheriting from numpy.ndarray in order to use numpy functions without including more wrapper functions. Instead, Pint wraps NumPy functions to ensure 
they behave correctly with units, thus providing the expected behavior. 
  