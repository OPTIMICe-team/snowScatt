# -*- coding: utf-8 -*-
""" refractive module core submodule

This module provides a list of ice and water refractive index models to compute 
the dielectric properties of ice according to the requested frequencies and
temeperatures. The module is completed with some Effective Medium approximation
functions to compute the refractive index of inhomogeneous mixtures of materials
which are directly used to compute the dielectric properties of snow as a
dilution of ice in air.

This initialization file loads handy functions implemented in the core submodule
which consistently call ice, water or snow refractive index modules

Example
-------
    $ python
    >>> import refractive
    >>> refractive.n(temperatures, frequencies, **kwargs)

and returns the complex refractive index of ice at the requested
Temperature [Kelvin] and Frequency [Hz]

Notes
-----
    It is possible to call the functions implemented in this module using
    nd-arrays. The function arguments must either have exactly the same
    shape allowing element-wise application of the functions or one of
    the two must be a scalar which will be spread across the nd computations

    Frequencies and Temperatures are always mandatory arguments as name of the
    substance, but specific algorithms requires special additional arguments to
    be passed in order to proceed (for instance snow density must be defined).
    The functions check for argument consistency and raise AttributeError if
    a wrong list of attributes is passed.

All of the argument quatities must be provided in SI units: Temperatures in 
Kelvin, frequencies in Hz, densities in kg/m3. The specific called algorithm 
check for arguments values to be within the limits of validity of the dielectric
model and raises ValueError in case they are not respected

"""

from .core import *
