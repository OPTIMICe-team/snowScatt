# -*- coding: utf-8 -*-
""" refractive module core submodule

    Copyright (C) 2017 - 2020 Davide Ori dori@uni-koeln.de
    Institute for Geophysics and Meteorology - University of Cologne

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

This module provides a list of ice and water refractive index models to
compute the dielectric properties of ice according to the requested
frequencies and temeperatures. The module is completed with some Effective
Medium approximation functions to compute the refractive index of
inhomogeneous mixtures of materials which are directly used to compute the
dielectric properties of snow as a dilution of ice in air.

This core file loads submodules and provide handy functions to
consistently call ice, water or snow refractive index modules

Example
-------
    $ python
    >>> import refractiveIndex
    >>> refractiveIndex.n(temperatures, frequencies, **kwargs)

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
check for arguments values to be within the limits of validity of the
dielectric model and raises ValueError in case they are not respected

"""

import numpy as np

from . import ice, snow, utilities, water

substances_list = ['ice', 'water', 'snow']


# model=None, model_ice=None, model_mix=None, densities=None):
def n(substance, temperatures, frequencies, **kwargs):
    """Complex index of refraction of the requested substance according to the
        requested specifications

    Parameters
    ----------
    temperatures : float
        nd array of temperatures [Kelvin]
    frequencies : float
        nd array of frequencies [Hz]
    **kwargs : additional arguments to be passed to the requested model

    Returns
    -------
    nd - complex
        Refractive index of the requested substance using the requested options

    Raises
    ------
    AttributeError
        If an uncorrect list of arguments is passed

    """
    return np.sqrt(eps(substance, temperatures, frequencies, **kwargs))


def eps(substance, temperatures, frequencies, **kwargs):
    """Complex relative dielectric permittivity of the requested substance
        according to the requested specifications

    Parameters
    ----------
    temperatures : float
        nd array of temperatures [Kelvin]
    frequencies : float
        nd array of frequencies [Hz]
    **kwargs : additional arguments to be passed to the requested model

    Returns
    -------
    nd - complex
        Refractive index of the requested substance using the requested options

    Raises
    ------
    AttributeError
        If an uncorrect list of arguments is passed

    """

    if (substance == 'ice'):
        return ice.eps(temperatures, frequencies, **kwargs) 
    elif (substance == 'water'):
        return water.eps(temperatures, frequencies, **kwargs) 
    elif (substance == 'snow'):
        return snow.eps(temperatures, frequencies, **kwargs)
    else:
        raise AttributeError("I do not recognize the %s as a " 
                             "valid substance I can only compute"
                             " dielectric properties of %s" % (
                                substance, substances_list))

################################################################################

def mk(frequency=None, wavelength=None, 
       refractive_index=None, substance=None, **kwargs):
    """ Interaction depth
        Inverse of a distance [meters]
        It is the first part of the |m|kd criterion for the DDA validity
    """
    m = refractive_index
    if (m is None):
        m = n(substance=substance,frequencies=frequency,**kwargs)
    return np.abs(m)*utilities.wavenumber(frequency,wavelength)
  
def skin_depth(frequency=None, wavelength=None, 
               refractive_index=None, substance=None, **kwargs):
    """ Skin depth in the material
        Distance [meters] that takes to the electric field to change due to the
        presence of the dielectric material according to Draine [1988]
    """
    m = refractive_index
    if (refractive_index is None):
        m = n(substance=substance,frequencies=frequency,**kwargs)
    
    if (wavelength is not None):
        return wavelength/(m.imag*2.0*np.pi)
    if (frequency is not None):
        wl=utilities.speed_of_light/frequency
        return wl/(m.imag*2.0*np.pi)

def magnetic2electric_ratio(size=None, frequency=None, wavelength=None, 
                            refractive_index=None, substance=None, **kwargs):
    """ Ratio between the absorption cross section due to magnetic dipoles and
        absorption due to electric dipoles. It must be small for the validity of
        the DDA algorithm which does not consider magnetic moments.
        Approximate formulation according to Draine and Lee [1984]
    """
    if (size is None):
        raise AttributeError('You must provide the size of the dipole')

    m = refractive_index
    if (m is None):
        m = n(substance=substance,frequencies=frequency,**kwargs)
    eps = utilities.n2eps(m)

    k = utilities.wavenumber(frequency, wavelength)
    return (k*size)**2.0 * ((eps.real)**2.0 + eps.imag**2.0)/90.0
