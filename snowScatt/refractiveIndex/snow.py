"""
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

This module computes snow dielectric properties as a homogeneous mixture of
ice and air or maybe even other stuff ...

The module can be also used as a standalone python script.

Example
-------
The python script is callable as

    $ python snow.py Temperature Frequency Density

and returns the complex refractive index of snow at the requested
Temperature [Kelvin], Frequency [Hz] and density [kg/m**3]

Notes
-----
    It is possible to call the functions implemented in this module using
    nd-arrays. The function arguments must either have exactly the same
    shape allowing element-wise application of the functions or one of
    the two must be a scalar which will be spread across the nd computations

Temperature should be provided in Kelvin, frequency in Hz and density in kg/m**3
The dielectric module checks for arguments values to be within the
limits of validity of the dielectric model and raises ValueError in case
they are not respected
"""

import numpy as np

from . import ice, mixing

ice_density = 916.7  # kg/m**3


def n(temperature, frequency, density, model_mix='Bruggeman', 
      model_ice='Matzler_2006', matzlerCheckTemperature=True):
    """ Effective refractive index of snow according to the specified models
     for ice dielectric properties, effective medium approximation function
     and effective density of the snowflake

    Parameters
    ----------
    temperature : float
        nd array of temperature [Kelvin]
    frequency : float
        nd array of frequency [Hz]
    density: float
        nd array of effective density [kg/m**3]
    model_mix : string
        Effective Medium Approximation model name default to Bruggeman
    model_ice : string
        dielectric model name default to Matzler (2006)
    matzlerCheckTemperature : bool
        check temperature range for Matzler (2006) (default True)

    Returns
    -------
    nd - complex
        Refractive index of snow at the requested frequency and temperature

    """
    return np.sqrt(eps(temperature, frequency, density, model_mix=model_mix,
                       model_ice=model_ice,
                       matzlerCheckTemperature=matzlerCheckTemperature))


def eps(temperature, frequency, density, model_mix='Bruggeman',
        model_ice='Matzler_2006',matzlerCheckTemperature=True):
    """ Effective complex relative dielectric constant of snow according to
    the specified models for ice dielectric properties, effective medium
    approximation function and effective density of the snowflake

    Parameters
    ----------
    temperature : float
        nd array of temperature [Kelvin]
    frequency : float
        nd array of frequency [Hz]
    density: float
        nd array of effective density [kg/m**3]
    model_mix : string
        Effective Medium Approximation model name default to Bruggeman
    model_ice : string
        dielectric model name default to Matzler (2006)
    matzlerCheckTemperature : bool
        check temperature range for Matzler (2006) (default True)

    Returns
    -------
    nd - complex
        Relative dielectric constant of snow at the requested frequency and
        temperature

    """
    if not hasattr(temperature, '__array__'):
        temperature = np.asarray(temperature)
    if not hasattr(frequency, '__array__'):
        frequency = np.asarray(frequency)
    if not hasattr(density, '__array__'):
        density = np.asarray(density)

    fraction = density/ice_density
    eps_ice = ice.eps(temperature, frequency, model=model_ice,matzlerCheckTemperature=matzlerCheckTemperature)
    eps_air = complex(1.0, 0.0)+0.0*eps_ice
    return mixing.eps([eps_ice, eps_air], [fraction, 1.0-fraction], 
        model=model_mix)

##############################################################################


if __name__ == "__main__":
    import sys
    n(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
