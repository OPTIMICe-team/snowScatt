""" refractive.utilities module

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

This module provides a short list of utilities and complementary functions
for the refractive index module.

Basic conversion from refractive index to dielectric permittivity
(and viceversa) is implemented.
The module also provides a conversion function from dielectric permittivity to
radar dielectric factor K2 which is of great importance in radar applications

"""

from __future__ import absolute_import

import numpy as np

speed_of_light = 299792458.0

def eps2n(eps): return np.sqrt(eps)

def n2eps(n): return n*n

def wavenumber(frequency=None,wavelength=None):
    if (frequency is None):
        if (wavelength is None):
            raise AttributeError('Either frequency or wavelength must be not None')
        else:
            return 2.0*np.pi/wavelength
    elif (wavelength is None):
        if (frequency is None):
            raise AttributeError('Either frequency or wavelength must be not None')
        else:
            return 2.0*np.pi*frequency/speed_of_light
    else:
        raise AttributeError('You cannot pass both frequency and wavelength')

def K(eps):
    """ Rayleigh complex dielectric factor
    This is basically the K complex factor that defines the Radar dielectric
    factor |K|**2. It is useful in Rayleigh theory to define absorption cross
    section from its imaginary part

    Parameters
    ----------
    eps : complex
        nd array of complex relative dielectric constants

    Returns
    -------
    nd - float
        Rayleigh complex dielectric factor K
    """
    return (eps-1.0)/(eps+2.0)


def K2(eps):
    """ Radar dielectric factor |K|**2

    Parameters
    ----------
    eps : complex
        nd array of complex relative dielectric constants

    Returns
    -------
    nd - float
        Radar dielectric factor |K|**2 real

    """
    K_complex = (eps-1.0)/(eps+2.0)
    return (K_complex*K_complex.conj()).real
