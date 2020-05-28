#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2020 Davide Ori 
University of Cologne

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import numpy as np

from snowScatt import refractiveIndex
from snowScatt._compute import _c
from snowScatt._compute import backscatter

def dB(x):
    """
    Converts from linear units to dB

    Parameters:
    -----------
    x - ndarray - double (strictly positive >0)
        array of values to be converted into dB
    Returns:
    --------
    dB(x) - ndarray - double
        x array converted to dB
    """
    return 10.0*np.log10(x)


def Bd(x):
    """
    Converts from dB to linear units

    Parameters:
    -----------
    x - ndarray - double
        array of values to be converted into linear units
    Returns:
    --------
    Bd(x) - ndarray - double
        x array converted to linear units
    """
    return 10.0**(0.1*x)


def specific_reflectivity(wl, bck, K2):
    """
    Compute radar reflectivity
    """
    return 1.0e18*wl**4*bck/(K2*np.pi**5)


def Ze(diameters, psd, wavelength, properties, 
       ref_index=None, temperature=None,
       mass=None, theta=0.0, bck=None, K2=None):
    """radar reflectivity 

    """
    freq = _c/wavelength
    if bck is None: # compute only if not precalculated
        bck = backscatter(diameters, wavelength, properties,
                          ref_index, temperature, mass, theta)
    if K2 is None: # compute only if not precalculated
        eps = refractiveIndex.water.eps(temperature, freq, 'Turner')
        K2 = refractiveIndex.utilities.K2(eps)
    z = specific_reflectivity(wavelength, bck, K2)
    Z = dB(np.sum(z*psd*np.gradient(diameters), axis=-1))

    return Z


def calcMoments(x, y, n):
    raise NotImplementedError('Sorry, this is not ready yet')
    return None