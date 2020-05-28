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

import numpy as np
import logging

from .radarMoments import specific_reflectivity
from .radarMoments import dB
from snowScatt._compute import backscatVel
from snowScatt._compute import backscatter
from snowScatt._compute import _c

from snowScatt import refractiveIndex


def dopplerSpectrum(diameters, psd, wavelength, properties, 
                    ref_index=None, temperature=None,
                    mass=None, theta=0.0,
                    dopplerVel=None):
    """radar Doppler spectrum simulator

    """
    freq = _c/wavelength
    bck, vel = backscatVel(diameters, wavelength, properties,
                           ref_index, temperature, mass, theta)
    eps = refractiveIndex.water.eps(temperature, freq, 'Turner')
    K2 = refractiveIndex.utilities.K2(eps)
    z = specific_reflectivity(wavelength, bck, K2)

    spectrum = z*psd*np.gradient(diameters)/np.gradient(vel)

    logging.debug(dB(np.sum(z*psd*np.gradient(diameters), axis=-1))-
                  dB(np.sum(spectrum*np.gradient(vel), axis=-1)))

    if dopplerVel is None:
        dopplerVel = np.linspace(-10.0, 10.0, 1024)

    velidx = vel.argsort()

    dopplerVel = vel[velidx]
    return spectrum[: ,velidx], dopplerVel


def sizeSpectrum(diameters, psd, wavelength, properties, 
                    ref_index=None, temperature=None,
                    mass=None, theta=0.0):
    """radar spectrum simulator

    """
    freq = _c/wavelength
    bck = backscatter(diameters, wavelength, properties,
                      ref_index, temperature, mass, theta)
    eps = refractiveIndex.water.eps(temperature, freq, 'Turner')
    K2 = refractiveIndex.utilities.K2(eps)
    z = specific_reflectivity(wavelength, bck, K2)

    spectrum = z*psd

    return spectrum