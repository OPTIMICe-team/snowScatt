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

from snowScatt.ssrgalib import ssrga
from snowScatt.ssrgalib import hexPrismK
from snowScatt.snowProperties import snowLibrary
from snowScatt.refractiveIndex import ice

c = 2.99792458e8
ice_density = 917.0

def compute_effective_size(size=None, ar=None, angle=None):
    """
    Returns the effective size of a spheroid along the propagation direction
    TODO: Depending on how the propagation direction is defined also
    orientation of the scatterer matters
    Parameters
    ----------
    size : scalar-double
        The horizontal size of the spheroid
    ar : scalar-double
        The aspect ratio of the spheroid (vertical/horizontal)
    angle : scalar-double
        The propagation direction along which the effective size has to be
        computed
    Returns
    -------
    size_eff : scalar-double
        The effective size of the spheroid along the zenith direction defined
        by angle
    """

    #size_eff = (0.5*size*ar)**2/(ar**2*np.cos(angle)**2+np.sin(angle)**2)

    #return 2.*np.sqrt(size_eff)
    return size/np.sqrt(np.cos(angle)**2+(np.sin(angle)/ar)**2)

def snow(diameters, wavelength, properties, ref_index=None, temperature=None, mass=None, theta=0.0, Nangles=180):
    wavelength = wavelength*np.ones_like(diameters)

    if ref_index is None:
        if temperature is None:
            raise AttributeError('You have to either specify directly the refractive index or provide the temperature so that refractive index will be calculated according to Iwabuchi 2011 model\n')
        print('computing refractive index of ice ...')
        ref_index = ice.n(temperature, c/wavelength, model='Iwabuchi_2011')*np.ones_like(diameters)
    else:
        ref_index = ref_index*np.ones_like(diameters)

    kappa, beta, gamma, zeta1, alpha_eff, ar_mono, mass_prop, vel = snowLibrary(diameters, properties)
    
    if mass is None:
        print('compute masses from snow properties')
        mass = mass_prop
    
    Vol = mass/ice_density
    
    K = hexPrismK(ref_index, ar_mono)

    Deff = compute_effective_size(diameters, alpha_eff, theta) # TODO substitute with a C function that takes into account prolate particles

    Cext, Cabs, Csca, Cbck, asym, phase = ssrga(Deff, Vol, wavelength, K, kappa, gamma, beta, zeta1, Nangles)

    return Cext, Cabs, Csca, Cbck, asym, phase, mass_prop, vel