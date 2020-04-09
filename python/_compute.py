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

_c = 2.99792458e8
_ice_density = 917.0

def _compute_effective_size(size=None, ar=None, angle=None):
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


def _convert_to_array(x):
    return np.asarray([x]) if np.isscalar(x) else np.asarray(x)


def snow(diameters, wavelength, properties, ref_index=None, temperature=None, mass=None, theta=0.0, Nangles=181):
    """
    This is the main function of the snowScatt module. It is a python interface 
    to the snowLibrary and the low level C functions that compute the SSRGA
    scattering properties.
    It can be invoked for either single particles or an array of particles.
    The number of particles for which the properties are computed is determined 
    by the length of the argument "diameters". In case the other arguments are 
    scalars, they are used as fixed parameters for all the different diameters.
    Otherwise, if they are arrays, they must be of the same length of the 
    diameters argument; in this case each diameter will be computed with its own
    set of parameters. Some parameters need to be fixed for all computations.
    Those are the properties label, the incidence angle theta and the number of
    angular subdivisions of the 0-pi range for the phase function Nangles.

    Parameters
    ----------
    diameters : array-like or scalar double
        Snowflake diameters [meters] for which the scattering and microphysical
        properties are to compute. The number of passed diameters will define
        the number of different particles (Nparticles).
    wavelength : array-like or scalar double
        Wavelength in the vacuum [meters] of the incident electromagnetic wave.
        If an array-like is passed than it must have length=Nparticles.
    properties : string
        Label that defines the type of particle. Call snowLibrary.info() for a
        list of available snowflake properties
    ref_index : array-like or scalar complex (optional)
        Refractive index of the ice. If an array-like is passed than it must
        have length=Nparticles. If not set than the temperature attribute must
        be set
    temperature : array-like or scalar double (optional)
        Ambient temperature. Ignored if ref_index is set. Computes the complex 
        refractive index of ice for the requested wavelength according to the
        Iwabuchi et al. (2011) model. If an array-like is passed than it must
        have length=Nparticles.
    mass : array-like or scalar double (optional)
        Mass of the snowflake particles. If an array-like is passed than it must
        have length=Nparticles. If left unset the mass is calculated from the
        snowLibrary properties.
    theta : scalar double (optional default is 0.0 zenith)
        Polar incidence angle [radians] defaults to 0.0 (zenith-pointing).
        The incidence angle is used only to compute the effective size of the
        particle along the propagation direction (together with the information
        on particle aspect ratio). It ranges from 0 to pi. The code consider the
        snowflake overall shape to be spheroidal and thus there will be simmetry
        around pi/2 for this angle.
    Nangles : scalar integer (optional default is 181)
        Number of angles to calculate the phase function interval (0 to pi
        extremes always included). The default value is 181 meaning that the 
        phase function is calculated with a 1 degree resolution. The scattering
        properties are calculated by integrating the phase function so
        increasing Nangle will also increase the computation accuracy

    Returns
    -------
    Cext : array(Nparticles) - double
        Extinction cross section (meters**2)
    Cabs : array(Nparticles) - double
        Absorption cross section (meters**2)
    Csca : array(Nparticles) - double
        Total scattering cross section (meters**2)
    Cbck : array(Nparticles) - double
        Radar backscattering cross section (meters**2)
    asym : array(Nparticles) - double
        asymmetry parameter (dimensionless)
    phase : 2D array(Nparticles, Nangles) - double
        Normalized phase function. int(phase*sin(th)dth) = 1 
    mass_prop : array(Nparticles) - double
        Particle mass as assumed by the snowLibrary properties (kilograms). It
        can be used to compare how much the user defined masses dfeviates from
        the average mass of the snowflakes for which the SSRGA parameters have
        been derived 
    vel : array(Nparticles) - double
        Average terminal fallspeed (meters/second). Computed using the Boehm
        model (2005).

    Raises
    ------
    AttributeError : if neither ref_index nor temperature are defined
    """

    diameters = _convert_to_array(diameters)
    wavelength = wavelength*np.ones_like(diameters)

    if ref_index is None:
        if temperature is None:
            raise AttributeError('You have to either specify directly the refractive index or provide the temperature so that refractive index will be calculated according to Iwabuchi 2011 model\n')
        print('computing refractive index of ice ...')
        temperature = temperature*np.ones_like(diameters)
        #print(temperature.shape, (_c/wavelength).shape)
        ref_index = ice.n(temperature, _c/wavelength, model='Iwabuchi_2011')
    else:
        ref_index = ref_index*np.ones_like(diameters)
        #print(type(ref_index), ref_index.shape, ref_index)

    kappa, beta, gamma, zeta1, alpha_eff, ar_mono, mass_prop, vel = snowLibrary(diameters, properties)
    
    if mass is None:
        print('compute masses from snow properties')
        mass = mass_prop
    else:
        mass = mass*np.ones_like(diameters)
    
    Vol = mass/_ice_density
    
    K = hexPrismK(ref_index, ar_mono)

    Deff = _compute_effective_size(diameters, alpha_eff, theta) # TODO substitute with a C function that takes into account prolate particles

    Cext, Cabs, Csca, Cbck, asym, phase = ssrga(Deff, Vol, wavelength, K, kappa, gamma, beta, zeta1, Nangles)

    return Cext, Cabs, Csca, Cbck, asym, phase, mass_prop, vel