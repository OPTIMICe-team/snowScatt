#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2020 Davide Ori 
# University of Cologne

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import logging

from .radarMoments import specific_reflectivity
from .radarMoments import dB
from snowScatt._compute import backscatVel
from snowScatt._compute import backscatter
from snowScatt._compute import _c

from snowScatt import refractiveIndex

def rescaleSpectrum(spectrum, dopplerVel, dopplerGrid):
	""" Rescale uneven spectrum centered over dopplerVel onto a (possibly)
		even dopplerGrid

	This function implements a basic interpolator of the spectrum defined over
	the dopplerVel points into the dopplerGrid points.
	It ensures that the integral (reflectivity) is conserved

	Parameters
	----------
	spectrum : array(Nparticles) - double
		the Doppler spectrum defined over the (uneven) grid of velocities
	dopplerVel : array(Nparticles) - double
		the (uneven) grid of velocities
	dopplerGrid : array(Nbins) - double
		the output grid of velocities (possibly equally spaced)

	Returns
	-------
	rescaledSpectrum : array(Nbins) - double
		the spectrum rescaled on the new grid
	"""
	raise NotImplementedError('I am still thinking about how to do this the best way')

def dopplerSpectrum(diameters, psd, wavelength, properties, 
					ref_index=None, temperature=None,
					mass=None, theta=0.0,
					dopplerVel=None):
	""" radar Doppler spectrum simulator

	Simulates the Doppler spectrum of the reflectivity given the particle 
	properties and the PSD

	Parameters
	----------
		diameters : array(Nparticles) - double
			spectrum of diameters of the particles [meters]
		psd : callable
			size distribution of the particle 
			concentration [meters**-1 meters**-3]
		wavelength : scalar - double
			electromagnetic wavelength to be passed to the snowScatt 
			properties calculator
		ref_index : scalar - complex (default to None)
			complex refractive index of ice to be passed to the snowScatt 
			properties calculator
		temperature : scalar - double
			absolute temperature, alternative formulation of ref_index when 
			ref_index is None to be passed to the snowScatt properties 
			calculator
		mass : array(Nparticles) - double
			mass of the snowflakes to be passed to the snowScatt properties 
			calculator if left None the mass is derived by the snowLibrary 
			properties
		theta : scalar - double
			zenith incident angle of the electromagnetic radiation, to be passed
			to the snowScatt properties calculator
		dopplerVel : array(Nparticles) - double
			override spectrum of Doppler velocities
			if left None the Doppler velocities are calculated from the particle
			properties. 

	Returns
	-------
		spectrum : array(Nparticles) - double
			specific radar reflectivity size spectrum [mm**6 m**-3 (m/s)**-1]
		dopplerVel : array(Nparticle) - double
			Doppler velocities computed from the spectrum of diamters and sorted
			in ascending order (not rescaled to a equi-spaced grid)

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
	""" radar spectrum simulator
	Simulates the size spectrum of the reflectivity given the particle 
	properties and the PSD

	Parameters
	----------
		diameters : array(Nparticles) - double
			spectrum of diameters of the particles [meters]
		psd : callable
			size distribution of the particle 
			concentration [meters**-1 meters**-3]
		wavelength : scalar - double
			electromagnetic wavelength to be passed to the snowScatt 
			properties calculator
		ref_index : scalar - complex (default to None)
			complex refractive index of ice to be passed to the snowScatt 
			properties calculator
		temperature : scalar - double
			absolute temperature, alternative formulation of ref_index when 
			ref_index is None to be passed to the snowScatt properties 
			calculator
		mass : array(Nparticles) - double
			mass of the snowflakes to be passed to the snowScatt properties 
			calculator if left None the mass is derived by the snowLibrary 
			properties
		theta : scalar - double
			zenith incident angle of the electromagnetic radiation, to be passed
			to the snowScatt properties calculator

	Returns
	-------
		spectrum : array(Nparticles) - double
			specific radar reflectivity size spectrum [mm**6 m**-3 m**-1]

	"""
	freq = _c/wavelength
	bck = backscatter(diameters, wavelength, properties,
					  ref_index, temperature, mass, theta)
	eps = refractiveIndex.water.eps(temperature, freq, 'Turner')
	K2 = refractiveIndex.utilities.K2(eps)
	z = specific_reflectivity(wavelength, bck, K2)

	spectrum = z*psd

	return spectrum