#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Copyright (C) 2020 Davide Ori 
#University of Cologne

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


import logging
import numpy as np

from snowScatt import refractiveIndex
from snowScatt._compute import _c
from snowScatt._compute import backscatter
from snowScatt._compute import _convert_to_array

def dB(x):
	"""
	Converts from linear units to dB

	Parameters
	----------
	x - ndarray - double (strictly positive >0)
		array of values to be converted into dB

	Returns
	-------
	dB(x) - ndarray - double
		x array converted to dB
	"""
	return 10.0*np.log10(x)


def Bd(x):
	"""
	Converts from dB to linear units

	Parameters
	----------
	x - ndarray - double
		array of values to be converted into linear units

	Returns
	-------
	Bd(x) - ndarray - double
		x array converted to linear units
	"""
	return 10.0**(0.1*x)


def specific_reflectivity(wl, bck, K2):
	"""
	Compute radar specific reflectivity

	Parameters
	----------
	wl : scalar - double
		electromagnetic wavelength [meters]
	bck : array(Nparticles) - double
		spectrum of radar backscattering cross section [meters2]
	K2 : scalar - double
		Rayleigh dielectric factor K**2 (dimensionless)
		K = (n2 - 1)/(n2 + 2) for the Clausius Mossotti relation

	Returns
	-------
	eta : ndarray(Nparticles) - double
		specific reflectivity in linear units (millimeter6 meters-3)
	"""
	return 1.0e18*wl**4*bck/(K2*np.pi**5)


def Ze(diameters, psd, wavelength, properties, 
	   ref_index=None, temperature=None,
	   mass=None, theta=0.0, bck=None, K2=None):
	"""radar reflectivity
	Compute radar reflectivity directly from hydrometeor parameters

	Parameters
	----------
	diameters : array(Nparticles) - double
			spectrum of diameters of the particles [meters]
	psd : callable
		size distribution of the particle 
		concentration [meters^-1 meters^-3]
	wavelength : scalar - double
		electromagnetic wavelength to be passed to the snowScatt 
		properties calculator
	properties : string
		name of the snowflake properties to call from the snowLibrary
	ref_index : scalar - complex (default to None)
		complex refractive index of ice to be passed to the snowScatt 
		properties calculator
	temperature : scalar - double
		absolute temperature, alternative formulation of ref_index when 
		ref_index is None to be passed to the snowScatt properties 
		calculator
	mass : array(Nparticles) - double (default to None)
		mass of the snowflakes to be passed to the snowScatt properties 
		calculator if left None the mass is derived by the snowLibrary 
		properties
	theta : scalar - double - (default to 0.0 vertical pointing)
		zenith incident angle of the electromagnetic radiation, to be passed
		to the snowScatt properties calculator
	bck : array(Nparticles) - double (default to None)
		radar backscattering cross-section [meters**2] override calculation of
		bck using particle parameters
	K2 : scalar - double 
		Rayleigh dielectric factor K^2 (dimensionless)
		K = (n^2 - 1)/(n^2 + 2) for the Clausius Mossotti relation
		override calculation of K2 from dielectric properties (useful for
		multirequency radar cross-calibration)

	Returns
	-------
	Z : scalar - double
		Radar reflectivity in logaritmic units [dBZ]

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


def calcMoments(spectrum, vel, n=4):
	""" calculate moments of spectrum(vel) up to order n.
	The maximum order implemented is 4 (kurtosis). The moments available are:
	0: reflectivity
	1: mean Doppler velocity
	2: spectrum width
	3: skweness
	4: kurtosis

	Parameters
	----------
	spectrum : array(Nvel) - double
		specific reflectivity spectrum in linear units
		(millimeter6 meters-3) !!! be careful this is already multiplied by dV
		so that sum(spectrum)=Z
	vel : array(Nvel) - double
		vector of velocities upon which spectrum is defined
	n : scalar - integer
		maximum order of moments to calculate
		for any moment n>=1 the 0...n-1 moments have to be calculated anyway
		so the computational burden does not increase
	Returns
	-------
	moments : array(n) - double
		array of Doppler radar moments ordered from 0 to n

	Raises
	------
	AttributeError : if the order n is larger than the maximum 4


	"""
	spec = _convert_to_array(spectrum)
	v = _convert_to_array(vel)
	#if len() # maybe check if the two vectors have equal sizes
	if n>4:
		raise AttributeError('Only moments up to 4 (kurtosis) are supported')

	Z = sum(spec)
	MDV = sum(spec*v)/Z
	res = v-MDV
	diff = res*res # **2
	SW = np.sqrt(sum(spec*diff)/Z)
	diff *= res # **3
	norm = SW*SW*SW # **3
	Sk = sum(spec*diff/(Z*norm)) # I guess from this moment on we can potentially iterate
	diff *= res # **4
	norm *= SW # **4
	k = sum(spec*diff/(Z*norm))
	
	moments = np.array([Z, MDV, SW, Sk, k])

	return moments[:n]