# -*- coding: utf-8 -*-
""" refractive.ice module.

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

This module provides a list of ice refractive index models to compute the
dielectric properties of ice according to the requested frequency and
temeperatures.
The module can be also used as a standalone python script.

Example
-------
The python script is callable as

    $ python ice.py Temperature Frequency

and returns the complex refractive index of ice at the requested
Temperature [Kelvin] and Frequency [Hz]

Notes
-----
    It is possible to call the functions implemented in this module using
    nd-arrays. The function arguments must either have exactly the same
    shape allowing element-wise application of the functions or one of
    the two must be a scalar which will be spread across the nd computations

Temperature should be provided in Kelvin and frequency in Hz
The specific called algorithm check for arguments values to be within the
limits of validity of the dielectric model and raises ValueError in case
they are not respected

"""

from os import path

import numpy as np
import pandas as pd
from scipy import interpolate

module_path = path.split(path.abspath(__file__))[0]
warren_ice_table = pd.read_csv(
    module_path+'/IOP_2008_ASCIItable.dat',
    delim_whitespace=True, names=['wl', 'mr', 'mi']
    )
warren_ice_table['f'] = 299792.458e9 / \
    warren_ice_table.wl  # wl is microns, should return Hz
warren_ice_table = warren_ice_table.set_index('f')
warren_ice_table = warren_ice_table.iloc[::-1]  # reverse order
warren_ice_eps = (warren_ice_table.mr.values+1j*warren_ice_table.mi.values)**2
warren_ice_interpolated = interpolate.interp1d(
    warren_ice_table.index.values, warren_ice_eps, assume_sorted=True)

iwabuchi_ice_table = pd.read_csv(
    module_path+'/iwabuchi_ice_eps.dat',
    index_col=0, dtype=np.float64, comment='#'
    )
iwabuchi_ice_table.index.name = 'f'
iwabuchi_ice_interp_real = interpolate.interp2d(np.arange(
    160., 275., 10.),
iwabuchi_ice_table.index.values, iwabuchi_ice_table.values[:, 0:12]
)
iwabuchi_ice_interp_imag = interpolate.interp2d(np.arange(
    160., 275., 10.),
iwabuchi_ice_table.index.values, iwabuchi_ice_table.values[:, 12:]
)


def iwabuchi_yang_2011(temperature, frequency):
    """
    Ice complex relative dielectric constant according to Iwabuchi (2011)
    'Temperature dependence of ice optical constants: Implications for
    simulating the single-scattering properties of cold ice clouds' J. Quant.
    Spec. Rad. Tran. 112, 2520-2525

    The model is valid for temperature ranging from 160 to 270 K.
    Frequency/wavelength range of validity is [150 MHz/2 meters; 44 nanometers]

    Source of the table is the additional material published along with the
    paper

    Parameters
    ----------
    temperature : float
        nd array of temperature [Kelvin] which will be ignored
    frequency : float
        nd array of frequency [Hz]

    Returns
    -------
    nd - complex
        Relative dielectric constant of ice at the requested frequency and
        temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument
    """

    if not hasattr(frequency, '__array__'):
        frequency = np.asarray(frequency)
    if not hasattr(temperature, '__array__'):
        temperature = np.asarray(temperature)

    if (frequency < 0).any():
        raise ValueError('A negative frequency value has been passed')

    if (temperature.size == frequency.size) and (frequency.size<1):
        eps_real = iwabuchi_ice_interp_real(temperature.flatten(
        ), frequency.flatten()).diagonal().reshape(frequency.shape)
        eps_imag = iwabuchi_ice_interp_imag(temperature.flatten(
        ), frequency.flatten()).diagonal().reshape(frequency.shape)
    elif (temperature.size == 1) and (frequency.size ==1):
        eps_real = iwabuchi_ice_interp_real(temperature, frequency)
        eps_imag = iwabuchi_ice_interp_imag(temperature, frequency)
    elif temperature.size == 1:
        temps = temperature*np.ones(frequency.shape)
        eps_real = iwabuchi_ice_interp_real(
            temps.flatten(), frequency.flatten()).diagonal().reshape(
            temps.shape)
        eps_imag = iwabuchi_ice_interp_imag(
            temps.flatten(), frequency.flatten()).diagonal().reshape(
            temps.shape)
    elif frequency.size == 1:
        freqs = frequency*np.ones(temperature.shape)
        eps_real = iwabuchi_ice_interp_real(
            temperature.flatten(), freqs.flatten()).diagonal().reshape(
            freqs.shape)
        eps_imag = iwabuchi_ice_interp_imag(
            temperature.flatten(), freqs.flatten()).diagonal().reshape(
            freqs.shape)
    else:
        raise AttributeError(
            'Passed temperature and frequency are non-scalars of different'
            'shapes')
    return eps_real + 1j*eps_imag


def warren_brandt_2008(frequency):
    """Ice complex relative dielectric constant according to Warren (2008)
    'Optical constants of ice from the ultraviolet to the microwave: A
    revised compilation.' J. Geophys. Res., 113, D14220,
    doi:10.1029/2007JD009744 which updates and corrects Warren, S. G. (1984),
    'Optical constants of ice from the ultraviolet to the microwave',
    Appl. Opt., 23, 1206–1225.

    The model is valid for temperature = 266 K, thus this parameter is dropped
    Source of the tables https://atmos.washington.edu/ice_optical_constants/

    Parameters
    ----------
    frequency : float
        nd array of frequency [Hz]

    Returns
    -------
    nd - complex
        Relative dielectric constant of ice at the requested frequency and
        temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument

    """
    if (np.asarray(frequency) < 0).any():
        raise ValueError('A negative frequency value has been passed')

    return warren_ice_interpolated(frequency)


def matzler_2006(temperature, frequency, checkTemperature=True):
    """Ice complex relative dielectric constant according to Matzler (2006)
    "Thermal Microwave Radiation: application to remote sensing, Chapter 5,
    pp 456-460"

    Parameters
    ----------
    temperature : float
        nd array of temperature [Kelvin]
    frequency : float
        nd array of frequency [Hz]
    checkTemperature : bool
        check temperature range for Matzler (2006) (default True)

    Returns
    -------
    nd - complex
        Relative dielectric constant of ice at the requested frequency and
        temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument

    """

    if (np.asarray(frequency) < 0).any():
        raise ValueError(
            'refractive: A negative frequency value has been passed')
    if (temperature < 0).any():
        raise ValueError(
            'refractive: A negative temperature value has been passed')
    if (((frequency < 0.01e9) + (frequency >= 300.0e9)).any()):
        raise ValueError(
            'Matzler model for refractive index of ice is valid between 10 MHz'
            ' and 300 GHz')
    if checkTemperature and (temperature < 240.).any():
        raise ValueError(
            'Matzler model for refractive index of ice is only valid above'
            ' 240 K')

    freqs = frequency*1.0e-9

    B1 = 0.0207
    b = 335.
    B2 = 1.16e-11
#    c = 299792458.

    eps1 = 3.1884+(temperature-273)*9.1e-4
    theta = 300./temperature-1.
    alpha = (0.00504+0.0062*theta)*np.exp(-22.1*theta)
    deltabeta = np.exp(-9.963+0.0372*(temperature-273.16))
    betaM = B1*np.exp(b/temperature)/(temperature*((np.exp(b /
              temperature)-1)*(np.exp(b/temperature)-1)))+B2*freqs*freqs
    beta = betaM+deltabeta
    eps2 = alpha/freqs + beta*freqs
    return eps1 + 1j*eps2

##############################################################################
def eps(temperature, frequency, model="Matzler_2006", matzlerCheckTemperature=True):
    """Ice complex relative dielectric constant according to the requested model

    Parameters
    ----------
    temperature : float
        nd array of temperature [Kelvin]
    frequency : float
        nd array of frequency [Hz]
    model : string
        dielectric model name default to Matzler (2006)
    matzlerCheckTemperature : bool
        check temperature range for Matzler (2006) (default True)

    Returns
    -------
    nd - complex
        Relative dielectric constant of ice at the requested frequency and 
        temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument

    """

    if not hasattr(frequency, '__array__'):
        frequency = np.asarray(frequency)
    if not hasattr(temperature, '__array__'):
        temperature = np.asarray(temperature)

    if (model == 'Matzler_2006'):
        return matzler_2006(temperature, frequency,
                            checkTemperature=matzlerCheckTemperature)
    elif (model == 'Warren_2008'):
        return warren_brandt_2008(frequency)
    elif (model == 'Iwabuchi_2011'):
        return iwabuchi_yang_2011(temperature, frequency)
    else:
        print("I do not recognize the ice refractive index specification,"
              " falling back to Matzler 2006")
        return matzler_2006(temperature, frequency)


def n(temperature, frequency, model="Matzler_2006",matzlerCheckTemperature=True):
    """Ice complex refractive index according to the requested model

    Parameters
    ----------
    temperature : float
        nd array of temperature [Kelvin]
    frequency : float
        nd array of frequency [Hz]
    model : string
        dielectric model name default to Matzler (2006)
    matzlerCheckTemperature : bool
        check temperature range for Matzler (2006) (default True)

    Returns
    -------
    nd - complex
        Refractive index of ice at the requested frequency and temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument

    """
    return np.sqrt(eps(temperature, frequency, model,matzlerCheckTemperature=matzlerCheckTemperature))
##############################################################################


if __name__ == "__main__":
    import sys
    n(float(sys.argv[1]), float(sys.argv[2]))
