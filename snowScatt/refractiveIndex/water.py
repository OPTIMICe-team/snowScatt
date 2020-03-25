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

This module provides a list of water refractive index models to compute the
dielectric properties of water according to the requested frequency and
temeperatures.
The module can be also used as a standalone python script.

Example
-------
The python script is callable as

    $ python water.py temperature frequency

and returns the complex refractive index of water at the requested
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

import numpy as np


def turner_kneifel_cadeddu(temperature, frequency):
    """ The "Turner-Kneifel-Cadeddu" liquid water absorption model (JTECH 2016).

        SPECIAL MODEL FOR SUPERCOOLED LIQUID WATER

    It was built using both laboratory observations (primarily at warm temperature) and 
    field data observed by MWRs at multiple frequency at supercool temperature. The field
    data were published in Kneifel et al. JAMC 2014.  The strength of the TKC model is the 
    use of an optimal estimation framework to determine the empirical coefficients of the 
    double-Debye model.  A full description of this model is given in

        Turner, D.D., S. Kneifel, and M.P. Cadeddu, 2016: An improved liquid
        water absorption model in the microwave for supercooled liquid clouds.
        J. Atmos. Oceanic Technol., 33(1), pp.33-44, doi:10.1175/JTECH-D-15-0074.1.

        Note that the model is designed to operate over the frequency range from 0.5 to 500
    GHz, and temperature from -40 degC to +50 degC; only for freshwater (no salinity)

    Parameters
    ----------
    temperature : array_like
        nd array of temperature [Kelvin]
    frequency : array_like
        nd array of frequency [Hz]

    Returns
    -------
    nd - complex
        Relative dielectric constant of ice at the requested frequency and temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument
    ValueError
        If frequency or temperature out of the limits of validity of the model
        is passed as an argument

    """
    if (frequency < 0).any():
        raise ValueError(
            'refractive: A negative frequency value has been passed')
    if (temperature < 0).any():
        raise ValueError(
            'refractive: A negative temperature value has been passed')
    if (frequency > 500.0e9).any():
        raise ValueError(
            'Ellison model for dielectric property of fresh water is only '
            'valid up to 1 THz')

    # ; Input cloud temperature (scalar float), in degC
    temp = temperature - 273.15

    # Some constants
    # cl = 299792458.0  # ;speed of light in vacuum

    # Empirical coefficients for the TKC model. The first 4 are a1, b1, c1,
    # and d1,
    # the next four are a2, b2, c2, and d2, and the last one is tc.
    coef = [8.111e+01, 4.434e-03, 1.302e-13, 6.627e+02,
            2.025e+00, 1.073e-02, 1.012e-14, 6.089e+02, 1.342e+02]

    # This helps to understand how things work below
    a_1 = coef[0]
    b_1 = coef[1]
    c_1 = coef[2]
    d_1 = coef[3]

    a_2 = coef[4]
    b_2 = coef[5]
    c_2 = coef[6]
    d_2 = coef[7]

    t_c = coef[8]

    # Compute the static dielectric permittivity (Eq 6)
    eps_s = 87.9144 - 0.404399*temp + 9.58726e-4*temp**2. - 1.32802e-6*temp**3.

    # Compute the components of the relaxation terms (Eqs 9 and 10)
    # First Debye component
    delta_1 = a_1 * np.exp(-b_1 * temp)
    tau_1 = c_1 * np.exp(d_1 / (temp + t_c))
    # Second Debye component
    delta_2 = a_2 * np.exp(-b_2 * temp)
    tau_2 = c_2 * np.exp(d_2 / (temp + t_c))

    # Compute the relaxation terms (Eq 7) for the two Debye components
    term1_p1 = (tau_1**2.*delta_1)/(1.0 + (2.0*np.pi*frequency*tau_1)**2.)
    term2_p1 = (tau_2**2.*delta_2)/(1.0 + (2.0*np.pi*frequency*tau_2)**2.)

    # Compute the real permittivitity coefficient (Eq 4)
    eps1 = eps_s - ((2.*np.pi*frequency)**2.)*(term1_p1 + term2_p1)

    # Compute the relaxation terms (Eq 8) for the two Debye components
    term1_p1 = (tau_1 * delta_1) / (1. + (2.*np.pi*frequency*tau_1)**2.)
    term2_p1 = (tau_2 * delta_2) / (1. + (2.*np.pi*frequency*tau_2)**2.)

    # Compute the imaginary permittivitity coefficient (Eq 5)
    eps2 = 2.0*np.pi*frequency*(term1_p1 + term2_p1)
    #epsilon = complex(eps1, eps2)
    return eps1 + 1j*eps2


def ellison(temperature, frequency):
    """Water complex relative dielectric constant according to Ellison (2005)
    "..." TODO: put the extensive correct reference here

    Parameters
    ----------
    temperature : array_like
        nd array of temperature [Kelvin]
    frequency : array_like
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
    ValueError
        If frequency or temperature out of the limits of validity of the model
        is passed as an argument

    """

    if (frequency < 0).any():
        raise ValueError(
            'refractive: A negative frequency value has been passed')
    if (temperature < 0).any():
        raise ValueError(
            'refractive: A negative absolute temperature value has been'
            ' passed')
    if (temperature < 273.15).any():
        raise ValueError(
            'refractive: A subfreeze temperature value has been passed'
            ' consider to use Turner Kneifel Cadeddu model')
    if (frequency > 1000.0e9).any():
        raise ValueError(
            'Ellison model for dielectric property of fresh water is only'
            ' valid up to 1 THz')

    a0 = 5.7230
    a1 = 2.2379e-2
    a2 = -7.1237e-4
    a3 = 5.0478
    a4 = -7.0315e-2
    a5 = 6.0059e-4
    a6 = 3.6143
    a7 = 2.8841e-2
    a8 = 1.3652e-1
    a9 = 1.4825e-3
    a10 = 2.4166e-4

    T = temperature-273.15
    es = (37088.6-82.168*T)/(421.854+T)
    einf = a6+a7*T
    e1 = a0+T*(a1+T*a2)  # a0+a1*T+a2*T*T
    ni1 = (45.0+T)/(a3+T*(a4+T*a5))  # (a3+a4*T+a5*T*T)
    ni2 = (45.0+T)/(a8+T*(a9+T*a10))  # (a8+a9*T+a10*T*T)
    A1 = frequency*1.0e-9/ni1
    A2 = frequency*1.0e-9/ni2
    eps1 = (es-e1)/(1+A1*A1)+(e1-einf)/(1+A2*A2)+einf
    eps2 = (es*A1-e1*A1)/(1+A1*A1)+(e1*A2-einf*A2)/(1+A2*A2)
    return eps1 + 1j*eps2


def pamtra_water(temperature, frequency):
    return ellison(temperature, frequency)
# PLACEHOLDER FOR WHAT PAMTRA IS CURRENTLY COMPUTING

##############################################################################


def eps(temperature, frequency, model="Ellison"):
    """Water complex relative dielectric constant according to the requested model

    Parameters
    ----------
    temperature : array_like
        nd array of temperature [Kelvin]
    frequency : array_like
        nd array of frequency [Hz]
    model : string
        dielectric model name default to Ellison (2005)

    Returns
    -------
    nd - complex
        Relative dielectric constant of water for the requested frequency and
        temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument

    """

    if not hasattr(temperature, '__array__'):
        temperature = np.asarray(temperature)
    if not hasattr(frequency, '__array__'):
        frequency = np.asarray(frequency)

    if (model == "Ellison"):
        return ellison(temperature, frequency)
    if (model == 'Turner'):
        return turner_kneifel_cadeddu(temperature, frequency)
    else:
        print("I do not recognize the ice refractive index specification, "
              "falling back to Ellison")
        return ellison(temperature, frequency)


def n(temperature, frequency, model="Ellison"):
    """Water complex refractive index according to the requested model

    Parameters
    ----------
    temperature : array_like
        nd array of temperature [Kelvin]
    frequency : array_like
        nd array of frequency [Hz]
    model : string
        dielectric model name default to Ellison (2005)

    Returns
    -------
    nd - complex
        Refractive index of water for the requested frequency and temperature

    Raises
    ------
    ValueError
        If a negative frequency or temperature is passed as an argument

    """
    return np.sqrt(eps(temperature, frequency, model))

##############################################################################


if __name__ == "__main__":
    import sys
    n(float(sys.argv[1]), float(sys.argv[2]))
