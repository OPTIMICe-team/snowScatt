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
import pandas as pd
from scipy import interp
from scipy.integrate import quad

from os import path

from snowScatt.ssrgalib import ssrga
from snowScatt.ssrgalib import hexPrismK
from snowScatt.snowProperties import snowLibrary
from snowScatt.refractiveIndex import ice

# Library of Self-Similar Rayleigh-Gans parameters
# TODO: this could be an object that provides a more descriptive information
# about the underlying models
ssrg_lib = {}
# Parameters derived for bullet-rosettes Hogan and Westbrook (2014)
HW14 = {'kappa': 0.19, 'beta': 0.23, 'gamma': 5./3., 'zeta1': 1.}
ssrg_lib['HW14'] = HW14
# Mean parameters derived for Leinonen-Szyrmer 2015 unrimed snow aggregates
L15_0 = {'kappa': 0.189177, 'beta': 3.06939,
         'gamma': 2.53192, 'zeta1': 0.0709529}
ssrg_lib['LS15A0.0'] = L15_0
# Mean parameters for Ori et al. 2014 unrimed assemblages of ice columns
Oea14 = {'kappa': 0.190031, 'beta': 0.030681461,
         'gamma': 1.3002167, 'zeta1': 0.29466184}
ssrg_lib['Oea14'] = Oea14

# Size dependent parameters for Leinonen and Szyrmer (2015) rimed aggregates
# model A (simultaneous aggregation and riming), also ELWP dependent.
# WARNING: Overrides aspect_ratio

module_path = path.split(path.abspath(__file__))[0]
leinonen_table = pd.read_csv(module_path + '/snowProperties/jussi_table.dat',
                             delim_whitespace=True)


def leinonen_coeff(D, elwp=0.0):
    table = leinonen_table[leinonen_table.ELWP == elwp].set_index('D')
    beta = interp(D, table.index.values, table.beta_z)
    gamma = interp(D, table.index.values, table.gamma_z)
    kappa = interp(D, table.index.values, table.kappa_z)
    zeta1 = interp(D, table.index.values, table.zeta1_z)
    alpha_eff = interp(D, table.index.values, table.alpha_eff)
    return beta, -gamma, kappa, zeta1, alpha_eff


ssrg_lib_idx = list(ssrg_lib.keys()) + ['leinonen_table']

# Deff = compute_effective_size(diameter,
#                               aspect_ratio,
#                               theta_inc)

# xe = wavenumber*Deff  # effective size parameter
# # First part of Eq. 1 in Hogan et al. (2017) (except phi(x))
# prefactor = 9.*wavenumber**4.*K2*volume**2./(4.*np.pi)
# phi_ssrg = shape_factor(xe, kappa, beta, gamma, zeta1, scatt_angle)
# # Here I am not sure of the term S2. For comparison with Rayleigh
# # formula it should be the option with D/2, but since ssrg works with
# # non-spherical particles it is probably the formulation with scatterer
# # volume by comparison with other scattering quantities in ssrg
# # NOTE: This should be not azimuthally averaged!!!
# #self.S2 = self.wavenumber**2*self.K*(self.diameter*0.5)**3*np.sqrt(phi_ssrg)
# S1 = 3.*wavenumber**2*K*volume*np.sqrt(phi_ssrg)/(4.*np.pi)
# S2 = S1*np.cos(scatt_angle)
# S34 = 0.0 + 0.0j


# self.estimate_amplitude_matrix(S1, S2, S34, Ra, Rb)
# # so far in our convention the imaginary part of dielectric properties is
# # positive for absorbing materials, thus you don't find -K.imag
# self.Cabs = 3.*self.wavenumber*self.volume*self.K.imag
# #self.Cext = 2.*self.wavelength*self.S2.imag
# #self.Csca = self.Cext - self.Cabs
# self.Csca = self.scattering_xsect()
# self.Cext = self.Csca + self.Cabs
# # We have to recompute phase function and thus the inner shape factor
# # because this is computed for exact backscattering and the current
# # configuration might have a different scattering angle
# self.Cbck = phase_function(self.prefactor, self.xe, self.kappa,
#                            self.beta, self.gamma, self.zeta1,
#                            theta=np.pi)
        
def _set_ssrg_par(self, par):
    """
    Convenience function to set the srrg parameters attributes according to
    the requested model
    Parameters
    ----------
    par : dict or str
        explicit which set of parameters to use (dict) or the parameter
        model (str) to pick from the library or filename
    """
    if isinstance(par, dict):
        try:
            kappa = par['kappa']
            beta = par['beta']
            gamma = par['gamma']
            zeta1 = par['zeta1']
        except:
            raise AttributeError('Invalid ssrg parameters')
    elif isinstance(par, str):
        if par[:14] == 'leinonen_table':
            elwp = float(par.split('_')[-1])
            r = leinonen_coeff(D, elwp)
            kappa = r[0]
            beta = r[1]
            gamma = r[2]
            zeta1 = r[3]
            aspect_ratio = r[4]
        else:
            try:
                kappa = ssrg_lib[par]['kappa']
                beta = ssrg_lib[par]['beta']
                gamma = ssrg_lib[par]['gamma']
                zeta1 = ssrg_lib[par]['zeta1']
            except:
                raise AttributeError('Invalid ssrg parameters')

def scattering_xsect():
    """
    Calculates the scattering cross section by integrating over the all
    the whole 4pi solid scattering angle
    Note that the scattering phase function is already azimuthally
    averaged, so we need to integrate only over theta [0, pi]
    """
    def diff_xsect(theta):
        """
        Differential scattering cross section multiplied by sin(theta)
        Parameters
        ----------
        theta : scalar-double
            scattering angle (to be integrated upon dcos(th))
        Returns
        -------
        sin(theta)dCsca/dtheta : scalar-double
            value of the integrand over dtheta
        """
        return np.sin(theta)*phase_function(self.prefactor, self.xe,
                                            self.kappa, self.beta,
                                            self.gamma, self.zeta1,
                                            theta)

        xsect = quad(diff_xsect, 0.0, np.pi)

        return 0.5*xsect[0]


def phase_function(prefactor, x, kappa, beta, gamma, zeta1, theta):
    """
    Compute the phase function for the current state of the scatterer
    """
    phi_ssrg = shape_factor(x, kappa, beta, gamma, zeta1, theta)
    return prefactor*phi_ssrg*(1.+np.cos(theta)**2)*0.5


def shape_factor(x, kappa, beta, gamma, zeta1, theta):
    """
    Compute the shape factor for the current state of the scatterer
    """

    xang = x*np.sin(theta*0.5)
    shape_factor = first(xang, kappa) + summation(xang, beta, gamma, zeta1)
    return np.pi**2*0.25*shape_factor


def first(x, kappa):
    """
    Compute the first term in the braces in Eq. 4 of Hogan (2017)
    scattering by the mean ice distribution in the particles' population.
    """
    scale_term = np.cos(x)*((1.+kappa/3.)*(1./(2.*x+np.pi)-1./(2.*x-np.pi)) -
                            kappa*(1./(2.*x+3.*np.pi)-1./(2.*x-3.*np.pi)))
    return scale_term**2.


def summation(x, beta, gamma, zeta1):
    """
    Provide the summation component of the phi shape function for ssrga.
    the second term in the braces in Eq. 4 of Hogan (2017)
    related to scattering by the modulation of ice distribution with respect
    to the mean.
    """

    # Compute a "good" stopping point
    jmax = int(5.*x/np.pi + 1.)

    # Compute first term separately for inclusion of zeta1
    summ = zeta1*2.**(-1.*gamma)*((0.5/(x+np.pi))**2.+(0.5/(x-np.pi))**2.)

    # Compute the rest
    for j in range(2, jmax):
        term_a = (2.*j)**(-1.*gamma)
        term_b = (0.5/(x+np.pi*j))**2.+(0.5/(x-np.pi*j))**2.
        summ = summ + term_a*term_b

    return summ*beta*np.sin(x)**2.


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


################## OLD CODE I STILL NEED #######################################

def leinonen_coeff(D, elwp):
    table = leinonen_table[leinonen_table.ELWP == elwp].set_index('D')
    # print(table.columns)
    beta = interp(D, table.index.values, table.beta_z)
    gamma = interp(D, table.index.values, table.gamma_z)
    kappa = interp(D, table.index.values, table.kappa_z)
    zeta1 = interp(D, table.index.values, table.zeta1_z)
    alpha_eff = interp(D, table.index.values, table.alpha_eff)
    return beta, gamma, kappa, zeta1, alpha_eff


c = 2.99792458e8

# INPUT
diameters = np.linspace(0.001, 0.025, 50)


def brandes(D): return 7.9e-5*D**2.1


def smalles(D): return 4.1e-5*D**2.5

# SI units
ice_density = 917.0
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

    print(Cext.shape, phase.shape)

    return Cext, Cabs, Csca, Cbck, asym, phase, mass_prop, vel


def backscattering(frequency, diameters, n, table=None, ELWP=None, mass=None, aspect_ratio=1.0, elevation=np.pi*0.5):
    wavelength = c/frequency
    if mass is None:
        mass = min(brandes(diameters*1.0e3), smalles((diameters*1.0e3)))
    volume = mass/917.

    eps = n*n
    K = (eps-1.0)/(eps+2.0)
    K2 = (K*K.conjugate()).real

    # CONSTANTS FOR MY PARTICLES
    kappa = 0.190031
    beta = 0.030681461
    gamma = 1.3002167
    zeta1 = 0.29466184

    if table == 'leinonen':
        #kappa = 0.189177
        #beta = 3.06939
        #gamma = 2.53192
        #zeta1 = 0.0709529
        beta, gamma, kappa, zeta1, alpha_eff = leinonen_coeff(diameters, ELWP)
        #diameters = diameters*alpha_eff
        diameters = compute_effective_size(diameters, alpha_eff, elevation)
        gamma = -gamma
        K2 = 0.21
        # print(kappa, beta, gamma, zeta1)
    else:
        diameters = compute_effective_size(diameters, aspect_ratio, elevation)

    k = 2.0*np.pi/wavelength  # wavenumber
    x = k*diameters           # size parameters

    # from Eq. 1 and Eq. 4 (all elements except PHI_SSRGA(x)) in Hogan (2016)
    prefactor = 9.0*np.pi * k**4. * K2 * volume**2. / 16.0

    term1 = np.cos(x)*((1.0+kappa/3.0)*(1.0/(2.0*x+np.pi)-1.0/(2.0*x-np.pi)) -
                       kappa*(1.0/(2.0*x+3.0*np.pi) - 1.0/(2.0*x-3.0*np.pi)))
    term1 = term1**2.

    # Initialize scattering variables
    c_bsc = 0.0  # backscattering cross section [m2]
    c_abs = 0.0  # absorption cross section [m2]
    c_sca = 0.0  # scattering cross section [m2]

    # define scattering angles for phase function
    theta = (np.arange(301.0)) * (180./300.)
    theta_rad = theta * np.pi/180.

    n_theta = len(theta)
    d_theta_rad = theta_rad[1] - theta_rad[0]

    ph_func = np.ndarray(n_theta)

    # ABSORPTION:
    Kxyz = K  # TODO: here I assume polarizability does not change
    c_abs = 3.*k*volume*Kxyz.imag  # Hogan et al., 2016, Eq. 8
    # print(k,volume,Kxyz,Kxyz.imag)
    # BACKSCATTERING:
    # Initialize the summation in the second term in the braces of Eq. 12
    thesum = 0.

    # Decide how many terms are needed
    jmax = int(5.*x/np.pi + 1.0)

    # Evaluate summation
    for j in range(jmax):
        if j == 0:
            zeta = zeta1
        else:
            zeta = 1.
        jj = j + 1.0
        term_zeta = zeta*(2.0*jj)**(-1.0*gamma)*np.sin(x)**2.
        term_x = ((1.0/(2.0*x+2.0*np.pi*jj)**2.) +
                  (1.0/(2.0*x-2.0*np.pi*jj)**2.))
        increment = term_zeta*term_x
        thesum = thesum + increment
    # Put the terms together
    c_bsc = prefactor*(term1 + beta*thesum)
    # print(mass, c_bsc)

    # SCATTERING PHASE FUNCTION AND SCATTERING CROSS SECTION
    for i_th in range(n_theta):
        new_x = x*np.sin(theta_rad[i_th]*0.5)
        # First term in the braces of Eq. 4 representing the mean structure
        new_term1 = np.cos(new_x)*((1.0+kappa/3.0)*(1.0/(2.0*new_x+np.pi) -
                                                    1.0/(2.0*new_x-np.pi)) -
                                   kappa*(1.0/(2.0*new_x+3.0*np.pi) -
                                          1.0/(2.0*new_x-3.0*np.pi)))
        new_term1 = new_term1**2.

        # Initialize the summation in the second term in the braces of Eq. 12
        new_sum = 0.
        # Decide how many terms are needed
        jmax = int(5.*new_x/np.pi + 1.0)

        # Evaluate summation
        for j in range(jmax):
            if j == 0:
                zeta = zeta1
            else:
                zeta = 1.
            term_a = zeta*(2.0*(j+1.0))**(-1.*gamma)*np.sin(new_x)**2.
            term_b = 1.0/(2.0*new_x + 2.0*np.pi*(j+1.0))**2.
            term_c = 1.0/(2.0*new_x - 2.0*np.pi*(j+1.0))**2.
            increment = term_a*(term_b+term_c)
            new_sum = new_sum + increment

        cos2th = (np.cos(theta_rad[i_th]))**2.
        ph_func[i_th] = prefactor*((1. + cos2th)/2.)*(new_term1 + beta*new_sum)

        c_sca = c_sca + (0.5*ph_func[i_th]*np.sin(theta_rad[i_th])*d_theta_rad)

    # normalize the phase function with c_sca
    ph_func = ph_func/(2.*c_sca)  # norm 2*c_sca is convention used by Janni

    # calculate asymmetry parameter
    asym = 0.
    for i_th in range(n_theta):
        sinth = np.sin(theta_rad[i_th])
        costh = np.cos(theta_rad[i_th])
        asym = asym + ph_func[i_th]*sinth*costh*d_theta_rad

    # check if integral over normalized phase function is indeed one
# ph_func_int = 0.
# for i_th=0, n_theta-1 do begin
#   ph_func_int = ph_func_int +  (ph_func[i_th] * sin(theta_rad[i_th])) * d_theta_rad
# endfor
# print, 'int(ph_func): ', ph_func_int
# if verbose eq 1 then print, 'Scattering phase function(theta): ', ph_func
# if verbose eq 1 then print, 'Backscattering Coeff (m2): ', c_bsc
# if verbose eq 1 then print, 'Scattering Coeff (m2): ', c_sca
# if verbose eq 1 then print, 'Absorption Coeff (m2): ', c_abs
    return [c_bsc, c_abs, c_sca, asym, mass]