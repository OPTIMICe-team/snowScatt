"""
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

Implement a few effective medium approximation formulas...

Both material dielectric models and EMAs usually work with dielec permittivity
and refractive index is derived as sqrt(eps), hence this modules is implemented
to always take eps as arguments of both the eps and n functions for
computational efficiency

This module works using just unitless quantities. Which is amazing by itself
but also solves a lot of implementation problems :)

The module can be also used as a standalone python script.

Example
-------
The python script is callable as

    $ python mixing.py refractive_indices volume_fractions

and returns the complex refractive index of the mixture

Notes
-----
    It is possible to call the functions implemented in this module using
    nd-arrays. The function arguments must either have exactly the same
    shape allowing element-wise application of the functions or one of
    the two must be a scalar which will be spread across the nd computations

Refractive indexes should be complex values.

"""

import numpy as np


def maxwell_garnett(eps, mix):
    """Maxwell-Garnett EMA for the refractive index.
       AKA: Rayleigh mixing formula [Sihvola, 1989]

       The MG relation for effective medium approximation is derived for the
       case of small, highly diluted, spatially well separated, optically soft
       and spherical inclusions in a medium. Also the size of the considered
       volume should be small.

       The MG relation is asymmetric, the order of medium and inclusion in the
       formula counts. This is a consequence of the aformentioned assumptions.

       J.C.M. Garnett, Philos. Trans. R. Soc. 203, 385 (1904); 205, 237 (1906).

    Parameters
    ----------
        eps: Tuple of complex
            dielectric permittivity of the media
        mix: Tuple of float
            the volume fractions of the media

    Returns:
    ----------
        The Maxwell-Garnett approximation for the complex refractive index of
        the effective medium

    The first element of the eps and mix tuples is taken as the matrix and the
    second as the inclusion.
    """

    cF = mix[1] / (mix[0]+mix[1]) * (eps[1]-eps[0]) / (eps[1]+2*eps[0])
    return eps[0]*(1.0+2.0*cF) / (1.0-cF)


def bruggeman(eps, mix):
    """Bruggeman EMA for the refractive index.

    'D.A.G. Bruggeman, Ann. Phys. 24, 636 (1935)'

    Parameters
    ----------
        eps: Tuple of complex
            dielectric permittivity of the media
        mix: Tuple of float
            the volume fractions of the media, len(mix)==len(m)
            (if sum(mix)!=1, these are taken relative to sum(mix))

    Returns:
    ----------
        The Bruggeman approximation for the complex refractive index of
        the effective medium

    The first element of the eps and mix tuples is taken as the matrix and the
    second as the inclusion.

    Bruggeman model has the advantage with respect to MG of beeing symmetric
    """

    f1 = mix[0]/np.sum(mix)
    f2 = mix[1]/np.sum(mix)
    e1 = eps[0]
    e2 = eps[1]
    a = -2*(f1+f2)
    b = (2*f1*e1 - f1*e2 + 2*f2*e2 - f2*e1)
    c = (f1+f2)*e1*e2
    return (-b - np.sqrt(b**2-4*a*c))/(2*a)


def sihvola(eps, mix, ni=0.85):
    """Sihvola EMA for the refractive index.

    For instructions, see mg_refractive in this module, except this routine
    only works for two components.

    The original formulation is defaulted to ni=0.85 which has been found to be
    the best for many snow applications. Also, the analitic solution for Sihvola
    modified EMA is way too complicated to be written and computed efficiently:
    a numerically converging solution is applied instead.

    A.H. Sihvola 'Self-Consistency Aspects of Dielectric Mixing Theories'
    IEEE Trans. Geos. Rem. Sens. vol 27, n 4, 1989

    Parameters
    ----------
        eps: Tuple of complex
            dielectric permittivity of the media
        mix: Tuple of float
            the volume fractions of the media

    Returns:
    ----------
        The Sihvola approximation for the complex refractive index of
        the effective medium

    The first element of the eps and mix tuples is taken as the matrix and the
    second as the inclusion.

    Sihvola model with default ni=0.85 is found to give best results in terms
    of computed scattering properties of snow [Petty 20...] and it is symmetric
    with respect to the order of inclusions and matrix

    WARNING: Routine copied from original pamtra fortran code, not sure it is
    correct!

    """
#    raise NotImplementedError

    b = eps[1].real + 2.0*eps[0] - 2.0*ni*eps[0] - \
        mix[1]*(eps[1].real - eps[0])*(1.0 + ni)

    a = eps[0]*(eps[1].real + (2.0 - ni)*eps[0] +
                mix[1]*(2.0 - ni)*(eps[1] - eps[0]).real)

    em_real = (np.sqrt(b**2+4.0*ni*a) - b)/(2.0*ni)

    em_imag = eps[1].imag*(-1.0*(em_real - eps[0]) + mix[1]*(em_real + 2.0*eps[0] + ni*(em_real - eps[0]))) / (
        (eps[1].real + 2.0*eps[0] + 2.0*ni*(em_real - eps[0])) - mix[1]*(1.0 + ni)*(eps[1].real - eps[0]))

    return em_real + (1j * em_imag)


def sihvola_paper(eps, mix, ni=0.85):
    """Sihvola EMA for the refractive index.

    At the moment this routine only works for two components. Further
    developments are needed to make it work for more components

    The original formulation is defaulted to ni=0.85 which has been found to be
    the best for many snow applications. Also, the analitic solution for Sihvola
    modified EMA is way too complicated to be written and computed efficiently:
    a numerically converging solution is applied instead.

    A.H. Sihvola 'Self-Consistency Aspects of Dielectric Mixing Theories'
    IEEE Trans. Geos. Rem. Sens. vol 27, n 4, 1989

    Parameters
    ----------
        eps: Tuple of complex
            dielectric permittivity of the media
        mix: Tuple of float
            the volume fractions of the media

    Returns:
    ----------
        The Sihvola approximation for the complex refractive index of
        the effective medium

    The first element of the eps and mix tuples is taken as the matrix and the
    second as the inclusion.

    Sihvola model with default ni=0.85 is found to give best results in terms
    of computed scattering properties of snow [Petty 20...] and it is symmetric
    with respect to the order of inclusions and matrix

    WARNING: Routine copied from original pamtra fortran code, not sure it
    is correct!

    From eq 4.XX in Sihvola paper
    (eeff-e0)/(eeff+2e0+v(eeff-e0))=f(e1-e0)/(e1+2e0+v(eeff-e0))

    Rearrange terms
    (eeff-e0)*(e1+2e0+v(eeff-e0)) = f(e1-e0)*(eeff+2e0+v(eeff-e0))

    Resolve brakets and unfold the unknow eeff
    eeff*e1 + eeff*2e0 + eeff**2*v - eeff*v*e0 - e0e1 -2*e0**2 - v*e0*eeff + v*e0**2 = 
        = f*e1*eeff + 2*f*e1*e0 + f*e1*v*eeff - f*e1*v*e0 - f*e0*eeff - 2*f*e0**2 - f*e0*v*eeff + f*v*e0**2


    Solve the 2nd order equation in eeff
    a*eeff**2 + b*eeff + c = 0
    with the following parameters

    a = v
    b = e1 +2e0 -v*e0 -v*e0 -f*e1 -f*e1*v +f*e0 -f*e0*v
    c = -e0e1 -2*e0*e1 +v*e0**2 -2*f*e1*e0 +f*e1*v*e0 +2*f*e0**2 -f*v*e0**2

    """

    v = ni
    e1 = eps[1]
    e0 = eps[0]
    f = mix[0] #or [1]??

    # a = v
    # b = -e1 + 2.0*e0 - 2.0*v*e0 + f*e1 + f*e1*v - f*e0 - f*e0*v
    # c = -e0*e1 - 2.0*e0**2.0 + v*e0**2 - 2.0*f*e1 * \
    #     e0 + f*e1*v*e0 + 2.0*f*e0**2.0 - f*v*e0**2.0
    raise NotImplementedError

    return 0.5*(-e0*f*v - e0*f + 2.0*e0*v - 2.0*e0 + e1*f*v + e1*f - e1 -
        np.sqrt(e0**2*f**2*v**2 + 2.0*e0**2*f**2*v + e0**2*f**2 -
        8.0*e0**2*f*v + 4.0*e0**2*f + 4.0*e0**2 - 2.0*e0*e1*f**2*v**2 -
        4.0*e0*e1*f**2*v - 2.0*e0*e1*f**2 + 10.0*e0*e1*f*v - 2.0*e0*e1*f +
        4.0*e0*e1 + e1**2*f**2*v**2 + 2.0*e1**2*f**2*v + e1**2*f**2 -
        2.0*e1**2*f*v - 2.0*e1**2*f + e1**2))/v, 0.5*(-e0*f*v - e0*f +
        2.0*e0*v - 2.0*e0 + e1*f*v + e1*f - e1 + np.sqrt(e0**2*f**2*v**2 +
        2.0*e0**2*f**2*v + e0**2*f**2 - 8.0*e0**2*f*v + 4.0*e0**2*f +
        4.0*e0**2 - 2.0*e0*e1*f**2*v**2 - 4.0*e0*e1*f**2*v - 2.0*e0*e1*f**2 +
        10.0*e0*e1*f*v - 2.0*e0*e1*f + 4.0*e0*e1 + e1**2*f**2*v**2 +
        2.0*e1**2*f**2*v + e1**2*f**2 - 2.0*e1**2*f*v - 2.0*e1**2*f + e1**2))/v


##############################################################################


def n(refractive_indices, volume_fractions, model='Bruggeman', ni=0.85):
    if model == 'Bruggeman':
        return np.sqrt(bruggeman(refractive_indices**2, volume_fractions))
    elif model == 'Sihvola':
        return np.sqrt(sihvola(refractive_indices**2, volume_fractions, ni=ni))
    elif model == 'Maxwell_Garnett':
        return np.sqrt(
            maxwell_garnett(refractive_indices**2, volume_fractions))
    else:
        print('Unknown model "%s", fallback to Bruggeman' % (model))
        return np.sqrt(bruggeman(refractive_indices**2, volume_fractions))


def eps(dielectric_permittivity, volume_fractions, model='Bruggeman', ni=0.85):
    if model == 'Bruggeman':
        return bruggeman(dielectric_permittivity, volume_fractions)
    elif model == 'Sihvola':
        return sihvola(dielectric_permittivity, volume_fractions, ni=ni)
    elif model == 'Maxwell_Garnett':
        return maxwell_garnett(dielectric_permittivity, volume_fractions)
    else:
        print('Unknown model "%s", fallback to Bruggeman' % (model))
        return bruggeman(dielectric_permittivity, volume_fractions)

##############################################################################


if __name__ == "__main__":
    import sys
    n(float(sys.argv[1]), float(sys.argv[2]))
