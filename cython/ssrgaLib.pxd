# -*- mode: python -*-
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

# This could be part of the  ssrga_module.pyx file,
# but it could be useful to have a specific namespace (import ssrgaLib)
# Here we make a wrapper to the functions we want to call from cython

# This is the vectorized ssrga function.
# Important dimensions are Nparticles and Ntheta.
# Every particles has its own set of parameters,
# except Ntheta which is set for every particle.
# This means that the variables are vectors Nparticles long.
# The phase function is Nparticles*Ntheta long.
cdef extern from "../src/ssrga.h":
    void ssrga(int Nparticles,
               double *Deff, double *Vol, double *wl, double complex *K,
               double *kappa, double *gamma, double *beta, double *zeta1,
               int Ntheta, double *Cext, double *Cabs, double *Csca,
               double *Cbck, double *asym, double *phase);

# Once I have the vectorized function this will not be necessary anymore
cdef extern from "../src/ssrga.h":
    void ssrga_single(double Deff, double Vol, double wl, double complex K,
                      double kappa, double gamma, double beta, double zeta0, int Ntheta,
                      double *Cext, double *Cabs, double *Csca, double *Cbck, double *asym, double *phase);

cdef extern from "../src/ssrga.h":
    void ssrgaBack(int Nparticles,
                     double *Deff, double *Vol, double *wl, double complex *K,
                     double *kappa, double *gamma, double *beta, double *zeta1,
                     double *Cbck);

# Once I have the vectorized function this will not be necessary anymore
cdef extern from "../src/ssrga.h":
    void back_single(double Deff, double Vol, double wl, double complex K,
                     double kappa, double gamma, double beta, double zeta0,
                     double *Cbck);

# Here potentially for debugging purposes
cdef extern from "../src/dielectric_factor.h":
    void dielectric_factor_hexagonal(double complex refind, double aspect, double complex *K);

# Vectorized calculation of effective dielectric factor for
# an aggregate of hexagonal prisms according to Chris Westbrook model.
# aspect ratio < 1 means plate-like > 1 means column/needle
cdef extern from "../src/dielectric_factor.h":
    void vectorK(int N, double complex *refind, double *aspect, double complex *K);
