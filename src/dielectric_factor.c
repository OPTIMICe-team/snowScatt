/*
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
*/

#include "dielectric_factor.h"

/* Compute the real and imagnary parts of the dielectric factor for a
   sphere using the Clausius-Mossotti formula. */
void dielectric_factor_sphere(double complex refind, double complex *K) { 
  double complex epsilon = refind*refind;
  *K = (epsilon-1.0) / (epsilon+2.0);
}


/* Compute the real and imaginary parts of the effective dielectric
   factor for a randomly oriented spheroidal monomers using Gans
   theory, where aspect<1 indicates oblate spheroids and aspect>1
   indicates prolate spheroids. */
void dielectric_factor_spheroid(double complex refind, double aspect, double complex *K) {
  static const double one_third = 1.0 / 3.0;
  static const double two_thirds = 2.0 / 3.0;

  double complex epsilon = refind*refind;
  double l_z;

  if (aspect <= 1.0) {    /* Oblate spheroid */
    double ecc = sqrt(1.0 - aspect*aspect);
    l_z = (1.0-aspect*asin(ecc)/ecc)/(ecc*ecc);
  }
  else {    /* Prolate spheroid */
    double ecc = sqrt(1.0 - 1.0/(aspect*aspect));
    l_z = (1.0/(ecc*ecc) - 1.0) * ((1.0 / (2.0*ecc))*log((1.0+ecc)/(1.0-ecc)) - 1.0);
  }

  double l_x = 0.5*(1.0 - l_z);
  /* Polarizability along the principal axes, removing the volume
     term; implicitly k_y=k_x */
  double complex k_z = (epsilon-1.0) / (3.0*l_z*(epsilon-1.0) + 3.0);
  double complex k_x = (epsilon-1.0) / (3.0*l_x*(epsilon-1.0) + 3.0);
  *K = csqrt(two_thirds * k_x * k_x + one_third * k_z * k_z);
}


/* Compute the real and imaginary parts of the effective dielectric
   factor for a randomly oriented hexagonal monomers using Westbrook
   (2012) theory, where aspect<1 indicates a hexagonal plate and
   aspect>1 indicates a hexagonal column. */
void dielectric_factor_hexagonal(double complex refind, double aspect, double complex *K) {
  static const double one_third = 1.0 / 3.0;
  static const double two_thirds = 2.0 / 3.0;

  double complex epsilon = refind*refind;
  double aspect_power = 0.5 * pow(aspect, -0.9);
  /* Westbrook (QJ 2014) equations 12 and 13 */
  double l_z = 0.5 * ((1.0-3.0*aspect)/(1.0+3.0*aspect) + 1.0);
  double l_x = 0.25* ((1.0-aspect_power)/(1.0+aspect_power) + 1.0);
  /* Polarizability along the principal axes, removing the volume
     term; implicitly k_y=k_x */
  double complex k_z = (epsilon-1.0) / (3.0*l_z*(epsilon-1.0) + 3.0);
  double complex k_x = (epsilon-1.0) / (3.0*l_x*(epsilon-1.0) + 3.0);
  *K = csqrt(two_thirds * k_x * k_x + one_third * k_z * k_z);
}

void vectorK(int N, double complex *refind, double *aspect, double complex *K) {
  for (int i=0; i<N; i++) {
    dielectric_factor_hexagonal(refind[i], aspect[i], K+i);
  }
}