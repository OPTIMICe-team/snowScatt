/*
This code is based on ssrga.c available at http://www.met.reading.ac.uk/clouds/ssrga/
Author: Robin Hogan <r.j.hogan@ecmwf.int>
Copyright (C) 2016  European Centre for Medium Range Weather Forecasts 
Language: C99

Modifications made to improve accessibility from external modules  
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

#include "ssrga.h"

static inline double prefactor(double wavenumber, double complex K,
                               double Vol) {
    static const double constant = 9.0 * M_PI / 16.0;
    double wn2 = wavenumber*wavenumber;
    double K2 = creal(K)*creal(K)+cimag(K)*cimag(K);
    return constant*wn2*wn2*Vol*Vol*K2;
}

static inline double mean_term(double x, double kappa) {
    double term = cos(x)*( (1.0+kappa/3.0)*
                               ( 1.0/(2.0*x+M_PI) - 1.0/(2.0*x-M_PI) )
                           -kappa*
                               ( 1.0/(2.0*x+3.0*M_PI) - 1.0/(2.0*x-3.0*M_PI))
                         );
    return term*term;
}

static inline double sum_term(double x,
                              double beta, double gamma, double zeta1) {
    double term1 = 0.5 / (x + M_PI);
    double term2 = 0.5 / (x - M_PI);
    double sum = zeta1*pow(2.0, -gamma)*(term1*term1 + term2*term2);

    int jmax = floor(5.0*x/M_PI + 1.0);
    //printf("max j %i \n",jmax);
    for (int j = 2; j <= jmax; j++) {
        term1 = 0.5/(x + j*M_PI);
        term2 = 0.5/(x - j*M_PI);
        sum += pow(2.0*j, -gamma)*(term1*term1 + term2*term2);
    }
    double sin_x = sin(x);
    return beta * sin_x * sin_x * sum;
}

// SSRGA function for a vector of particles
void ssrga(int Nparticles,
           double *Deff, double *Vol, double *wl, double complex *K,
           double *kappa, double *gamma, double *beta, double *zeta1,
           int Ntheta, double *Cext, double *Cabs, double *Csca,
           double *Cbck, double *asym, double *phase) {

    for (int i=0; i<Nparticles; i++) {
        ssrga_single(Deff[i], Vol[i], wl[i], K[i],
                     kappa[i], gamma[i], beta[i], zeta1[i],
                     Ntheta, Cext+i, Cabs+i, Csca+i,  // here we do some magic
                     Cbck+i, asym+i, phase+i*Ntheta); // C pointer arithmetic
    }
}

// SSRGA function for a single particle
void ssrga_single(double Deff, double Vol, double wl, double complex K,
                  double kappa, double gamma, double beta, double zeta1,
                  int Ntheta, double *Cext, double *Cabs, double *Csca,
                  double *Cbck, double *asym, double *phase) {

    double wavenumber = 2.0*M_PI/wl;
    double prefact = prefactor(wavenumber, K, Vol);
    double x = wavenumber*Deff;
    
    *Cabs = 3.0 * Vol * wavenumber * fabs(cimag(K));
    double dTheta = M_PI / (Ntheta - 1.0);
    int i_theta;
    double sum = 0.0;
    double sum_asym = 0.0;

    for (i_theta=Ntheta-1; i_theta>=0; i_theta--) {
        double theta = i_theta*dTheta;
        double cos_th = cos(theta);
        double x_eff = x*sin(0.5*theta);
        phase[i_theta] = mean_term(x_eff, kappa) + 
                         sum_term(x_eff, beta, gamma, zeta1);
        phase[i_theta] *= (1.0+cos_th*cos_th)*0.5;

        double differential = phase[i_theta]*sin(theta)*dTheta;
        sum += differential;
        sum_asym += differential*cos_th;
    }

    *asym = sum_asym/sum;
    *Cbck = prefact*phase[Ntheta-1];
    *Csca = 0.5*prefact*sum;
    *Cext = *Csca+*Cabs;

    double phase_norm = 1.0/sum;
    for (i_theta = 0; i_theta < Ntheta; i_theta++) {
        phase[i_theta] *= phase_norm;
    }
}

// SSRGA compute only backscattering for a vector of particles
void ssrgaBack(int Nparticles, double *Deff, double *Vol,
               double *wl, double complex *K,
               double *kappa, double *gamma, double *beta, double *zeta1,
               double *Cbck) {
    for (int i=0; i<Nparticles; i++) {
        back_single(Deff[i], Vol[i], wl[i], K[i],
                    kappa[i], gamma[i], beta[i], zeta1[i],
                    Cbck+i); // C pointer arithmetic
    }
}

// SSRGA backscattering for a single particle 
void back_single(double Deff, double Vol, double wl, double complex K,
                 double kappa, double gamma, double beta, double zeta1,
                 double *Cbck) {
    double wavenumber = 2.0*M_PI/wl;
    double prefact = prefactor(wavenumber, K, Vol);
    double x = wavenumber*Deff;
    *Cbck = prefact*(mean_term(x, kappa) + sum_term(x, beta, gamma, zeta1));
}