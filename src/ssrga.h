#ifndef _SSRGA_
#define _SSRGA_

#include <stdio.h>
#include <complex.h>
#include <math.h>

void ssrga(int Nparticles,
	       double *Deff, double *Vol, double *wl, double complex *K,
	       double *kappa, double *gamma, double *beta, double *zeta1,
	       int Ntheta, double *Cext, double *Cabs, double *Csca,
	       double *Cbck, double *asym, double *phase);

void ssrgaBack(int Nparticles, double *Deff, double *Vol,
	             double *wl, double complex *K,
	             double *kappa, double *gamma, double *beta, double *zeta1,
	             double *Cbck);

void ssrga_single(double Deff, double Vol, double wl, double complex K,
	              double kappa, double gamma, double beta, double zeta1,
	              int Ntheta, double *Cext, double *Cabs, double *Csca,
	              double *Cbck, double *asym, double *phase);

void back_single(double Deff, double Vol, double wl, double complex K,
	             double kappa, double gamma, double beta, double zeta1,
	             double *Cbck);
// Ve do not need to declare inline functions
//inline double prefactor(double wavenumber, double complex K, double Vol);
//inline double mean_term(double x, double kappa);
//inline double sum_term(double x, double beta, double gamma, double zeta);

#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif