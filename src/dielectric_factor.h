#ifndef _dielectricFactor_
#define _dielectricFactor_
#include <complex.h>
#include <math.h>

void dielectric_factor_sphere(double complex refind, double complex *K);

void dielectric_factor_spheroid(double complex refind, double aspect, double complex *K);

void dielectric_factor_hexagonal(double complex refind, double aspect, double complex *K);

void vectorK(int N, double complex *refind, double *aspect, double complex *K);

#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif