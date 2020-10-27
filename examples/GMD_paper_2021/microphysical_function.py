import numpy as np


Dice = np.logspace(-6, -1.1, 100) # limit for W band refl
Dcloud = np.logspace(-12, -4, 100)
Drain = np.logspace(-6, -2, 100)

ice_density = 900.0 # limited ice density from P3


def spheroid_volume(d, ar):
	ar*np.pi*d**3/6.0

	
def solid_spheroid_mass(d, ar):
	ice_density*spheroid_volume(d, ar)


def gammaPSD(N0, mu, lam):
    def N(D):
        return N0*D**mu*np.exp(-lam*D)
    return N

def iceP3mass(dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr):
    functions = [lambda x: np.nan,
                 lambda x: cs1*x**ds1,
                 lambda x: cs*x**ds,
                 lambda x: cgp*x**dg,
                 lambda x: csr*x**dsr]
    def mass(D):
        conditions = [D<=0.0,
                      (0.0<D)*(D<=dcrit),
                      (dcrit<D)*(D<=dcrits),
                      (dcrits<D)*(D<=dcritr),
                      dcritr<D]
        return np.piecewise(D, conditions, functions)
    return mass


def iceP3aspect(dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr):
    functions = [lambda x: np.nan,
                 lambda x: 1.0,
                 lambda x: 0.6,
                 lambda x: 1.0,
                 lambda x: 0.6] # I can perhaps simplify?
    def aspect(D):
        conditions = [D<=0.0,
                      (0.0<D)*(D<=dcrit),
                      (dcrit<D)*(D<=dcrits),
                      (dcrits<D)*(D<=dcritr),
                      dcritr<D]
        return np.piecewise(D, conditions, functions)
    return aspect


def calc_q(N0, mu, lam, dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr, D):
    N = gammaPSD(N0, mu, lam)
    mass = iceP3mass(dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr)
    return (N(D)*mass(D)*np.gradient(D)).sum()
vector_q = np.vectorize(calc_q, excluded=['D'], otypes=[np.float])

def calc_N(N0, mu, lam, D):
    N = gammaPSD(N0, mu, lam)
    return (N(D)*np.gradient(D)).sum()
vector_N = np.vectorize(calc_N, excluded=['D'], otypes=[np.float])
