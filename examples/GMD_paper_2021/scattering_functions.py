import numpy as np
import pandas as pd

from pytmatrix import tmatrix, tmatrix_aux, radar, refractive

from microphysical_function	import Dice, Dcloud, Drain, gammaPSD


waves = {'X': tmatrix_aux.wl_X, 'Ka': tmatrix_aux.wl_Ka, 'W': tmatrix_aux.wl_W}


######################################################################################
# Generic Tmatrix function for single spheroid backscattering
######################################################################################
def tm_reflectivity(size, wl, n, ar=1.0): # Takes size and wl in meters
        scatt = tmatrix.Scatterer(radius=0.5e3*size, # conversion to millimeter radius
                                  radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM,
                                  wavelength=wl*1.0e3, # conversion to millimeters
                                  m=n,
                                  axis_ratio=1.0/ar)
        scatt.set_geometry(tmatrix_aux.geom_vert_back)
        return radar.radar_xsect(scatt) # mm**2 ... need just to integrate and multiply by Rayleigh factor

######################################################################################
# Liquid scattering routines - use Tmatrix
######################################################################################
scat_cloud = pd.DataFrame(index=Dcloud, columns=[w for w in waves.keys()])
scat_rain = pd.DataFrame(index=Drain, columns=[w for w in waves.keys()])
for w in waves.keys():
	n_water = refractive.m_w_0C[waves[w]]
	for d in Dcloud:
		scat_cloud.loc[d, w] = tm_reflectivity(d, 1.0e-3*waves[w], n_water)
	for d in Drain:
		scat_rain.loc[d, w] = tm_reflectivity(d, 1.0e-3*waves[w], n_water)


def calc_liquid_Z(N0, mu, lam, D, z):
	N = gammaPSD(N0, mu, lam)
	return (N(D)*z*np.gradient(D)).sum()
vector_liquid_Z = np.vectorize(calc_liquid_Z, excluded=['D', 'z'], otypes=[np.float])


z_tm_solid_sphere = pd.DataFrame(index=Dice, columns=[w for w in waves.keys()])
z_tm_BF95_spheroid = pd.DataFrame(index=Dice, columns=[w for w in waves.keys()])
BF95density = lambda x, ar: np.minimum(0.0121*6.0*x**(-1.1)/(np.pi*ar), 900.0)
for w in waves.keys():
	wl = waves[w]
	for d in Dice:
		z_tm_solid_sphere.loc[d, w] = tm_reflectivity(d, 1.0e-3*wl, refractive.mi(wl, 0.900), 1.0)
		z_tm_BF95_spheroid.loc[d, w] = tm_reflectivity(d, 1.0e-3*wl, refractive.mi(wl, 1.0e-3*BF95density(d, 0.6)), 0.6)

a_graupel = np.linspace(21, 472, 1000)
a_partrimed = np.linspace(0.0121, 0.024, 1000)

#z_tm_graupel = pd.read_csv('z_tm_graupel_'+freq_str+'.csv', index_col=0, dtype=np.float64, engine='c')
#z_tm_graupel.columns = z_tm_graupel.columns.astype(np.float)
#z_tm_partrimed = pd.read_csv('z_tm_partrimed_'+freq_str+'.csv', index_col=0, dtype=np.float64, engine='c')
#z_tm_partrimed.columns = z_tm_partrimed.columns.astype(np.float)