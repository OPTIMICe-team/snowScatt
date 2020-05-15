"""
This module just list some constants that are used across multiple submodules in
the snowScatt package. The purpose of that is to ensure the consistency of those
values.
"""

 # The speed of light in the vacuum m/s. This is the TRUE value!!
_c = 2.99792458e8

 # a value for the density of solid ice Ih kg/m**3
_ice_density = 917.0

# a value for gravity acceleration m/s**2
_g = 9.807

# The standard mean sea level air density kg/m** p=1013 hPa, T=20°C
_rho0 = 1.2038631624242195

# Standard dynamic viscosity TODO: shouldn't play huge role, but check it
_mu0 = 1.717696e-5 # Pa*s

# Standard kinematic viscosity
_nu0 = _mu0/_rho0