import os
import socket
from datetime import datetime
import numpy as np
import xarray as xr
import snowScatt

## Input parameters

sizes = np.linspace(0.1e-3, 1.0e-1, 1000) # list of sizes
particle = 'Leinonen15tab00'
frequency = np.array([5.6e9, 9.6e9, 13.6e9, 35.6e9, 94.0e9]) # list of frequencies
temperature = [270.0] # list of temperatures
Nangles = 721 # number of angles of the phase function subdivision
filename = 'leinonen_unrimed_LUT.nc' # output filename

## Create empty xarray variables
dims = ['size', 'frequency', 'temperature']
coords = {'size': sizes,
          'frequency': frequency,
          'temperature': temperature}
empty = np.empty([len(sizes), len(frequency), len(temperature)])
Cext = xr.DataArray(empty, dims=dims, coords=coords,
                    attrs={'long_name':'Extinction cross-section',
                           'units':'meters**2'})
Cabs = xr.DataArray(empty, dims=dims, coords=coords,
                    attrs={'long_name':'Absorption cross-section',
                           'units':'meters**2'})
Csca = xr.DataArray(empty, dims=dims, coords=coords,
                    attrs={'long_name':'Scattering cross-section',
                           'units':'meters**2'})
Cbck = xr.DataArray(empty, dims=dims, coords=coords,
                    attrs={'long_name':'Radar backscattering cross section',
                           'units':'meters**2'})
asym = xr.DataArray(empty, dims=dims, coords=coords,
                    attrs={'long_name':'Asymmetry parameter',
                           'units':'dimensionless'})
dims = ['size', 'scat_angle', 'frequency', 'temperature']
angles = np.linspace(0.0, np.pi, Nangles)
coords['scat_angle'] = angles
phase = xr.DataArray(np.empty([len(sizes), Nangles,
                               len(frequency), len(temperature)]),
                     dims=dims, coords=coords,
                     attrs={'long_name':'Phase function',
                            'units':'dimensionless???'})
mass = xr.DataArray(np.empty_like(sizes), dims=['size'],
                    coords={'size':sizes},
                    attrs={'long_name':'mass',
                           'units':'kilograms'})
vel = xr.DataArray(np.empty_like(sizes), dims=['size'], coords={'size':sizes},
                   attrs={'long_name':'Terminal fallspeed according to Boehm',
                          'units':'meters/second'})

## Compute
for fi, freq in enumerate(frequency):
    wl = snowScatt._compute._c/freq
    for ti, temp in enumerate(temperature):
        SS_RGA = snowScatt.calcProperties(diameters=sizes,
                                          wavelength=wl,
                                          properties=particle,
                                          temperature=temp,
                                          Nangles=Nangles)
        ssCext, ssCabs, ssCsca, ssCbck, ssasym, ssphase, mass_prop, ssvel = SS_RGA
        Cext.loc[sizes, freq, temp] = ssCext
        Cabs.loc[sizes, freq, temp] = ssCabs
        Csca.loc[sizes, freq, temp] = ssCsca
        Cbck.loc[sizes, freq, temp] = ssCbck
        asym.loc[sizes, freq, temp] = ssasym
        phase.loc[sizes, angles, freq, temp] = ssphase
# These last two depend only on size, no need to recompute
mass.loc[sizes] = mass_prop
vel.loc[sizes] = ssvel


## Finalize dataset and write netCDF file

variables = {'Cext':Cext,
             'Cabs':Cabs,
             'Csca':Csca,
             'Cbck':Cbck,
             'asym':asym,
             'phase':phase,
             'mass':mass,
             'vel':vel,}
global_attributes = {'created_by':os.environ['USER'],
                     'host_machine':socket.gethostname(),
                     'particle_properties':particle,
                     'created_on':str(datetime.now()),
                     'comment':'this is just a test'}

dataset = xr.Dataset(data_vars=variables,
                     coords=coords,
                     attrs=global_attributes)

dataset.to_netcdf(filename)