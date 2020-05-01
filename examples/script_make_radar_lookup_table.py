import os
import socket
from datetime import datetime
import numpy as np
import xarray as xr
import snowScatt

## Input parameters

Dmax = np.linspace(0.1e-3, 1.0e-1, 1000) # list of sizes
sizes = xr.IndexVariable(dims='size', data=Dmax,
                         attrs={'long_name':'Size - Maximum dimension',
                                'units':'meters'})
particle = 'Leinonen15tab00'
frequency =  np.array([5.6e9, 9.6e9, 13.6e9, 35.6e9, 94.0e9]) # frequencies
frequency = xr.IndexVariable(dims='frequency', data=frequency,
                             attrs={'units':'Hertz'})
temperature = xr.IndexVariable(dims='temperature', data=[270.0], # temperatures
                               attrs={'units':'Kelvin'})
Nangles = 721  # number of angles of the phase function subdivision
angles = xr.IndexVariable(dims='scat_angle',
                          data=np.linspace(0, np.pi, Nangles),
                          attrs={'long_name':'scattering angle',
                                 'units':'radians'})
filename = 'leinonen_unrimed_LUT.nc' # output filename

## Create empty xarray variables
dims = ['size', 'frequency', 'temperature']
coords = {'size': sizes,
          'frequency': frequency,
          'temperature': temperature}

Cext = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'Extinction cross-section',
                           'units':'meters**2'})
Cabs = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'Absorption cross-section',
                           'units':'meters**2'})
Csca = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'Scattering cross-section',
                           'units':'meters**2'})
Cbck = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'Radar backscattering cross section',
                           'units':'meters**2'})
asym = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'Asymmetry parameter',
                           'units':'dimensionless'})
dims = ['size', 'scat_angle', 'frequency', 'temperature']
angles = np.linspace(0.0, np.pi, Nangles)
coords['scat_angle'] = angles
phase = xr.DataArray(dims=dims, coords=coords,
                     attrs={'long_name':'Phase function',
                            'units':'dimensionless???'})
mass = xr.DataArray(dims=['size'],
                    coords={'size':sizes},
                    attrs={'long_name':'mass',
                           'units':'kilograms'})
vel = xr.DataArray(np.empty_like(sizes), dims=['size'], coords={'size':sizes},
                   attrs={'long_name':'Terminal fallspeed according to Boehm',
                          'units':'meters/second'})

## Compute
for fi, freq in enumerate(frequency):
    wl = snowScatt._compute._c/freq.values
    for ti, temp in enumerate(temperature):
        SS_RGA = snowScatt.calcProperties(diameters=Dmax,
                                          wavelength=wl,
                                          properties=particle,
                                          temperature=temp.values,
                                          Nangles=Nangles)
        ssCext, ssCabs, ssCsca, ssCbck, ssasym, ssphase, mass_p, ssvel = SS_RGA
        Cext.loc[sizes, freq, temp] = ssCext
        Cabs.loc[sizes, freq, temp] = ssCabs
        Csca.loc[sizes, freq, temp] = ssCsca
        Cbck.loc[sizes, freq, temp] = ssCbck
        asym.loc[sizes, freq, temp] = ssasym
        phase.loc[sizes, angles, freq, temp] = ssphase

# These last two depend only on size, no need to recompute
mass.loc[sizes] = mass_p
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