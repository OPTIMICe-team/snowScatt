#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:18:28 2020

@author: dori
"""

from pytmatrix import tmatrix, scatter, tmatrix_aux, radar
from scattnlay import scattnlay

import snowScatt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import ConvexHull
import configparser
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from glob import glob
import os
import socket
import itertools
from datetime import datetime

plt.rcParams.update({'font.size':8})
filename = 'data/scattering_data_temp.nc'

particle_types = [
                  'Leinonen/subsequent-0.5',
                  'Leinonen/subsequent-1.0',
                  'needle',
                  'mixcolumndend',
                  ]

ssrga_particle_names = [
                        'Leinonen15tabB05',
                        'Leinonen15tabB10',
                        'vonTerzi_needle',
                        'vonTerzi_mixcoldend',
                        ]

legend_labels = [
                  'rimed 0.5',
                  'rimed 1.0',
                  'unrimed needle',
                  'unrimed mix',
                  ]
label_mapping = dict(zip(particle_types, legend_labels))

scatt_models = ['DDA', 'SSRGA']
scattering_models = xr.IndexVariable(dims='scattering_model',
                                     data=scatt_models,
                                     attrs={'long_name':'name of the scattering model used'})

part_names = [glob(pn+'/*/*noalign*.txt')+glob(pn+'/*/*.agg') for pn in particle_types]
part_names = list(itertools.chain.from_iterable(part_names))
particle_names = xr.IndexVariable(dims='particle_name', data=part_names,
                         attrs={'long_name':'Name identifier of the particle'})

freqs = [1.8000, 5.6000, 9.6000, 13.600, 35.600, 89.000, 94.000,
         157.00, 183.31, 220.00, 243.20, 325.15, 424.70, 448.00,
         664.00, 874.40,]
frequencies = xr.IndexVariable(dims='frequency', data=np.array(freqs)*1.0e9,
                               attrs={'units':'Hertz'})

## Create empty xarray variables
dims = ['particle_name', 'frequency', 'scattering_model']
coords = {'particle_name': particle_names,
          'frequency': frequencies,
          'scattering_model': scattering_models}

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
ssa = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'single scattering albedo',
                          'units':'dimensionless'})
volume = xr.DataArray(dims=['particle_name',],
                      coords={'particle_name' : particle_names},
                      attrs={'long_name':'volume occupied by the particle',
                             'units':'meters**3'})
ref_real = xr.DataArray(dims=['frequency',],
                        coords={'frequency' : frequencies},
                        attrs={'long_name':'ice refractive index',
                               'units':'dimensionless'})
ref_imag = xr.DataArray(dims=['frequency',],
                        coords={'frequency' : frequencies},
                        attrs={'long_name':'ice refractive index',
                               'units':'dimensionless'})
DmaD = xr.DataArray(dims=['particle_name'],
                    coords={'particle_name':particle_names},
                    attrs={'long_name':'particle maximum dimension',
                           'units':'meters'})

ice_density = snowScatt._compute._ice_density
c = snowScatt._compute._c

config_parser = configparser.RawConfigParser()
def parseCx(filename):
    with open(filename, 'r') as f:
        parse = '[HEADER]\n' + f.read()
        config_parser.read_string(parse)
        section = config_parser['HEADER']
        return float(section['Cext']), float(section['Cabs'])

def parse_command(command):
    eq_rad = float(command.split('-eq_rad')[-1].split()[0])
    lam = float(command.split('-lambda')[-1].split()[0])
    m1 = float(command.split('-m ')[-1].split()[0])
    m2 = float(command.split('-m ')[-1].split()[1])
    return eq_rad, lam, complex(m1, m2)

findstr = ['command', 'lambda', 'Total number']
def parse_log(logfile):
    with open(logfile, 'r') as f:
        results = {}
        for target in findstr:
            line = f.readline()
            while (not line.startswith(target)):
                line = f.readline()
            results[target] = line
    eq_rad , lam, ref_index = parse_command(results['command'])
    N = int(results['Total number'].split()[-1])
    return eq_rad, lam, ref_index, N

Nangles = 720
thssrga = np.linspace(0.0, 180.0, Nangles)

cached_dataset = True

###############################################################################
# HERE WE GO! LET'S CRACK THIS CODE
###############################################################################
if cached_dataset:
    dataset = xr.open_dataset(filename)
    Cext = dataset.Cext
    Cabs = dataset.Cabs
    Cbck = dataset.Cbck
    Csca = dataset.Csca
    asym = dataset.asym
    ssa = dataset.omega
    volume = dataset.volume
    ref_real = dataset.ref_real
    ref_imag = dataset.ref_imag
    
else:
    for particle, ssrga_particle in zip(particle_types, ssrga_particle_names):
        print('\n\n',particle, ssrga_particle)
        shapefiles = glob(particle+'/*/*.agg') + glob(particle+'/*/*.txt')
        for shapefile in shapefiles:
            results_folder = '/'.join(shapefile.split('/')[:-1]) + '/'
            results = glob(results_folder + '/[0-9]*')
            results = [pf for pf in results if float(pf.split('/')[-1]) in freqs] # eliminate possible 555.5 GHz prognosis folder
            results = [pf for pf in results if os.path.exists(pf+'/mueller_integr')] # check we are in the correct results
            if len(results):
                # Load the shapefile and count the number of voxels
                shape = np.loadtxt(shapefile)
                Ndipoles = len(shape)
                
                # Find maximum dimension in voxel units
                try:
                    hull3d=ConvexHull(shape)
                    hull3d=hull3d.points[hull3d.vertices]
                except: # if it is too small it fails, but it is also useless
                    hull3d=shape
                dmax = 0
                for pi in range(0, hull3d.shape[0]-1):
                    p0 = hull3d[pi,:]
                    for pj in range(pi+1, hull3d.shape[0]):
                        p1 = hull3d[pj, :]
                        r = p0-p1
                        dist = np.dot(r, r)
                        if dist > dmax:
                            dmax = dist
                dmax = dmax**0.5 # it will be scaled later
                
                # Fetch data from each frequency
                for freq_folder in results:
                    print(freq_folder)
                    fstr = freq_folder.split('/')[-1]
                    f = float(fstr+'e9')
                    wl = c/f
                    k = 2.0*np.pi/wl
                    ksquared = k*k
                    fCx = freq_folder + '/CrossSec-X'
                    fCy = freq_folder + '/CrossSec-Y'
                    fileC = freq_folder + '/CrossSec'
                    fMueller = freq_folder + '/mueller_integr'
                    fLog = freq_folder + '/log'
                    eq_rad, lam, ref_index, N = parse_log(fLog)
                    ref_real.loc[f] = ref_index.real
                    ref_imag.loc[f] = ref_index.imag
                    Vol = np.pi*(2.0*eq_rad)**3/6
                    volume.loc[shapefile] = Vol
                    mass = Vol*ice_density
                    d = np.cbrt(Vol/Ndipoles)
                    Dmax = dmax*d
                    DmaD.loc[shapefile] = Dmax
                    
                    Cextx, Cabsx = parseCx(fCx)
                    Cscax = Cextx-Cabsx
                    Cexty, Cabsy = parseCx(fCy)
                    Cext.loc[shapefile, f,'DDA'] = Cexty
                    Cabs.loc[shapefile, f,'DDA'] = Cabsy
                    Csca.loc[shapefile, f,'DDA'] = Cext.loc[shapefile, f,'DDA']-Cabs.loc[shapefile, f,'DDA']
                    
                    mueller = pd.read_csv(fMueller, delim_whitespace=True,
                                          index_col='theta')
                    th = mueller.index[mueller.index <= 180]
                    sinth = np.abs(np.sin(th*np.pi/180.0))
                    costh = np.cos(th*np.pi/180.0)
                    asym.loc[shapefile, f,'DDA'] = (sinth*costh*mueller.loc[th].s11).sum()/(sinth*mueller.loc[th].s11).sum()
                    Msum = (sinth*mueller.loc[th].s11).sum()*np.pi/(len(th)-1)
                    Cbck.loc[shapefile, f,'DDA'] = 2.0*np.pi*(mueller.loc[180.0].s11+mueller.loc[180.0].s22)/ksquared
                    ssa.loc[shapefile, f,'DDA'] = Csca.loc[shapefile, f,'DDA']/Cext.loc[shapefile, f,'DDA']
                    
                    SS_RGA = snowScatt.calcProperties(diameters=Dmax,
                                                      wavelength=wl,
                                                      properties=ssrga_particle,
                                                      ref_index=ref_index,
                                                      massScattering=mass,
                                                      Nangles=Nangles)
                    ssCext, ssCabs, ssCsca, ssCbck, ssasym, ssphase, mass_prop, vel, area = SS_RGA
                    Cext.loc[shapefile, f,'SSRGA'] = ssCext[0]
                    Cabs.loc[shapefile, f,'SSRGA'] = ssCabs[0]
                    Csca.loc[shapefile, f,'SSRGA'] = ssCsca[0]
                    Cbck.loc[shapefile, f,'SSRGA'] = ssCbck[0]
                    asym.loc[shapefile, f,'SSRGA'] = ssasym[0]
                    ssa.loc[shapefile, f,'SSRGA'] = ssCsca[0]/ssCext[0]
                    #print('Ce ', Cext.loc[shapefile, f,:].values, ' ', ssCext )
                    #print('Cb ', Cbck.loc[shapefile, f,:].values, ' ', ssCbck )
                    #print('g ', asym.loc[shapefile, f,:].values, ' ', ssasym )
    
    
    ## Finalize dataset and write netCDF file
    variables = {'Cext':Cext,
                 'Cabs':Cabs,
                 'Csca':Csca,
                 'Cbck':Cbck,
                 'asym':asym,
                 'omega':ssa,
                 'ref_real':ref_real,
                 'ref_imag':ref_imag,
                 'volume':volume,
                 'Dmax':DmaD,
                 #'phase':phase,
                 #'mass':mass,
                 #'vel':vel,
                 #'area':area
                 }
    
    global_attributes = {'created_by':os.environ['USER'],
                         'host_machine':socket.gethostname(),
                         #'particle_properties':particle,
                         'created_on':str(datetime.now()),
                         'comment':'this is just a test'}
    dataset = xr.Dataset(data_vars=variables,
                         coords=coords,
                         attrs=global_attributes)
    dataset.to_netcdf(filename, mode='w')





colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['v', '*', 'X', 'o',  '1', 's', 'P', 'D']
marker_size = 10
cmap='viridis'
fC, axc = plt.subplots(2, 2, figsize=(7, 5.5), constrained_layout=True)
scat = []
for ipt, particle_type in enumerate(particle_types):
    if ('Leinonen' in particle_type):
        cmap = 'spring'
    else:
        cmap = 'winter'
    print(particle_type)
    part_sel = [str(v.values) for v in Cext.particle_name if str(v.values).startswith(particle_type)]
    
    coloring_var = 1.0e-9*Cext.frequency.broadcast_like(Cext.loc[part_sel, :, 'DDA'])
    coloring_var.attrs = {'colorbar_title':'Frequency   [GHz]'}
    
    reff = np.cbrt(3*0.25*volume.loc[part_sel].broadcast_like(Cext.loc[part_sel, :, 'DDA'])/np.pi)
    wl = c/Cext.frequency.broadcast_like(Cext.loc[part_sel, :, 'DDA'])

    colb = axc[0, 1].scatter(Cabs.loc[part_sel, :, 'DDA'],
                      Cabs.loc[part_sel, :, 'SSRGA'],
                      label=label_mapping[particle_type],
                      marker=markers[ipt],
                      s=marker_size,
                      c = coloring_var,
                      cmap=cmap,
                      zorder=2,
                     )
    scat.append(colb)
    
    axc[1, 1].scatter(asym.loc[part_sel, :, 'DDA'],
                      asym.loc[part_sel, :, 'SSRGA'],
                      label=label_mapping[particle_type],
                      marker=markers[ipt],
                      s=marker_size,
                      c = coloring_var,
                      cmap=cmap,
                      zorder=2,
                     )
    axc[0, 0].scatter(Csca.loc[part_sel, :, 'DDA'],
                      Csca.loc[part_sel, :, 'SSRGA'],
                      #label=label_mapping[particle_type],
                      marker=markers[ipt],
                      s=marker_size,
                      c = coloring_var,
                      cmap=cmap,
                      zorder=2,
                     )
    axc[1, 0].scatter(Cbck.loc[part_sel, :, 'DDA'],
                      Cbck.loc[part_sel, :, 'SSRGA'],
                      label=label_mapping[particle_type],
                      marker=markers[ipt],
                      s=marker_size,
                      c = coloring_var,
                      cmap=cmap,
                      zorder=2,
                     )
    
lims = np.array([1e-14, 1e-3])
ticks = [1e-13, 1e-10, 1e-7, 1e-4]
for ax in [axc[0, 0], axc[1, 0], axc[0, 1]]:#, axc[1, 1]]:
    line_solid = ax.plot(lims, lims, c='k', zorder=1, label='1:1 match')
    dash_upper = ax.plot(lims, 4.0*lims, ls=':', c='k', zorder=1, label='$\pm$ 3 dB')
    dash_lower = ax.plot(lims, 0.25*lims, ls=':', c='k', zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

lims = [0, 1]
for ax in [axc[1, 1]]:
    ax.plot(lims, lims, c='k', zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

for ax, l in zip(axc.flatten(), ['(a)', '(b)', '(c)', '(d)']):
    ax.grid()
    ax.set_xlabel(' DDA   [m$^2$]')
    ax.set_ylabel(' SSRGA   [m$^2$]')
    ax.text(0, 1.02, l,  fontweight='black', transform=ax.transAxes)
axc[0, 0].legend()
axc[1, 1].set_xlabel(' DDA')
axc[1, 1].set_ylabel(' SSRGA')

axc[0, 0].set_title('C$_\mathrm{sca}$')
axc[0, 1].set_title('C$_\mathrm{abs}$')
axc[1, 0].set_title('C$_\mathrm{bck}$')
axc[1, 1].set_title('g')

fC.colorbar(scat[0], ax=axc[:, 1], location='right', aspect=40,
            label='RIMED     frequency [GHz]')
fC.colorbar(scat[-1], ax=axc[:, 0], location='left', aspect=40,
            label='UNRIMED     frequency [GHz]')
fC.savefig('evaluation_scattering.png', dpi=300)
fC.savefig('evaluation_scattering.pdf')


#%%

particle_types = [
                  'Leinonen/subsequent-0.5',
                  'Leinonen/subsequent-1.0',
                  'needle',
                  'mixcolumndend',
                  ]

ssrga_particle_names = [
                        'Leinonen15tabB05',
                        'Leinonen15tabB10',
                        'vonTerzi_needle',
                        'vonTerzi_mixcoldend',
                        ]

legend_labels = [
                  'SSRGA LS15 B05',
                  'SSRGA LS15 B10',
                  'SSRGA CaE needle',
                  'SSRGA CaE mix',
                  ]
label_mapping = dict(zip(particle_types, legend_labels))


def mass2reff(mass):
    vol = mass/ice_density
    reff = np.cbrt(vol*3.0/(4.0*np.pi))
    return reff

marker_size = 10
Dmax = np.linspace(0.01e-3, 0.05, 1000)
freq = 89.0e9
wavelength = c/freq
ref_index = complex(ref_real.loc[freq],ref_imag.loc[freq])
fS, axs = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

xmie = 10**np.linspace(-2, 0.8, 400)
xmie = xmie.reshape([len(xmie), 1])
ones = np.ones_like(xmie)
_, _, solidQs, _, solidQb, _, _, _, _, _ = scattnlay(xmie,
                                                     ones*ref_index)
axs[0].plot(xmie, solidQb, c='k', label='solid sphere')
axs[1].plot(xmie, solidQs, c='k', label='solid sphere')

Dmass, _, _ = snowScatt.snowMassVelocityArea(Dmax, 'vonTerzi_needle')
reff_sph = mass2reff(Dmass)
volume_fractions = 6.0*Dmass/(ice_density*np.pi*Dmax**3)
ones = np.ones_like(volume_fractions)
ref_soft = snowScatt.refractiveIndex.mixing.eps([ref_index**2*ones,
                                                 complex(1.0, 0.0)*ones],
                                                [volume_fractions,
                                                 1.0-volume_fractions])
ref_soft = np.sqrt(ref_soft)
xDmax = np.pi*Dmax/wavelength
xsoft = 2.0*np.pi*reff_sph/wavelength
xDmax = xDmax.reshape([len(xDmax), 1])
ref_soft = ref_soft.reshape([len(xDmax), 1])
_, _, Qss, _, Qbs, _, _, _, _, _ = scattnlay(xDmax, ref_soft)
corr = ((xDmax[:,0]/xsoft)**2)
Qss = Qss*corr
Qbs = Qbs*corr

axs[0].plot(xsoft, Qbs, '--', c='k', label='soft-sphere')
axs[1].plot(xsoft, Qss, '--', c='k', label='soft-sphere')

###############################################################################
Dmax = np.linspace(0.01e-3, 0.02, 1000)
aspect_ratio = 0.6
spheroidCs = 0.0*Dmax
spheroidCb = 0.0*Dmax
Dmass, _, _ = snowScatt.snowMassVelocityArea(Dmax, 'vonTerzi_needle')

reff_sph = mass2reff(Dmass)
volume_fractions = 6.0*Dmass/(ice_density*np.pi*aspect_ratio*Dmax**3) # ar

ones = np.ones_like(volume_fractions)
ref_soft = snowScatt.refractiveIndex.mixing.eps([ref_index**2*ones,
                                                 complex(1.0, 0.0)*ones],
                                                [volume_fractions,
                                                 1.0-volume_fractions])
ref_soft = np.sqrt(ref_soft) 

for iD, D in enumerate(Dmax):
    spheroid = tmatrix.Scatterer(radius=0.5*D,
                                 radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM,
                                 wavelength=wavelength,
                                 m=ref_soft[iD],
                                 axis_ratio=1.0/aspect_ratio
                                 )
    spheroid.set_geometry(tmatrix_aux.geom_vert_back)
    spheroidCs[iD] = scatter.sca_xsect(spheroid)
    spheroidCb[iD] = radar.radar_xsect(spheroid)

spheroidQb = spheroidCb/(np.pi*reff_sph**2)
spheroidQs = spheroidCs/(np.pi*reff_sph**2)
spheroid_x = 2.0*np.pi*reff_sph/wavelength

axs[0].plot(spheroid_x, spheroidQb, c='m', label='soft-spheroid')
axs[1].plot(spheroid_x, spheroidQs, c='m', label='soft-spheroid')
###############################################################################

for ipt, (particle_type, ssrga_particle) in enumerate(zip(particle_types, ssrga_particle_names)):
    part_sel = [str(v.values) for v in Cext.particle_name if str(v.values).startswith(particle_type)]
    print(particle_type, len(part_sel))
    
    reff = np.cbrt(3.0*0.25*volume.loc[part_sel].broadcast_like(Cext.loc[part_sel, freq, 'DDA'])/np.pi)
    wl = c/Cext.frequency.loc[freq].broadcast_like(Cext.loc[part_sel, freq, 'DDA'])
    size_parameter = 2.0*np.pi*reff/wl
    coloring_var.attrs = {'colorbar_title':'x = 2$\pi r_{eff}$/$\lambda$'}
    
    Qext = Cext.loc[part_sel, freq, 'DDA']/(np.pi*reff**2)
    Qsca = Csca.loc[part_sel, freq, 'DDA']/(np.pi*reff**2)
    Qbck = Cbck.loc[part_sel, freq, 'DDA']/(np.pi*reff**2)
    
    SS_RGA = snowScatt.calcProperties(diameters=np.linspace(0.1e-3, 0.1, 1000),
                                      wavelength=wavelength,
                                      properties=ssrga_particle,
                                      ref_index=ref_index,
                                      Nangles=Nangles)
    ssCext, ssCabs, ssCsca, ssCbck, ssasym, ssphase, mass_prop, vel, area = SS_RGA
    ssreff = np.cbrt(3.0*0.25*mass_prop/(np.pi*ice_density))
    ss_size_par = 2.0*np.pi*ssreff/wavelength
    ssQext = ssCext/(np.pi*ssreff**2)
    ssQsca = ssCsca/(np.pi*ssreff**2)
    ssQbck = ssCbck/(np.pi*ssreff**2)
    mask = ss_size_par < 6
    axs[1].plot(ss_size_par[mask], ssQsca[mask], lw=3,
                label=label_mapping[particle_type])
    axs[0].plot(ss_size_par[mask], ssQbck[mask], lw=3,
                label=label_mapping[particle_type])

JL05 = pd.read_csv('data/tables/dataJL_B0.5.csv')
reff05 = mass2reff(JL05.mkg*1.0e-3)
x05 = 2.0*np.pi*reff05/wavelength
Qb05 = JL05.Wb/(np.pi*reff05**2)
Qs05 = JL05.Ws/(np.pi*reff05**2)
axs[0].scatter(x05, Qb05, c='grey', label='DDA LS15 B05', s=marker_size)
axs[1].scatter(x05, Qs05, c='grey', label='DDA LS15 B05', s=marker_size)
JL10 = pd.read_csv('data/tables/dataJL_B1.0.csv')
reff10 = mass2reff(JL10.mkg*1.0e-3)
x10 = 2.0*np.pi*reff10/wavelength
Qb10 = JL10.Wb/(np.pi*reff10**2)
Qs10 = JL10.Ws/(np.pi*reff10**2)
axs[0].scatter(x10, Qb10, c='k', label='DDA LS15 B10', s=marker_size)
axs[1].scatter(x10, Qs10, c='k', label='DDA LS15 B10', s=marker_size)
###############################################################################

###############################################################################

RHfile = 'data/tables/scatdb.csv'
dataRH = pd.read_csv(RHfile)
dataRH = dataRH[dataRH.flaketype == 9] # sector snowflake
#dataRH = dataRH[dataRH.flaketype == 20] # aggregate spherical
#dataRH = dataRH[dataRH.flaketype == 21] # aggregate oblate
#dataRH = dataRH[dataRH.flaketype == 22] # aggregate prolate
dataRH = dataRH[abs(dataRH.temperaturek-263) < 2]
dataRH = dataRH[abs(dataRH.frequencyghz-90) < 30]

dataRH.set_index('max_dimension_mm', inplace=True)
dataRH.sort_index(inplace=True)
dataRH['aeff'] = dataRH.aeffum*1.0e-6
dataRH['xeff'] = dataRH.aeff*2.0*np.pi*dataRH.frequencyghz*1.0e9/c
dataRH['Qs'] = dataRH.csca/(np.pi*dataRH.aeff**2)
dataRH['Qb'] = dataRH.cbk/(np.pi*dataRH.aeff**2)
axs[0].scatter(dataRH.xeff, dataRH.Qb, c='c', label='DDA Liu sector', s=marker_size)
axs[1].scatter(dataRH.xeff, dataRH.Qs, c='c', label='DDA Liu sector', s=marker_size)

axs[0].set_ylim([1e-8, 100])
axs[1].set_ylim([1e-4, 10])
for ax in axs:
    ax.set_xlim([0.1, 10])
    ax.set_xscale('log')
    ax.set_yscale('log')
axs[0].set_xlabel('x = 2$\pi r_{eff}/\lambda$')
axs[1].set_xlabel('x = 2$\pi r_{eff}/\lambda$')
axs[0].set_ylabel('Q$_{bck}$ = C$_{bck}/\pi r^2_{eff}$')
axs[1].set_ylabel('Q$_{sca}$ = C$_{sca}/\pi r^2_{eff}$')

axs[0].grid()
axs[1].grid()

handles, labels = axs[0].get_legend_handles_labels()
leg = axs[0].legend(handles[:3], labels[:3], loc=2, ncol=1)
axs[1].legend(handles[3:7], labels[3:7], loc=4, ncol=1)
axs[0].legend(handles[7:], labels[7:], loc=3, ncol=1)
axs[0].add_artist(leg)

axs[0].text(0, 1.02, '(a)',  fontweight='black', transform=axs[0].transAxes)
axs[1].text(0, 1.02, '(b)',  fontweight='black', transform=axs[1].transAxes)

axs[0].set_title('Backscattering efficiency')
axs[1].set_title('Scattering efficiency')

fS.savefig('efficiencies.png', dpi=300)
fS.savefig('efficiencies.pdf')