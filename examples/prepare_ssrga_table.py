#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:08:20 2020

@author: dori
"""

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from glob import glob
#import gzip
import pandas as pd
from scipy.spatial import ConvexHull
import snowScatt.ssrga as ssrga
from snowScatt._constants import _ice_density
from snowScatt.fallSpeed import Boehm1992 as B92
from snowScatt.fallSpeed import KhvorostyanovCurry2005 as KC05
from datetime import datetime as dt


path_to_shapefiles = '/home/dori/develop/pySAM/dat*/*.dat'#'../data/simultaneous-0.0/'
shapefiles = glob(path_to_shapefiles)# + '*.agg')

cols = ['Dmax', 'mass', 'area', 'area_function', 'resolution',
        'vel_Bohm', 'vel_KC']
data = pd.DataFrame(index=np.arange(len(shapefiles)), columns=cols) 

deg = 'random'
deg = '00'
for i, shapefile in enumerate(shapefiles):
    shape = np.loadtxt(shapefile)[:,0:3]
    d = 40.0e-6 # if I do not have metadata I am just assuming it is 40um
    d = 20.0e-6
    # d = 1.0 # if the shapefile is not on a regular grid d == 1
    # Calculates Dmax myself
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
    dmax = d*dmax**0.5
    
    # Calculates projected area, along z axis
    xy = (0, 1) # drop z coordinate
    projection=pd.DataFrame(shape[:, xy]).drop_duplicates().values
    #projection=pd.DataFrame(np.round(shape[:, xy]/d)).drop_duplicates().values*d # if you do not have a regular grid you need to regularize it first...
    area = projection.shape[0]*d**2
    mass = shape.shape[0]*d**3*_ice_density
    vel_B92 = B92(dmax, mass, area) # perhaps aspect ratio?
    vel_KC = KC05(dmax, mass, area)
    # Calculate area function, along z axis
    area_func = ssrga.area_function(shape*d, d, dmax, theta=np.pi*float(deg)/180.0) # multiply by resolution if you have a regular grid
    #area_func = ssrga.area_function(shape, d, dmax, theta=np.pi*float(deg)/180.0) # if you have full precision floating point coordinates do this
    #area_func = ssrga.area_function(shape*d, d, dmax, theta=np.pi*np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])/180.0)
    
    # Put data in a DataFrame
    data.loc[i] = [dmax, mass, area, area_func, d, vel_B92, vel_KC]
data.to_hdf('area_functions_'+deg+'.h5', key='area') # save area_functions on a hdf5 file

#%% Create bins for SSRGA
minBin = 2.0e-3
maxBin = 17.0e-3#23.0e-3
resBin = 1.0e-3
bin_edges = np.arange(minBin-resBin*0.5, maxBin+resBin, resBin)
bin_center = np.arange(minBin, maxBin+resBin*0.5, resBin)
bins = pd.cut(data['Dmax'], bin_edges)
groups = data.groupby(bins)

# Define function that computes table data for each bin
def reduction(x):
    d = {}
    d['Dmax'] = np.median(x['Dmax'])
    d['area'] = np.median(x['area'])
    d['mass'] = np.median(x['mass'])
    d['vel_Bohm'] = np.median(x['vel_Bohm'])
    d['vel_KC'] = np.median(x['vel_KC'])
    max_len = np.max([y.shape[1] for y in x['area_function']])
    Nparticles = len(x)
    Nsamples = x['area_function'].iloc[0].shape[0] # assume same number of samples
    area_func = np.zeros((Nparticles*Nsamples, max_len))
    for iy, y in enumerate(x['area_function']):
        area_func[iy*Nsamples:iy*Nsamples+y.shape[0], :y.shape[1]] = y
    res = ssrga.fitSSRGA(area_func,
                         x['Dmax'].values,
                         x['resolution'].values,
                         max_index_largescale=12, do_plots=False)
    kappa, beta, gamma, zeta, alpha_eff, volume = res
    d['kappa'] = kappa
    d['gamma'] = gamma
    d['beta'] = beta
    d['zeta'] = zeta
    d['alpha_eff'] = alpha_eff
    return pd.Series(d, index=d.keys())

# Make simple power-Law fits for mass, area and velocity
bv, av = np.polyfit(x=np.log10(data['Dmax'].values.astype(np.float)),
                    y=np.log10(data['vel_Bohm'].values.astype(np.float)),
		           deg=1)
av = 10.0**(av)
bm, am = np.polyfit(x=np.log10(data['Dmax'].values.astype(np.float)),
                    y=np.log10(data['mass'].values.astype(np.float)),
		           deg=1)
am = 10.0**(am)
ba, aa = np.polyfit(x=np.log10(data['Dmax'].values.astype(np.float)),
                    y=np.log10(data['area'].values.astype(np.float)),
		           deg=1)
aa = 10.0**(aa)

# Apply reduction to the groups and write the table
reducted = groups.apply(reduction)
reducted.to_hdf('ssrga_'+deg+'.h5', key='area')
reducted['Diam_max'] = reducted.index.mid
reducted.set_index('Diam_max', inplace=True)
reducted=reducted.astype(np.float64)
avgstr = 'am={},bm={},av={},bv={},aa={},ba={},'.format(am, bm, av, bv, aa, ba)
with open('table.csv', 'w') as csv:
    csv.write('# Example data file \n')
    csv.write('# created {} \n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
    csv.write('# '+avgstr+'monomer_alpha={},\n'.format(0.3))
reducted.to_csv('table.csv', mode='a', float_format='%7.6e')