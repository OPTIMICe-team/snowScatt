#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:08:20 2020

@author: dori
"""

import numpy as np
from glob import glob
import gzip
import pandas as pd
from scipy.spatial import ConvexHull
import snowScatt.ssrga as ssrga

rho_ice = 917.0
rho_air = 1.287
nu_air = 1.717696e-5/rho_air
        
path_to_shapefiles = '../data/simultaneous-0.0/'
shapefiles = glob(path_to_shapefiles + '*.agg')

cols = ['Dmax', 'area', 'mass', 'area_function']
data = pd.DataFrame(index=np.arange(len(shapefiles)), columns=cols) 


# This assumes already the shape to be on a regular grid spaced by the
# resolution. shape must contain integer coordinates
# It also goes only into z direction
# TODO: make it general, take resolution as input for generalized shapes
# TODO: consider the possibility of having polar angle incidence !=0, for this
# case assume also a certain number of samples in azimuth (user defined)
def area_function(shape):
    # I just have to count the number of occupied voxels per z coordinate
    z_sorted, z_counts = np.unique(shape[:, 2], return_counts=True)
    return z_counts

for i, shapefile in enumerate(shapefiles):
    # load the shapefile
    # with gzip.open(shapefile, 'r') as sf:
        #shape = np.loadtxt(sf)
    shape = np.loadtxt(shapefile)
    
    # load the metadata, for voxel resolution and cross-checking Dmax and area
    meta = shapefile + '.gz.meta'
    try:
        attributes=eval(open(meta).read())
        d=attributes['grid_res']
        dmax_att=attributes['max_diam']
        meta = True
    except:
        d = 40.0e-6 # if I do not have metadata I am just assuming it is 40um
        meta = False
    
    # Calculates Dmax myself
    try:
        hull3d=ConvexHull(shape)
        hull3d=hull3d.points[hull3d.vertices]
    except:
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
    if not meta:
        dmax_att = dmax
    
    # Calculates projected area, along z axis
    xy = (0, 1) # drop z coordinate
    projection=pd.DataFrame(shape[:, xy]).drop_duplicates().values
    area = projection.shape[0]*d**2
    mass = shape.shape[0]*d**3*rho_ice
    
    # Calculate area function, along z axis
    area_func = area_function(shape)
    
    # Put data in a DataFrame
    data.loc[i] = [dmax, mass, area, area_func]

# Create bins for SSRGA
minBin = 2.0e-3
maxBin = 23.0e-3
resBin = 1.0e-3
bin_edges = np.arange(minBin-resBin*0.5, maxBin+resBin, resBin)
bin_center = np.arange(minBin, maxBin+resBin*0.5, resBin)
bins = pd.cut(data['Dmax'], bin_edges)
groups = data.groupby(bins)
def reduction(x):
    d = {}
    d['Dmax'] = np.median(x['Dmax'])
    d['area'] = np.median(x['area'])
    d['mass'] = np.median(x['mass'])
    max_len = np.max([len(y) for y in x['area_function']])
    area_func = np.zeros((len(x['area_function']),max_len))
    for iy, y in enumerate(x['area_function']):
        area_func[iy, :len(y)] = y
    #d['area_function'] = area_func
    res = ssrga.fitSSRGA(area_func,
                         x['Dmax'].values,
                         np.ones(len(x['Dmax']))*40.0e-6,
                         max_index_largescale=12, do_plots=True)
    kappa, beta, gamma, zeta, alpha_eff, volume = res
    d['kappa'] = kappa
    d['gamma'] = gamma
    d['beta'] = beta
    d['zeta'] = zeta
    d['alpha_eff'] = alpha_eff
    d['volume'] = volume
    return pd.Series(d, index=d.keys())

reducted = groups.apply(reduction)