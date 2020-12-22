#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:08:20 2020

@author: dori

This script show an example on how to use snowScatt to prepare a snow-table
that snowScatt can use to compute the snowflake scattering and microphysical
properties. To run this script you need a reasonable amount of snow shapes
generated using either a physical or an heuristic model.
One possible option is the aggregation and riming code from Jussi Leinonen
https://github.com/jleinonen/aggregation

The code needs to be adapted to the specific situation. The path where to find
the shapefiles must be defined. Weather the shapefiles are saved using full
floating point notation or a regular grid also makes a difference.
The relevant information on how to make these adjustments is left as inline
comments in the script.

The script is divided into two parts that can potentially be run independently.
This is only the first part of the script were the area_functions for each
individual shape is calculated and the result is saved into a hdf5 file.

Run the second script prepare_ssrga_table.py to load the hdf5 and derive the
SSRGA fits

"""

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from glob import glob
import pandas as pd
from scipy.spatial import ConvexHull
import snowScatt.ssrga as ssrga
from snowScatt._constants import _ice_density
from snowScatt.fallSpeed import Boehm1992 as B92
from snowScatt.fallSpeed import KhvorostyanovCurry2005 as KC05
from datetime import datetime as dt

# Create a reasonable pattern for the shapefiles you want to include
path_to_shapefiles = '/home/dori/develop/pySAM/dat*/*.dat'#'../data/simultaneous-0.0/'
path_to_shapefiles = '/data/optimice/aggregate_model/Jussis_aggregates_ssrga/Jussis_aggregates_mixcolumndend_rimed/rimeelwp_0.2/mixcolumndend_rimelwp_*_nmono_*.txt'
shapefiles = glob(path_to_shapefiles)# + '*.agg')

cols = ['Dmax', 'mass', 'area', 'area_function', 'resolution',
        'vel_Bohm', 'vel_KC']
data = pd.DataFrame(index=np.arange(len(shapefiles)), columns=cols) 

# Set the incidence angle. Snowflakes are assumed to be oriented.
# If the incidence angle differs from n*pi also azimuth averaging is applied automatically
# For total random orientations set deg = 'random'
deg = 'random'
deg = '00' # Use string literals convertible to numbers

# Set the flag for regular grid
# DDA calculations accelerated with FFT require a regular grids.
# Regular grids have only integers as coordinates
regular = False

# Set the resolution of all the shapefiles in meters.
# In principle it would be possible to have a different resolution for each shape
# In this case you have to set it in a smart way inside the following for loop
d = 10.0e-6 # if I do not have metadata I am just assuming it is 40um

# Set a name for the hdf5 file with the area functions
area_functions_file = 'area_functions2_'+deg+'.h5'

#################################################################################
# DONE. I will take care of the rest
#################################################################################

for i, shapefile in enumerate(shapefiles):
    print(i,' of ', len(shapefiles), ' processed')
    shape = np.loadtxt(shapefile)[:, 0:3] # the first 3 numbers are the coordinates
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
    dmax = dmax**0.5
    if regular:
        dmax = dmax*d
        dscale = 1
    else:
        dscale = d
    
    # Calculates projected area, along z axis
    xy = (0, 1) # drop z coordinate
    projection = pd.DataFrame(np.round(shape[:, xy]/dscale)).drop_duplicates().values
    area = projection.shape[0]*d**2
    mass = shape.shape[0]*d**3*_ice_density
    vel_B92 = B92(dmax, mass, area) # perhaps aspect ratio?
    vel_KC = KC05(dmax, mass, area)
    # Calculate area function, along z axis
    area_func = ssrga.area_function(shape*d/dscale, d, dmax, theta=np.pi*float(deg)/180.0) # multiply by resolution 
    # Put data in a DataFrame
    data.loc[i] = [dmax, mass, area, area_func, d, vel_B92, vel_KC]
data.to_hdf(area_functions_file, key='area') # save area_functions on a hdf5 file