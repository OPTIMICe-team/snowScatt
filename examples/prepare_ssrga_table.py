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

The script is divided into two parts that can potentially be run independently.
This is the second part that reads an already generated file of area_functions
and fits the SSRGA parameters. To generate the hdf5 area_functions file use the
first script prepare_area_functions.py

The microphysical properties recorded in the hdf5 file are collected and 
grouped according to a user specified binning scheme.
For each bin the statistical SSRGA parameters and microphysical
properties are derived.
The power-law fits to mass-size and area-size relations are derived from the
whole population of shapefiles.

"""
from datetime import datetime as dt
import pandas as pd
import numpy as np

import snowScatt.ssrga as ssrga

# path to the hdf5 file containing the shape properties
deg = '00'
area_functions_file = 'area_functions_'+deg+'.h5'
data = pd.read_hdf(area_functions_file)

#%% Create bins for SSRGA
minBin = 2.0e-3 # Set minimum center size for bins
maxBin = 15.0e-3#23.0e-3 # Set maximum center size for bins
resBin = 1.0e-3 # Set regular bin resolution

# NOTE: there is nothing that prevents you from having a not evenly spaced binning
# snowScatt will work also with un regular tables

table_name = 'table.csv'

#################################################################################
# DONE. I will take care of the rest
#################################################################################

# Deriving binning from bin parameters
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
# reducted.to_hdf('ssrga_'+deg+'.h5', key='area') # unnecessary, but snowScatt could move to binary tables
reducted['Diam_max'] = reducted.index.mid
reducted.set_index('Diam_max', inplace=True)
reducted=reducted.astype(np.float64)
avgstr = 'am={},bm={},av={},bv={},aa={},ba={},'.format(am, bm, av, bv, aa, ba)
with open('table.csv', 'w') as csv:
    csv.write('# Example data file \n')
    csv.write('# created {} \n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
    csv.write('# '+avgstr+'monomer_alpha={},\n'.format(0.3))
reducted.to_csv(table_name, mode='a', float_format='%7.6e')