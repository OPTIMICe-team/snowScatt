#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:40:47 2020

@author: dori

This script reproduces Fig 1 of the paper
snowScatt 1.0: Consistent model of microphysical and scattering properties of rimed and unrimed snowflakes based on the self-similar Rayleigh-Gans Approximation
It does not require specific input data
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

path = '../../python/snowProperties/'
particle = path + 'ssrga_coeffs_dendrite.csv'
table = pd.read_csv(particle, comment='#').set_index('Diam_max').interpolate(limit_direction='both') 
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.scatter(table.index*1.0e3, table.beta, c='k')
axt = ax.twinx()
axt.scatter(table.index*1.0e3, table.gamma, c='r')
ax.set_xlabel('Dmax   [mm]')
ax.set_ylabel(r'$\beta$')
ax.set_yscale('log')
ax.set_ylim(0.1,10)
axt.set_yscale('log')
axt.set_ylim(1, 4)
axt.set_ylabel(r'$\gamma$', color='red')
ax.grid()
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
axt.grid(color='red', alpha=0.3, which='both')
axt.spines['right'].set_color('red')
axt.tick_params(axis='y', colors='red', which='both')
axt.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
axt.yaxis.set_major_formatter(FormatStrFormatter('%d'))
axt.yaxis.label.set_color('red')
name = particle.split('/')[-1][6:-4]
ax.set_title(r'SSRGA $\beta,\gamma$  for '+ name[7:] + ' aggregates')
fig.tight_layout()
fig.savefig(name + '.png', dpi=300)
fig.savefig(name + '.pdf', dpi=300)