#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:40:47 2020

@author: dori
"""

from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

path = '../python/snowProperties/'
particles = glob(path+'ssrga*.csv')
for particle in particles:
    table = pd.read_csv(particle, comment='#').set_index('Diam_max').interpolate(limit_direction='both') 
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.scatter(table.index*1.0e3, table.beta, c='k')
    axt = ax.twinx()
    axt.scatter(table.index*1.0e3, table.gamma, c='r')
    ax.set_xlabel('Dmax   [mm]')
    ax.set_ylabel(r'$\beta$')
    axt.set_ylabel(r'$\gamma$', color='red')
    ax.grid()
    axt.grid(color='red', alpha=0.3)
    axt.spines['right'].set_color('red')
    axt.tick_params(axis='y', colors='red')
    axt.yaxis.label.set_color('red')
    name = particle.split('/')[-1][6:-4]
    ax.set_title(r'SSRGA $\beta,\gamma$  for '+ name[7:] + ' aggregates')
    fig.savefig(name + '.png', dpi=300)
    fig.savefig(name + '.pdf', dpi=300)