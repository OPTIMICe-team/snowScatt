#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:44:27 2020

@author: dori
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snowScatt
import random

plt.rcParams.update({'font.size':14})

random.seed(12345678576678*234567890987654)

dda = pd.read_csv('data/tables/dataJL_A0.0.csv') # 550 particles
freq = 94.0e9
dda_f = 'Wb'
K2 = 0.72

freq = 35.6e9
dda_f = 'Ab'
K2 = 0.72

wavelength = snowScatt._compute._c/freq
scaleZ = 1.0e18*wavelength**4/(np.pi**5*K2)
number_of_samples = [1, 2, 4, 8]
D0 = np.linspace(0.5, 5, 50)

#%% Create bins for SSRGA
minBin = 2.0
maxBin = 22.0
resBin = 1.5
bin_edges = np.arange(minBin-resBin*0.5, maxBin+resBin, resBin)
bin_center = np.arange(minBin, maxBin+resBin*0.5, resBin)
bins = pd.cut(dda['Dmax'], bin_edges)
groups = dda.groupby(bins)

fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
ax[0].bar(bin_center, groups.count().Dmax)
ax[0].set_ylabel('Number of particles')
ax[0].set_xlabel('D$_\mathrm{max}$   [mm]')


def expPSD(D, lam):
    return np.exp(-lam*D)


def mean_refl(x):
    d = {}
    d['sigma'] = np.mean(x[dda_f])
    d['mass'] = np.sqrt(np.mean(x['mkg']**2))*1.0e-3 # I really have to fix this mass in grams
    d['D'] = np.median(x['Dmax'])
    return pd.Series(d, index=d.keys())


def random_refl_many(how_many):
    def random_refl(x):
        d = {}
        d['sigma'] = np.mean(random.choices(x[dda_f].values, k=how_many))
        return pd.Series(d, index=d.keys())
    return random_refl
mean_prop = groups.apply(mean_refl)
mean_Z = mean_prop.sigma.values
mean_mass = mean_prop.mass.values
mean_D = mean_prop.D.values

PSDs = expPSD(np.tile(bin_center, (len(D0), 1)),
              1.0/np.tile(D0, (len(bin_center), 1)).T)
def calcZ(Cb):
    return 10.0*np.log10((PSDs*np.tile(Cb, (len(D0), 1))).sum(axis=1)*scaleZ)

Zdda = calcZ(mean_Z)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def hex2rgb(hexcol):
    h = hexcol.lstrip('#')
    n = 16**(len(h)//3)-1
    return tuple(int(h[i:i+2], 16)/n for i in (0, 2, 4))

def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]
white = [1, 1, 1]

quantile = 0.1
alpha = 0.5
for iN, Nsamples in enumerate(number_of_samples):
    color = make_rgb_transparent(hex2rgb(colors[iN]), white, alpha)
    samples = np.stack([groups.apply(random_refl_many(Nsamples)).sigma.values for i in range(1000)])
    lower = np.quantile(a=samples, q=quantile, axis=0)
    upper = np.quantile(a=samples, q=1.0-quantile, axis=0)
    ax[1].fill_between(bin_center, lower, upper, facecolor=color,
                       label=str(Nsamples)+' samples spread')
    Zsamples = np.apply_along_axis(calcZ, axis=1, arr=samples) - Zdda
    Zlower = np.quantile(a=Zsamples, q=quantile, axis=0)
    Zupper = np.quantile(a=Zsamples, q=1.0-quantile, axis=0)
    ax[2].fill_between(D0, Zlower, Zupper, facecolor=color,
                       label=str(Nsamples)+' samples spread')


ax[1].plot(bin_center, mean_Z, lw=2, c='k', label='mean DDA')
ax[2].plot(D0, Zdda-Zdda, lw=2, c='k', label='mean DDA')

Crnd = groups.apply(random_refl_many(1)).sigma
ax[1].plot(bin_center, Crnd, c='black', ls=':', lw=3, label='1 random sample ')
ax[2].plot(D0, calcZ(Crnd)-Zdda, c='black', ls=':', lw=3, label='1 random sample')
SS_RGA = snowScatt.calcProperties(diameters=1.0e-3*mean_D,
                                  wavelength=wavelength,
                                  massScattering=mean_mass,
                                  properties='Leinonen15tabA00',
                                  temperature=263.15,
                                  Nangles=720)
_, _, _, ssCbck, _, _, ssMass, _, _ = SS_RGA
ax[1].plot(bin_center, ssCbck, lw=3, c='k', ls='--', label='SSRGA')
Zssrga = calcZ(ssCbck)-Zdda
ax[2].plot(D0, Zssrga, lw=3, c='k', ls='--', label='SSRGA')

ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_ylabel('C$_\mathrm{bck}$')
ax[1].set_xlabel('D$_\mathrm{max}$   [mm]')
ax[1].grid()

handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles[:3], labels[:3], ncol=1, loc=4)
ax[2].legend(handles[3:], labels[3:], ncol=1, loc=4)
ax[2].set_ylabel('bias = Z mean DDA - Z   [dB]')
ax[2].set_xscale('log')
ax[2].set_xlabel('$D_0$    [mm]')
ax[2].grid()

ax[0].text(0, 1.02, '(a)',  fontweight='black', transform=ax[0].transAxes)
ax[1].text(0, 1.02, '(b)',  fontweight='black', transform=ax[1].transAxes)
ax[2].text(0, 1.02, '(c)',  fontweight='black', transform=ax[2].transAxes)
ax[0].set_title('DB size distribution')
ax[1].set_title('Single particle C$_\mathrm{bck}$')
ax[2].set_title('Z$_e$ bias')



fig.savefig('random_samples.png', dpi=300)
fig.savefig('random_samples.pdf')