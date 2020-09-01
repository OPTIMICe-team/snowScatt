#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:37:31 2020

@author: dori
"""

import numpy as np
import pandas as pd
import snowScatt
import matplotlib.pyplot as plt

import datetime
begin = datetime.datetime.now()

from snowScatt.instrumentSimulator.radarMoments import Ze

def Nexp(D, lam):
    return np.exp(-lam*D)

def dB(x):
    return 10.0*np.log10(x)

frequency =  np.array([13.6e9, 35.6e9, 94.0e9]) # frequencies
temperature = 270.0
Nangles = 721

Dmax = np.linspace(0.1e-3, 20.0e-3, 1000) # list of sizes
lams = 1.0/np.linspace(0.01e-3, 11.0e-3, 100)

rime = ['00', '01', '02', '05', '10', '20']
particles = 'Leinonen15tabB'
fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for r in rime:
    particle = particles + r
    bck = pd.DataFrame(index=Dmax, columns=frequency)
    for fi, freq in enumerate(frequency):
        wl = snowScatt._compute._c/freq
        eps = snowScatt.refractiveIndex.water.eps(temperature, freq, 'Turner')
        K2 = snowScatt.refractiveIndex.utilities.K2(eps)
        ssCbck, ssvel = snowScatt.backscatVel(diameters=Dmax,
                                              wavelength=wl,
                                              properties=particle,
                                              temperature=temperature)
        bck[freq] = wl**4*ssCbck/(K2*np.pi**5)

    Zx = np.array([dB((1.0e18*bck.iloc[:, 0]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
    Zk = np.array([dB((1.0e18*bck.iloc[:, 1]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
    Zw = np.array([dB((1.0e18*bck.iloc[:, 2]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
    
    ax0.plot(Zk-Zw, Zx-Zk, label='L-B'+r)

    # check consistency of methods OK
    #wl = snowScatt._compute._c/frequency[0]
    #Zx0 = Ze(Dmax, PSD, wl, particle, temperature=temperature)
    #wl = snowScatt._compute._c/frequency[1]
    #Zk0 = Ze(Dmax, PSD, wl, particle, temperature=temperature)
    #wl = snowScatt._compute._c/frequency[2]
    #Zw0 = Ze(Dmax, PSD, wl, particle, temperature=temperature)
    
    #ax1.plot(Zk0-Zw0, Zx0-Zk0, label=r)
    
    #print('diffs ', (Zx-Zx0).sum(), (Zk-Zk0).sum(), (Zw-Zw0).sum())

particle = 'Ori_collColumns'
bck = pd.DataFrame(index=Dmax, columns=frequency)
for fi, freq in enumerate(frequency):
    wl = snowScatt._compute._c/freq
    eps = snowScatt.refractiveIndex.water.eps(temperature, freq, 'Turner')
    K2 = snowScatt.refractiveIndex.utilities.K2(eps)
    ssCbck, ssvel = snowScatt.backscatVel(diameters=Dmax,
                                          wavelength=wl,
                                          properties=particle,
                                          temperature=temperature)
    bck[freq] = wl**4*ssCbck/(K2*np.pi**5)
Zx = np.array([dB((1.0e18*bck.iloc[:, 0]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
Zk = np.array([dB((1.0e18*bck.iloc[:, 1]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
Zw = np.array([dB((1.0e18*bck.iloc[:, 2]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
ax0.plot(Zk-Zw, Zx-Zk, label='Ori-table', lw=4)

particle = 'Oea14'
bck = pd.DataFrame(index=Dmax, columns=frequency)
for fi, freq in enumerate(frequency):
    wl = snowScatt._compute._c/freq
    eps = snowScatt.refractiveIndex.water.eps(temperature, freq, 'Turner')
    K2 = snowScatt.refractiveIndex.utilities.K2(eps)
    ssCbck, ssvel = snowScatt.backscatVel(diameters=Dmax,
                                          wavelength=wl,
                                          properties=particle,
                                          temperature=temperature)
    bck[freq] = wl**4*ssCbck/(K2*np.pi**5)
Zx = np.array([dB((1.0e18*bck.iloc[:, 0]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
Zk = np.array([dB((1.0e18*bck.iloc[:, 1]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
Zw = np.array([dB((1.0e18*bck.iloc[:, 2]*Nexp(Dmax, l)*np.gradient(Dmax)).sum()) for l in lams ])
ax0.plot(Zk-Zw, Zx-Zk, label='Ori-constant', lw=4)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('OriVSOri.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)