#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:00:59 2023

@author: dori
"""

import numpy as np
import pandas as pd
import snowScatt
import matplotlib.pyplot as plt

import datetime
begin = datetime.datetime.now()

def Nexp(D, lam):
    return np.exp(-lam*D)

def dB(x):
    return 10.0*np.log10(x)

frequency =  np.array([13.6e9, 35.6e9, 94.0e9]) # frequencies
temperature = 270.0
Nangles = 721

Dmax = np.linspace(0.1e-3, 20.0e-3, 1000) # list of sizes
lams = 1.0/np.linspace(0.01e-3, 11.0e-3, 100)

particles = [p for p in snowScatt.snowProperties._readProperties.snowList.keys() if 'Maherndl_columns' in p]

colors = plt.cm.turbo(np.linspace(0,1,len(particles)))

fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for ip, particle in enumerate(particles):
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
    
    ax0.plot(Zk-Zw, Zx-Zk, label=particle, c=colors[ip], lw=2)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('MNcolumns.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)

begin = datetime.datetime.now()
particles = [p for p in snowScatt.snowProperties._readProperties.snowList.keys() if 'Maherndl_needles' in p]

colors = plt.cm.turbo(np.linspace(0,1,len(particles)))

fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for ip, particle in enumerate(particles):
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
    
    ax0.plot(Zk-Zw, Zx-Zk, label=particle, c=colors[ip], lw=2)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('MNneedles.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)

begin = datetime.datetime.now()
particles = [p for p in snowScatt.snowProperties._readProperties.snowList.keys() if 'Maherndl_dendrites' in p]

colors = plt.cm.turbo(np.linspace(0,1,len(particles)))

fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for ip, particle in enumerate(particles):
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
    
    ax0.plot(Zk-Zw, Zx-Zk, label=particle, c=colors[ip], lw=2)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('MNdendrites.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)

begin = datetime.datetime.now()
particles = [p for p in snowScatt.snowProperties._readProperties.snowList.keys() if 'Maherndl_plates' in p]

colors = plt.cm.turbo(np.linspace(0,1,len(particles)))

fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for ip, particle in enumerate(particles):
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
    
    ax0.plot(Zk-Zw, Zx-Zk, label=particle, c=colors[ip], lw=2)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('MNplates.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)

begin = datetime.datetime.now()
particles = [p for p in snowScatt.snowProperties._readProperties.snowList.keys() if 'Maherndl_mixtures' in p]

colors = plt.cm.turbo(np.linspace(0,1,len(particles)))

fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for ip, particle in enumerate(particles):
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
    
    ax0.plot(Zk-Zw, Zx-Zk, label=particle, c=colors[ip], lw=2)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('MNmixtures.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)

begin = datetime.datetime.now()
particles = [p for p in snowScatt.snowProperties._readProperties.snowList.keys() if 'Maherndl_rosettes' in p]

colors = plt.cm.turbo(np.linspace(0,1,len(particles)))

fig, (ax0) = plt.subplots(1, 1)
# 
PSD = np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for ip, particle in enumerate(particles):
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
    
    ax0.plot(Zk-Zw, Zx-Zk, label=particle, c=colors[ip], lw=2)

for ax in [ax0,]:
    ax.legend()
    ax.grid()
    ax.set_xlabel('Ka - W   [dB]')
    ax.set_ylabel('Ku - Ka  [dB]')
fig.savefig('MNrosettes.png', dpi=300)
print('Execution ', datetime.datetime.now()-begin)