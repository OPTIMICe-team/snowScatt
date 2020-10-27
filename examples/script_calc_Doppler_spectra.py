#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:37:31 2020

@author: dori

This script uses snowScatt to define the scattering properties of a series of
particle types including Leinonen model A with varius riming degree 
It also uses snowScatt to calculate the particle fallspeed and uses the
backscattering and fallspeed coherent information to estimate the Doppler
spectrum

It is mainly of interest for those who want to calculate radar Doppler spectrum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
begin = datetime.datetime.now()

#import logging
#logging.basicConfig(level=logging.DEBUG)
import snowScatt

from snowScatt.instrumentSimulator.radarMoments import Ze
from snowScatt.instrumentSimulator.radarSpectrum import dopplerSpectrum
from snowScatt.instrumentSimulator.radarSpectrum import sizeSpectrum


def Nexp(D, lam):
    return np.exp(-lam*D)

def dB(x):
    return 10.0*np.log10(x)

def Bd(x):
    return 10.0**(0.1*x)

frequency =  np.array([13.6e9, 35.6e9, 94.0e9]) # frequencies
temperature = 270.0
Nangles = 721

Dmax = np.linspace(0.1e-3, 80.0e-3, 1000) # list of sizes
lams = 1.0/np.linspace(0.1e-3, 4.0e-3, 10) # list of lambdas

#rime = ['00', '01', '02', '05', '10', '20']
rime = ['00']
particles = 'Leinonen15tabA'
fig, (ax0, ax1) = plt.subplots(1, 2)
# 
PSD = 10.0*np.stack([np.array(Nexp(Dmax, l)) for l in lams])
for r in rime:
    particle = particles + r
    wl = snowScatt._compute._c/frequency[2]
    spec0, vel = dopplerSpectrum(Dmax, PSD, wl, particle,
                                 temperature=temperature)
    spec1 = sizeSpectrum(Dmax, PSD, wl, particle, temperature=temperature)
    Zx = Ze(Dmax, PSD, wl, particle, temperature=temperature)

ax0.plot(vel, dB(spec0.T))
ax1.plot(Dmax*1.0e3, dB(spec1.T))
for ax in (ax0, ax1):
    ax.legend()
    ax.grid()
    ax.set_ylabel('spectral power')
    ax.set_ylim([-50, 0])
ax0.set_xlabel('velocity [m/s]')
ax1.set_xlabel('size   [mm]')
fig.tight_layout()
print('Execution ', datetime.datetime.now()-begin)

Z0 = dB(np.nansum((spec0*np.gradient(vel)), axis=-1))
Z1 = dB(np.nansum((spec1*np.gradient(Dmax)), axis=-1))