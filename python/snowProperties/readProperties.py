#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2020 Davide Ori 
University of Cologne

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import pandas as pd
import numpy as np
from os import path
from scipy import interp

module_path = path.split(path.abspath(__file__))[0]


def interpolate_coeff(D, table):
    beta = interp(D, table.index.values, table.beta_z)
    gamma = interp(D, table.index.values, table.gamma_z) # TODO fix the name of the columns
    kappa = interp(D, table.index.values, table.kappa_z)
    zeta1 = interp(D, table.index.values, table.zeta1_z)
    alpha_eff = interp(D, table.index.values, table.alpha_eff)
    ar_mono = interp(D, table.index.values, table.ar_mono)
    mass = interp(D, table.index.values, table.m)
    vel = interp(D, table.index.values, table.vel)
    return beta, -gamma, kappa, zeta1, alpha_eff, ar_mono, mass, vel # TODO fix sign of gamma

## Library of average parameters
snowLib = {}
libKeys = ['kappa', 'beta', 'gamma', 'zeta1', 'aspect', 'ar_mono', 'am', 'bm', 'av', 'bv', 'msg']

# Mean parameters from Heymsfield Westbrook 2014 aggregates of bullet rosettes
HW14 = {'kappa': 0.19, 'beta': 0.23, 'gamma': 5./3., 'zeta1': 1.,
        'aspect': 0.6, 'am':0.015, 'bm':2.08, 'av':3.581, 'bv':0.3,
        'ar_mono': 1.0, # Could it be that I have to consider ar of bullets not rosettes?
        'msg': 'Heymsfield and Westbrook 2014 aggregates of bullett rosettes'}
snowLib['HW14'] = HW14

# Mean parameters derived for Leinonen-Szyrmer 2015 unrimed snow aggregates
L15_0 = {'kappa': 0.189177, 'beta': 3.06939,
         'gamma': 2.53192, 'zeta1': 0.0709529,
         'aspect': 0.6, 'am':0.015, 'bm':2.08, 'av':3.581, 'bv':0.3,
         'ar_mono': 0.3,
         'msg': 'Leinonen 2015 unrimed aggregates of dendrites'}
snowLib['LS15A0.0'] = L15_0

# Mean parameters for Ori et al. 2014 unrimed assemblages of ice columns
Oea14 = {'kappa': 0.190031, 'beta': 0.030681461,
         'gamma': 1.3002167, 'zeta1': 0.29466184,
         'aspect': 0.6, 'am':0.015, 'bm':2.08, 'av':3.581, 'bv':0.3,
         'ar_mono': 5.0,
         'msg':'Ori 2014 assemblages of columns'}
snowLib['Oea14'] = Oea14

## Library of tabulated files
snowList = {}
fileKeys = ['path', 'msg']
# Leinonen 2015 Table for unrimed snowflakes
L15tab00 = {'path':module_path+'/Jussi_0.0_fake.dat',
            'msg':'Table of Leinonen unrimed snowflakes'}
snowList['Leinonen15tab00'] = L15tab00

# Leinonen 2015 Table for rimed snowflakes ELWP=0.1
L15tab01 = {'path':module_path+'/Jussi_0.1_fake.dat',
 'msg':'Table of Leinonen rimed snowflakes ELWP=0.1'}
snowList['Leinonen15tab01'] = L15tab01

# Leinonen 2015 Table for rimed snowflakes ELWP=0.2
L15tab02 = {'path':module_path+'/Jussi_0.2_fake.dat',
 'msg':'Table of Leinonen rimed snowflakes ELWP=0.2'}
snowList['Leinonen15tab02'] = L15tab02

# Leinonen 2015 Table for rimed snowflakes ELWP=0.5
L15tab05 = {'path':module_path+'/Jussi_0.5_fake.dat',
 'msg':'Table of Leinonen rimed snowflakes ELWP=0.5'}
snowList['Leinonen15tab05'] = L15tab05

# Leinonen 2015 Table for rimed snowflakes ELWP=1.0
L15tab10 = {'path':module_path+'/Jussi_1.0_fake.dat',
 'msg':'Table of Leinonen rimed snowflakes ELWP=1.0'}
snowList['Leinonen15tab10'] = L15tab10

# Leinonen 2015 Table for rimed snowflakes ELWP=2.0
L15tab20 = {'path':module_path+'/Jussi_2.0_fake.dat',
 'msg':'Table of Leinonen rimed snowflakes ELWP=2.0'}
snowList['Leinonen15tab20'] = L15tab20


## Class to manage the library of snow properties

class snowProperties():
	def __init__(self):
		self._library = snowLib
		self._fileList = snowList
		print('Initialize a library of snow properties')

	def __call__(self, diameters, identifier):
		print('return the snow properties for selected sizes')
		if identifier in self._library.keys():
			print('got AVG ', identifier, self._library[identifier])
			kappa = np.ones_like(diameters)*self._library[identifier]['kappa']
			beta = np.ones_like(diameters)*self._library[identifier]['beta']
			gamma = np.ones_like(diameters)*self._library[identifier]['gamma']
			zeta1 = np.ones_like(diameters)*self._library[identifier]['zeta1']
			alpha_eff = np.ones_like(diameters)*self._library[identifier]['aspect']
			ar_mono = np.ones_like(diameters)*self._library[identifier]['ar_mono']
			mass = self._library[identifier]['am']*diameters**self._library[identifier]['bm']
			vel = self._library[identifier]['av']*diameters**self._library[identifier]['bv']

		elif identifier in self._fileList.keys():
			print('got TABLE ', identifier, self._fileList[identifier])
			table = pd.read_csv(self._fileList[identifier]['path']).set_index('D')
			beta, gamma, kappa, zeta1, alpha_eff, ar_mono, mass, vel = interpolate_coeff(diameters, table)
		else:
			raise AttributeError('I do not know ', identifier, 'call info() for a list of available properties\n')
		return kappa, beta, gamma, zeta1, alpha_eff, ar_mono, mass, vel

	def add_snow(self, label, newSnowDict):
		if all (key in newSnowDict for key in (libKeys)):
			print('temporary add to the internal database a new AVERAGE property')
			self._library[label] = newSnowDict
		elif all (key in newSnowDict for key in (fileKeys)):
			print('temporary add to the internal database a new TABLE property')
			self._fileList[label] = newSnowDict
		else:
			raise AttributeError('I do not recognize the newSnowDict attribute.\n You have to either pass a new dictionary of average properties (keys: {}) or a dictionary for a table file (keys: {}).\n'.format(libKeys, fileKeys))

	def info(self, section='all'):
		print('print the information content in the database\n')
		if section in ['all', 'avg']:
			print('##  List of AVERAGE properties\n')
			for i in self._library.keys():
				print('Name: ', i)
				print({k:self._library[i][k] for k in libKeys[:-1]})
				print(self._library[i]['msg']+'\n')
		if section in ['all', 'tables']:
			print('##  List of tabulated size resolved properties\n')
			for i in self._fileList.keys():
				print('Name: ', i)
				print(self._fileList[i]['msg'])
				print('Filepath :', self._fileList[i]['path']+'\n')

		print('\n\n######################################\nThis is the content of '+section+' sections of the database')
		print('You can pass the argument section=["all", "avg", "tables"] to restrict the output to a certain section')

snowLibrary = snowProperties()