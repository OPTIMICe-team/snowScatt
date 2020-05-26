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

import logging
import pandas as pd
import numpy as np
from os import path
from scipy import interp
from snowScatt.fallSpeed import Boehm1992
from snowScatt.fallSpeed import Boehm1989
from snowScatt.fallSpeed import HeymsfieldWestbrook2010
from snowScatt.fallSpeed import KhvorostyanovCurry2005
from snowScatt._constants import _ice_density

module_path = path.split(path.abspath(__file__))[0]

fallspeeds = {'HeymsfieldWestbrook10':HeymsfieldWestbrook2010,
			  'KhvorostyanovCurry05':KhvorostyanovCurry2005,
			  'Boehm92':Boehm1992,
			  'Boehm89':Boehm1989,
			 }


def _interpolate_coeff(D, table, ar_mono=1.0):
    beta = interp(D, table.index.values, table.beta)
    gamma = interp(D, table.index.values, table.gamma)
    kappa = interp(D, table.index.values, table.kappa)
    zeta1 = interp(D, table.index.values, table.zeta)
    alpha_eff = interp(D, table.index.values, table.alpha_eff)
    ar_mono = np.ones_like(D)*ar_mono#interp(D, table.index.values, table.ar_mono)
    # TODO evaluate to shift these to another function
    mass = interp(D, table.index.values, table.mass, left=np.nan, right=np.nan)
    vel = interp(D, table.index.values, table.vel_Bohm, left=np.nan, right=np.nan)
    area = interp(D, table.index.values, table.area, left=np.nan, right=np.nan)

    return beta, gamma, kappa, zeta1, alpha_eff, ar_mono, mass, vel, area


## Library of average parameters
snowLib = {}
libKeys = ['kappa', 'beta', 'gamma', 'zeta1', 'aspect', 'ar_mono', 'am', 'bm', 'aa', 'ba', 'msg']

# Mean parameters from Heymsfield Westbrook 2014 aggregates of bullet rosettes
HW14 = {'kappa': 0.19, 'beta': 0.23, 'gamma': 5./3., 'zeta1': 1.,
        'aspect': 0.6, 'am':0.015, 'bm':2.08, 'aa':np.nan, 'ba':np.nan,
        'ar_mono': 1.0, # Could it be that I have to consider ar of bullets not rosettes?
        'msg': 'Heymsfield and Westbrook 2014 aggregates of bullett rosettes'}
snowLib['HW14'] = HW14

# Mean parameters derived for Leinonen-Szyrmer 2015 unrimed snow aggregates
L15_0 = {'kappa': 0.189177, 'beta': 3.06939,
         'gamma': 2.53192, 'zeta1': 0.0709529,
         'aspect': 0.6, 'am':0.015, 'bm':2.08, 'aa':np.nan, 'ba':np.nan,
         'ar_mono': 0.3,
         'msg': 'Leinonen 2015 unrimed aggregates of dendrites'}
snowLib['LS15A0.0'] = L15_0

# Mean parameters for Ori et al. 2014 unrimed assemblages of ice columns
Oea14 = {'kappa': 0.190031, 'beta': 0.030681461,
         'gamma': 1.3002167, 'zeta1': 0.29466184,
         'aspect': 0.6, 'am':0.015, 'bm':2.08, 'aa':np.nan, 'ba':np.nan,
         'ar_mono': 5.0,
         'msg':'Ori 2014 assemblages of columns'}
snowLib['Oea14'] = Oea14

## Library of tabulated files
snowList = {}
fileKeys = ['path', 'msg']
# Leinonen 2015 Table for unrimed snowflakes
L15tabA00 = {'path':module_path+'/ssrga_coeffs_simultaneous_0.0.csv',
             'msg':'Table of Leinonen unrimed snowflakes'}
snowList['Leinonen15tabA00'] = L15tabA00

# Leinonen 2015 Table for rimed snowflakes ELWP=0.1 model A simultaneous
L15tabA01 = {'path':module_path+'/ssrga_coeffs_simultaneous_0.1.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=0.1 model A simultaneous'}
snowList['Leinonen15tabA01'] = L15tabA01

# Leinonen 2015 Table for rimed snowflakes ELWP=0.2 model A simultaneous
L15tabA02 = {'path':module_path+'/ssrga_coeffs_simultaneous_0.2.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=0.2 model A simultaneous'}
snowList['Leinonen15tabA02'] = L15tabA02

# Leinonen 2015 Table for rimed snowflakes ELWP=0.5 model A simultaneous
L15tabA05 = {'path':module_path+'/ssrga_coeffs_simultaneous_0.5.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=0.5 model A simultaneous'}
snowList['Leinonen15tabA05'] = L15tabA05

# Leinonen 2015 Table for rimed snowflakes ELWP=1.0 model A simultaneous
L15tabA10 = {'path':module_path+'/ssrga_coeffs_simultaneous_1.0.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=1.0 model A simultaneous'}
snowList['Leinonen15tabA10'] = L15tabA10

# Leinonen 2015 Table for rimed snowflakes ELWP=2.0 model A simultaneous
L15tabA20 = {'path':module_path+'/ssrga_coeffs_simultaneous_2.0.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=2.0 model A simultaneous'}
snowList['Leinonen15tabA20'] = L15tabA20

# Leinonen 2015 Table for rimed snowflakes ELWP=0.1 model B subsequent
L15tabB01 = {'path':module_path+'/ssrga_coeffs_subsequent_0.1.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=0.1 model B subsequent'}
snowList['Leinonen15tabB01'] = L15tabB01

# Leinonen 2015 Table for rimed snowflakes ELWP=0.2 model B subsequent
L15tabB02 = {'path':module_path+'/ssrga_coeffs_subsequent_0.2.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=0.2 model B subsequent'}
snowList['Leinonen15tabB02'] = L15tabB02

# Leinonen 2015 Table for rimed snowflakes ELWP=0.5 model B subsequent
L15tabB05 = {'path':module_path+'/ssrga_coeffs_subsequent_0.5.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=0.5 model B subsequent'}
snowList['Leinonen15tabB05'] = L15tabB05

# Leinonen 2015 Table for rimed snowflakes ELWP=1.0 model B subsequent
L15tabB10 = {'path':module_path+'/ssrga_coeffs_subsequent_1.0.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=1.0 model B subsequent'}
snowList['Leinonen15tabB10'] = L15tabB10

# Leinonen 2015 Table for rimed snowflakes ELWP=2.0 model B subsequent
L15tabB20 = {'path':module_path+'/ssrga_coeffs_subsequent_2.0.csv',
             'msg':'Table of Leinonen rimed snowflakes ELWP=2.0 model B subsequent'}
snowList['Leinonen15tabB20'] = L15tabB20

# Leinonen 2015 Table for rimed snowflakes model C rimeonly
L15tabC = {'path':module_path+'/ssrga_coeffs_rimec.csv',
           'msg':'Table of Leinonen rime only graupel'}
snowList['Leinonen15tabC'] = L15tabC

# Leonie von Terzi Table for unrimed assmblages of dendrites
LvTdendrite = {'path':module_path+'/ssrga_coeffs_dendrite.csv',
               'msg':'Table of von Terzi assemblages of dendrites'}
snowList['vonTerzi_dendrite'] = LvTdendrite

# Leonie von Terzi Table for unrimed assmblages of columns
LvTcolumn = {'path':module_path+'/ssrga_coeffs_column.csv',
             'msg':'Table of von Terzi assemblages of columns'}
snowList['vonTerzi_column'] = LvTcolumn

# Leonie von Terzi Table for unrimed assmblages of plates
LvTplate = {'path':module_path+'/ssrga_coeffs_plate.csv',
            'msg':'Table of von Terzi assemblages of plates'}
snowList['vonTerzi_plate'] = LvTplate

# Leonie von Terzi Table for unrimed assmblages of needles
LvTneedle = {'path':module_path+'/ssrga_coeffs_needle.csv',
             'msg':'Table of von Terzi assemblages of needles'}
snowList['vonTerzi_needle'] = LvTneedle

# Leonie von Terzi Table for unrimed assmblages of mixtures of columns and dendrites
LvTmixcoldend = {'path':module_path+'/ssrga_coeffs_mixcolumndend.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites'}
snowList['vonTerzi_mixcoldend'] = LvTmixcoldend


## Class to manage the library of snow properties
class snowProperties():
	def __init__(self):
		self._library = snowLib
		self._fileList = snowList
		logging.info('Initialize a library of snow properties')

	def __call__(self, diameters, identifier, velocity_model='Boehm92'):
		logging.debug('return the snow properties for selected sizes')
		if identifier in self._library.keys():
			logging.debug('got AVG ', identifier, self._library[identifier])
			kappa = np.ones_like(diameters)*self._library[identifier]['kappa']
			beta = np.ones_like(diameters)*self._library[identifier]['beta']
			gamma = np.ones_like(diameters)*self._library[identifier]['gamma']
			zeta1 = np.ones_like(diameters)*self._library[identifier]['zeta1']
			alpha_eff = np.ones_like(diameters)*self._library[identifier]['aspect']
			ar_mono = np.ones_like(diameters)*self._library[identifier]['ar_mono']
			mass = self._library[identifier]['am']*diameters**self._library[identifier]['bm']
			# vel = self._library[identifier]['av']*diameters**self._library[identifier]['bv']
			area = self._library[identifier]['aa']*diameters**self._library[identifier]['ba']

		elif identifier in self._fileList.keys():
			logging.debug('got TABLE ', identifier, self._fileList[identifier])
			with open(self._fileList[identifier]['path']) as f:
				line = f.read().split('#')[-1].split('\n')[0]
				am = float(line.split('am=')[-1].split(',')[0])
				bm = float(line.split('bm=')[-1].split(',')[0])
				#av = float(line.split('av=')[-1].split(',')[0])
				#bv = float(line.split('bv=')[-1].split(',')[0])
				aa = float(line.split('aa=')[-1].split(',')[0])
				ba = float(line.split('ba=')[-1].split(',')[0])
				ar_mono = float(line.split('monomer_alpha=')[-1].split(',')[0])
			table = pd.read_csv(self._fileList[identifier]['path'], comment='#').set_index('Diam_max').interpolate(limit_direction='both') # set index and fill nans inside

			beta, gamma, kappa, zeta1, alpha_eff, ar_mono, mass, vel, area = _interpolate_coeff(diameters, table, ar_mono)
			mass = am*diameters**bm
			area = aa*diameters**ba

		else:
			raise AttributeError('I do not know ', identifier, 'call info() for a list of available properties\n')
		
		# Limit density to solid ice sphere and area to full disk
		a_disk = 0.25*np.pi*diameters**2
		m_solid = 2.0*a_disk*diameters*_ice_density/3.0
		area = np.minimum(a_disk, area)
		mass = np.minimum(m_solid, mass)

		vel = fallspeeds[velocity_model](diameters, mass, area) # TODO also need to pass kwargs additional

		return np.asarray(kappa), np.asarray(beta), np.asarray(gamma), np.asarray(zeta1), np.asarray(alpha_eff), np.asarray(ar_mono), np.asarray(mass), np.asarray(vel), np.asarray(area)


	def add_snow(self, label, newSnowDict):
		if all (key in newSnowDict for key in (libKeys)):
			logging.info('temporary add to the internal database a new AVERAGE property')
			self._library[label] = newSnowDict
		elif all (key in newSnowDict for key in (fileKeys)):
			logging.info('temporary add to the internal database a new TABLE property')
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


# Create the Library
snowLibrary = snowProperties()