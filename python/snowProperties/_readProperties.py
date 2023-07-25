#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Copyright (C) 2020 Davide Ori 
#University of Cologne

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.



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
    vel = interp(D, table.index.values, table.vel_Bohm, left=np.nan, right=np.nan)  # vel_Bohm needs to be there
    area = interp(D, table.index.values, table.area, left=np.nan, right=np.nan)

    return beta, gamma, kappa, zeta1, alpha_eff, ar_mono, mass, vel, area

################################################################################################################
## Library of average parameters
################################################################################################################

snowLib = {}
libKeys = ['kappa', 'beta', 'gamma', 'zeta1', 'aspect', 'ar_mono', 'am', 'bm', 'aa', 'ba', 'msg']

# Mean parameters from Heymsfield Westbrook 2014 aggregates of bullet rosettes
HW14 = {'kappa': 0.19, 'beta': 0.23, 'gamma': 5./3., 'zeta1': 1.,
        'aspect': 0.6, 'am':0.015, 'bm':2.08, 'aa':np.nan, 'ba':np.nan,
        'ar_mono': 1.0, # Could it be that I have to consider ar of bullets not rosettes?
        'msg': 'Hogan and Westbrook 2014 aggregates of bullett rosettes'}
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
         'aspect': 0.9, 'am':0.157, 'bm':2.1, 'aa':np.nan, 'ba':np.nan,
         'ar_mono': 5.0,
         'msg':'Ori 2014 assemblages of columns'}
snowLib['Oea14'] = Oea14

################################################################################################################
## Library of tabulated files
################################################################################################################

snowList = {}
fileKeys = ['path', 'msg']

################################################################################################################
# Leinonen rimed aggregates Leinonen Szyrmer 2015

# Leinonen 2015 Table for unrimed snowflakes
L15tabA00 = {'path':module_path+'/ssrga_coeffs_simultaneous_0.0.csv',
             'msg':'Table of Leinonen unrimed snowflakes'}
snowList['Leinonen15tabA00'] = L15tabA00
snowList['Leinonen15tabB00'] = L15tabA00 # make a B00 entry that points to the same

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

################################################################################################################
# Uni Cologne "particle zoo" mainly Leonie von Terzi

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

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_01 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.1.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.1'}
snowList['vonTerzi_mixcoldend_rimed01'] = LvTmixcoldend_01

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_02 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.2.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.2'}
snowList['vonTerzi_mixcoldend_rimed02'] = LvTmixcoldend_02

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_03 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.3.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.3'}
snowList['vonTerzi_mixcoldend_rimed03'] = LvTmixcoldend_03

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_04 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.4.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.4'}
snowList['vonTerzi_mixcoldend_rimed04'] = LvTmixcoldend_04

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_05 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.5.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.5'}
snowList['vonTerzi_mixcoldend_rimed05'] = LvTmixcoldend_05

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_06 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.6.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.6'}
snowList['vonTerzi_mixcoldend_rimed06'] = LvTmixcoldend_06

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_07 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.7.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.7'}
snowList['vonTerzi_mixcoldend_rimed07'] = LvTmixcoldend_07

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_08 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.8.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.8'}
snowList['vonTerzi_mixcoldend_rimed08'] = LvTmixcoldend_08

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_09 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_0.9.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 0.9'}
snowList['vonTerzi_mixcoldend_rimed09'] = LvTmixcoldend_09

# Leonie von Terzi Table for rimed assemblages of mixtures of columns and dendrites
LvTmixcoldend_10 = {'path':module_path+'/ssrga_coeffs_mixcolumndend_rimed_1.0.csv',
           'msg':'Table of von Terzi assemblages of mixtures of columns and dendrites with rimeelwp 1.0'}
snowList['vonTerzi_mixcoldend_rimed10'] = LvTmixcoldend_10

################################################################################################################
# Davide Ori Table of unrimed aggregates obtained from a continuous collection of columnar crystals
DOcollColumns = {'path':module_path+'/ssrga_coeffs_Ori_collection_columns.csv',
                 'msg':'Table of Davide Ori aggregates of collection of columns'}
snowList['Ori_collColumns'] = DOcollColumns

################################################################################################################
# Nina Maherndl - parametrizing SSRGA with respect to normalized rime fraction

# Nina Maherndl rimed aggregates of column (M=0 unrimed) QJRMS 2023
NMcolM00000 = {'path':module_path+'/ssrga_coeffs_column_M_0p00.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0'}
snowList['Maherndl_columnsM00000'] = NMcolM00000

# Nina Maherndl rimed aggregates of column (M=0.0129) QJRMS 2023
NMcolM00129 = {'path':module_path+'/ssrga_coeffs_column_M_0p0129.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.0129'}
snowList['Maherndl_columnsM00129'] = NMcolM00129

# Nina Maherndl rimed aggregates of column (M=0.0205) QJRMS 2023
NMcolM00205 = {'path':module_path+'/ssrga_coeffs_column_M_0p0205.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.0205'}
snowList['Maherndl_columnsM00205'] = NMcolM00205

# Nina Maherndl rimed aggregates of column (M=0.0324) QJRMS 2023
NMcolM00324 = {'path':module_path+'/ssrga_coeffs_column_M_0p0324.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.0324'}
snowList['Maherndl_columnsM00324'] = NMcolM00324

# Nina Maherndl rimed aggregates of column (M=0.0514) QJRMS 2023
NMcolM00514 = {'path':module_path+'/ssrga_coeffs_column_M_0p0514.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.0514'}
snowList['Maherndl_columnsM00514'] = NMcolM00514

# Nina Maherndl rimed aggregates of column (M=0.0816) QJRMS 2023
NMcolM00816 = {'path':module_path+'/ssrga_coeffs_column_M_0p0816.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.0816'}
snowList['Maherndl_columnsM00816'] = NMcolM00816

# Nina Maherndl rimed aggregates of column (M=0.1290) QJRMS 2023
NMcolM01290 = {'path':module_path+'/ssrga_coeffs_column_M_0p1290.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.1290'}
snowList['Maherndl_columnsM01290'] = NMcolM01290

# Nina Maherndl rimed aggregates of column (M=0.2045) QJRMS 2023
NMcolM02045 = {'path':module_path+'/ssrga_coeffs_column_M_0p2045.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.2045'}
snowList['Maherndl_columnsM02045'] = NMcolM02045

# Nina Maherndl rimed aggregates of column (M=0.3245) QJRMS 2023
NMcolM03245 = {'path':module_path+'/ssrga_coeffs_column_M_0p3245.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.3245'}
snowList['Maherndl_columnsM03245'] = NMcolM03245

# Nina Maherndl rimed aggregates of column (M=0.5145) QJRMS 2023
NMcolM05145 = {'path':module_path+'/ssrga_coeffs_column_M_0p5145.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.5145'}
snowList['Maherndl_columnsM05145'] = NMcolM05145

# Nina Maherndl rimed aggregates of column (M=0.8155) QJRMS 2023
NMcolM08155 = {'path':module_path+'/ssrga_coeffs_column_M_0p8155.csv',
               'msg':'Table of Nina Maherndl aggregates of columns M=0.8155'}
snowList['Maherndl_columnsM08155'] = NMcolM08155

#############################################################################

# Nina Maherndl rimed aggregates of needles (M=0 unrimed) QJRMS 2023
NMnedM00000 = {'path':module_path+'/ssrga_coeffs_needle_M_0p00.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.0000'}
snowList['Maherndl_needlesM00000'] = NMnedM00000

# Nina Maherndl rimed aggregates of needles (M=0.0129) QJRMS 2023
NMnedM00129 = {'path':module_path+'/ssrga_coeffs_needle_M_0p0129.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.0129'}
snowList['Maherndl_needlesM00129'] = NMnedM00129

# Nina Maherndl rimed aggregates of needles (M=0.0205) QJRMS 2023
NMnedM00205 = {'path':module_path+'/ssrga_coeffs_needle_M_0p0205.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.0205'}
snowList['Maherndl_needlesM00205'] = NMnedM00205

# Nina Maherndl rimed aggregates of needles (M=0.0324) QJRMS 2023
NMnedM00324 = {'path':module_path+'/ssrga_coeffs_needle_M_0p0324.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.0324'}
snowList['Maherndl_needlesM00324'] = NMnedM00324

# Nina Maherndl rimed aggregates of needles (M=0.0514) QJRMS 2023
NMnedM00514 = {'path':module_path+'/ssrga_coeffs_needle_M_0p0514.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.0514'}
snowList['Maherndl_needlesM00514'] = NMnedM00514

# Nina Maherndl rimed aggregates of needles (M=0.0816) QJRMS 2023
NMnedM00816 = {'path':module_path+'/ssrga_coeffs_needle_M_0p0816.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.0816'}
snowList['Maherndl_needlesM00816'] = NMnedM00816

# Nina Maherndl rimed aggregates of needles (M=0.1290) QJRMS 2023
NMnedM01290 = {'path':module_path+'/ssrga_coeffs_needle_M_0p1290.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.1290'}
snowList['Maherndl_needlesM01290'] = NMnedM01290

# Nina Maherndl rimed aggregates of needles (M=0.2045) QJRMS 2023
NMnedM02045 = {'path':module_path+'/ssrga_coeffs_needle_M_0p2045.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.2045'}
snowList['Maherndl_needlesM02045'] = NMnedM02045

# Nina Maherndl rimed aggregates of needles (M=0.3245) QJRMS 2023
NMnedM03245 = {'path':module_path+'/ssrga_coeffs_needle_M_0p3245.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.3245'}
snowList['Maherndl_needlesM03245'] = NMnedM03245

# Nina Maherndl rimed aggregates of needles (M=0.5145) QJRMS 2023
NMnedM05145 = {'path':module_path+'/ssrga_coeffs_needle_M_0p5145.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.5145'}
snowList['Maherndl_needlesM05145'] = NMnedM05145

# Nina Maherndl rimed aggregates of needles (M=0.8155) QJRMS 2023
NMnedM08155 = {'path':module_path+'/ssrga_coeffs_needle_M_0p8155.csv',
               'msg':'Table of Nina Maherndl aggregates of needles M=0.8155'}
snowList['Maherndl_needlesM08155'] = NMnedM08155

#############################################################################

# Nina Maherndl rimed aggregates of dendrites (M=0 unrimed) QJRMS 2023
NMdenM00000 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p00.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.0000'}
snowList['Maherndl_dendritesM00000'] = NMdenM00000

# Nina Maherndl rimed aggregates of dendrites (M=0.0129) QJRMS 2023
NMdenM00129 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p0129.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.0129'}
snowList['Maherndl_dendritesM00129'] = NMdenM00129

# Nina Maherndl rimed aggregates of dendrites (M=0.0205) QJRMS 2023
NMdenM00205 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p0205.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.0205'}
snowList['Maherndl_dendritesM00205'] = NMdenM00205

# Nina Maherndl rimed aggregates of dendrites (M=0.0324) QJRMS 2023
NMdenM00324 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p0324.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.0324'}
snowList['Maherndl_dendritesM00324'] = NMdenM00324

# Nina Maherndl rimed aggregates of dendrites (M=0.0514) QJRMS 2023
NMdenM00514 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p0514.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.0514'}
snowList['Maherndl_dendritesM00514'] = NMdenM00514

# Nina Maherndl rimed aggregates of dendrites (M=0.0816) QJRMS 2023
NMdenM00816 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p0816.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.0816'}
snowList['Maherndl_dendritesM00816'] = NMdenM00816

# Nina Maherndl rimed aggregates of dendrites (M=0.1290) QJRMS 2023
NMdenM01290 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p1290.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.1290'}
snowList['Maherndl_dendritesM01290'] = NMdenM01290

# Nina Maherndl rimed aggregates of dendrites (M=0.2045) QJRMS 2023
NMdenM02045 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p2045.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.2045'}
snowList['Maherndl_dendritesM02045'] = NMdenM02045

# Nina Maherndl rimed aggregates of dendrites (M=0.3245) QJRMS 2023
NMdenM03245 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p3245.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.3245'}
snowList['Maherndl_dendritesM03245'] = NMdenM03245

# Nina Maherndl rimed aggregates of dendrites (M=0.5145) QJRMS 2023
NMdenM05145 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p5145.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.5145'}
snowList['Maherndl_dendritesM05145'] = NMdenM05145

# Nina Maherndl rimed aggregates of dendrites (M=0.8155) QJRMS 2023
NMdenM08155 = {'path':module_path+'/ssrga_coeffs_dendrite_M_0p8155.csv',
               'msg':'Table of Nina Maherndl aggregates of dendrites M=0.8155'}
snowList['Maherndl_dendritesM08155'] = NMdenM08155

#############################################################################


# Nina Maherndl rimed aggregates of plates (M=0 unrimed) QJRMS 2023
NMplaM00000 = {'path':module_path+'/ssrga_coeffs_plate_M_0p00.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.0000'}
snowList['Maherndl_platesM00000'] = NMplaM00000

# Nina Maherndl rimed aggregates of plates (M=0.0129) QJRMS 2023
NMplaM00129 = {'path':module_path+'/ssrga_coeffs_plate_M_0p0129.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.0129'}
snowList['Maherndl_platesM00129'] = NMplaM00129

# Nina Maherndl rimed aggregates of plates (M=0.0205) QJRMS 2023
NMplaM00205 = {'path':module_path+'/ssrga_coeffs_plate_M_0p0205.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.0205'}
snowList['Maherndl_platesM00205'] = NMplaM00205

# Nina Maherndl rimed aggregates of plates (M=0.0324) QJRMS 2023
NMplaM00324 = {'path':module_path+'/ssrga_coeffs_plate_M_0p0324.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.0324'}
snowList['Maherndl_platesM00324'] = NMplaM00324

# Nina Maherndl rimed aggregates of plates (M=0.0514) QJRMS 2023
NMplaM00514 = {'path':module_path+'/ssrga_coeffs_plate_M_0p0514.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.0514'}
snowList['Maherndl_platesM00514'] = NMplaM00514

# Nina Maherndl rimed aggregates of plates (M=0.0816) QJRMS 2023
NMplaM00816 = {'path':module_path+'/ssrga_coeffs_plate_M_0p0816.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.0816'}
snowList['Maherndl_platesM00816'] = NMplaM00816

# Nina Maherndl rimed aggregates of plates (M=0.1290) QJRMS 2023
NMplaM01290 = {'path':module_path+'/ssrga_coeffs_plate_M_0p1290.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.1290'}
snowList['Maherndl_platesM01290'] = NMplaM01290

# Nina Maherndl rimed aggregates of plates (M=0.2045) QJRMS 2023
NMplaM02045 = {'path':module_path+'/ssrga_coeffs_plate_M_0p2045.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.2045'}
snowList['Maherndl_platesM02045'] = NMplaM02045

# Nina Maherndl rimed aggregates of plates (M=0.3245) QJRMS 2023
NMplaM03245 = {'path':module_path+'/ssrga_coeffs_plate_M_0p3245.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.3245'}
snowList['Maherndl_platesM03245'] = NMplaM03245

# Nina Maherndl rimed aggregates of plates (M=0.5145) QJRMS 2023
NMplaM05145 = {'path':module_path+'/ssrga_coeffs_plate_M_0p5145.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.5145'}
snowList['Maherndl_platesM05145'] = NMplaM05145

# Nina Maherndl rimed aggregates of plates (M=0.8155) QJRMS 2023
NMplaM08155 = {'path':module_path+'/ssrga_coeffs_plate_M_0p8155.csv',
               'msg':'Table of Nina Maherndl aggregates of plates M=0.8155'}
snowList['Maherndl_platesM08155'] = NMplaM08155

#############################################################################

# Nina Maherndl rimed aggregates of mixed monomers (M=0 unrimed) QJRMS 2023
NMmixM00000 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p00.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.0000'}
snowList['Maherndl_mixturesM00000'] = NMmixM00000

# Nina Maherndl rimed aggregates of mixed monomers (M=0.0129) QJRMS 2023
NMmixM00129 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p0129.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.00129'}
snowList['Maherndl_mixturesM00129'] = NMmixM00129

# Nina Maherndl rimed aggregates of mixed monomers (M=0.0205) QJRMS 2023
NMmixM00205 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p0205.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.0205'}
snowList['Maherndl_mixturesM00205'] = NMmixM00205

# Nina Maherndl rimed aggregates of mixed monomers (M=0.0324) QJRMS 2023
NMmixM00324 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p0324.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.0324'}
snowList['Maherndl_mixturesM00324'] = NMmixM00324

# Nina Maherndl rimed aggregates of mixed monomers (M=0.0514) QJRMS 2023
NMmixM00514 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p0514.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.0514'}
snowList['Maherndl_mixturesM00514'] = NMmixM00514

# Nina Maherndl rimed aggregates of mixed monomers (M=0.0816) QJRMS 2023
NMmixM00816 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p0816.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.0816'}
snowList['Maherndl_mixturesM00816'] = NMmixM00816

# Nina Maherndl rimed aggregates of mixed monomers (M=0.1290) QJRMS 2023
NMmixM01290 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p1290.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.1290'}
snowList['Maherndl_mixturesM01290'] = NMmixM01290

# Nina Maherndl rimed aggregates of mixed monomers (M=0.2045) QJRMS 2023
NMmixM02045 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p2045.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.2045'}
snowList['Maherndl_mixturesM02045'] = NMmixM02045

# Nina Maherndl rimed aggregates of mixed monomers (M=0.3245) QJRMS 2023
NMmixM03245 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p3245.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.3245'}
snowList['Maherndl_mixturesM03245'] = NMmixM03245

# Nina Maherndl rimed aggregates of mixed monomers (M=0.5145) QJRMS 2023
NMmixM05145 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p5145.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.5145'}
snowList['Maherndl_mixturesM05145'] = NMmixM05145

# Nina Maherndl rimed aggregates of mixed monomers (M=0.8155) QJRMS 2023
NMmixM08155 = {'path':module_path+'/ssrga_coeffs_mixed_M_0p8155.csv',
               'msg':'Table of Nina Maherndl aggregates of mixtures M=0.8155'}
snowList['Maherndl_mixturesM08155'] = NMmixM08155

#############################################################################

# Nina Maherndl rimed aggregates of rosettes (M=0 unrimed) QJRMS 2023
NMrosM00000 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p00.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.0000'}
snowList['Maherndl_rosettesM00000'] = NMrosM00000

# Nina Maherndl rimed aggregates of rosettes (M=0.0129) QJRMS 2023
NMrosM00129 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p0129.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.0129'}
snowList['Maherndl_rosettesM00129'] = NMrosM00129

# Nina Maherndl rimed aggregates of rosettes (M=0.0205) QJRMS 2023
NMrosM00205 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p0205.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.0205'}
snowList['Maherndl_rosettesM00205'] = NMrosM00205

# Nina Maherndl rimed aggregates of rosettes (M=0.0324) QJRMS 2023
NMrosM00324 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p0324.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.0324'}
snowList['Maherndl_rosettesM00324'] = NMrosM00324

# Nina Maherndl rimed aggregates of rosettes (M=0.0514) QJRMS 2023
NMrosM00514 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p0514.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.0514'}
snowList['Maherndl_rosettesM00514'] = NMrosM00514

# Nina Maherndl rimed aggregates of rosettes (M=0.0816) QJRMS 2023
NMrosM00816 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p0816.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.0816'}
snowList['Maherndl_rosettesM00816'] = NMrosM00816

# Nina Maherndl rimed aggregates of rosettes (M=0.1290) QJRMS 2023
NMrosM01290 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p1290.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.1290'}
snowList['Maherndl_rosettesM01290'] = NMrosM01290

# Nina Maherndl rimed aggregates of rosettes (M=0.2045) QJRMS 2023
NMrosM02045 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p2045.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.2045'}
snowList['Maherndl_rosettesM02045'] = NMrosM02045

# Nina Maherndl rimed aggregates of rosettes (M=0.3245) QJRMS 2023
NMrosM03245 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p3245.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.3245'}
snowList['Maherndl_rosettesM03245'] = NMrosM03245

# Nina Maherndl rimed aggregates of rosettes (M=0.5145) QJRMS 2023
NMrosM05145 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p5145.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.5145'}
snowList['Maherndl_rosettesM05145'] = NMrosM05145

# Nina Maherndl rimed aggregates of rosettes (M=0.8155) QJRMS 2023
NMrosM08155 = {'path':module_path+'/ssrga_coeffs_rosette_M_0p8155.csv',
               'msg':'Table of Nina Maherndl aggregates of rosettes M=0.8155'}
snowList['Maherndl_rosettesM08155'] = NMrosM08155


class snowProperties():
	def __init__(self):
		# In principle this could be a parent class, separate init for average and table properties. 
		# The average is nothing more than a table with only one entry for all sizes, or??? it is just the dictionary interface that is easier
		raise NotImplementedError('It would be probably easier if the snow library returns a snowProperty object based on an identifier on call, but we are not ready yet')


## Class to manage the library of snow properties
class snowLibraryClass():
	def __init__(self):
		self._library = snowLib
		self._fileList = snowList
		logging.info('Initialize a library of snow properties')

	def __call__(self, diameters, identifier,
				 velocity_model='Boehm92', kwargsVelocity={}, # TODO bring this to _compute module, it doesn't make too much sense to be here
				 massVelocity=None, areaVelocity=None):
		logging.debug('return the snow properties for selected sizes')
		if identifier in self._library.keys():
			logging.debug('got AVG '+identifier+str(self._library[identifier]))
			kappa = np.ones_like(diameters)*self._library[identifier]['kappa']
			beta = np.ones_like(diameters)*self._library[identifier]['beta']
			gamma = np.ones_like(diameters)*self._library[identifier]['gamma']
			zeta1 = np.ones_like(diameters)*self._library[identifier]['zeta1']
			alpha_eff = np.ones_like(diameters)*self._library[identifier]['aspect']
			ar_mono = np.ones_like(diameters)*self._library[identifier]['ar_mono']
			mass = self._library[identifier]['am']*diameters**self._library[identifier]['bm']
			area = self._library[identifier]['aa']*diameters**self._library[identifier]['ba']

		elif identifier in self._fileList.keys():
			logging.debug('got TABLE '+identifier+str(self._fileList[identifier]))
			with open(self._fileList[identifier]['path']) as f:
				line = f.read().split('#')[-1].split('\n')[0]
				am = float(line.split('am=')[-1].split(',')[0])
				bm = float(line.split('bm=')[-1].split(',')[0])
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

		if massVelocity is None:
			massVelocity = mass
		if areaVelocity is None:
			areaVelocity = area

		vel = fallspeeds[velocity_model](diameters, massVelocity, areaVelocity, **kwargsVelocity) # TODO also need to pass kwargs additional and bring to the _compute module

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
snowLibrary = snowLibraryClass()
