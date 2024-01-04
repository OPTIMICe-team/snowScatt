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
import numpy
from setuptools import Extension, setup
#from Cython.Build import cythonize 

setup(name="snowScatt", # name of the package should be handled by .toml
      ext_modules=[Extension(name="snowScatt.ssrgalib",
                             sources=["cython/ssrga_module.pyx",
                                      "src/ssrga.c",
                                      "src/dielectric_factor.c"],
                             extra_compile_args=["-O2", "-ffast-math", "-Wall", "-fPIC",
                                                 "-fPIC", "-std=c99", "-lm", "-lmvec",
                                                 "-ldl", "-lc"],
                             language="c",
                             include_dirs=[numpy.get_include()]
                             )
                  ],
      )
