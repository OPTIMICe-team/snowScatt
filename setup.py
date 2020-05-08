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

long_description = """A Python code for computing the scattering properties
of complex shaped snowflakes using the Self-Similar Rayleigh-Gans Approximation.

Requires NumPy and SciPy.
"""

import sys
from Cython.Build import cythonize

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('snowScatt', parent_package, top_path,
        version = '0.1.0-alpha',
        author  = "Davide Ori",
        author_email = "davide.ori87@gmail.com",
        description = "SSRGA scattering computation and snow properties",
        license = "MIT",
        url = 'https://github.com/DaveOri/snowScatt.git',
        download_url = \
            'https://github.com/DaveOri/snowScatt.git',
        long_description = long_description,
        classifiers = [
            "Development Status :: 3 - alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Physics",
        ]
    )

    return config


from distutils.extension import Extension
# Create extension objects to be passed to the setup() function
ssrgalib = Extension(name="snowScatt.ssrgalib",  # indicate where it should be available this will be the name of the module to be imported
                     sources=["cython/ssrga_module.pyx",
                              "src/ssrga.c",
                              "src/dielectric_factor.c"
                             ],
                     extra_compile_args=["-O2", "-ffast-math", "-Wall", "-fPIC",
                                         "-fPIC", "-std=c99", "-lm", "-lmvec",
                                         "-ldl", "-lc",
                                        ],
                     language="c")


if __name__ == "__main__":

    from numpy.distutils.core import setup
    setup(configuration=configuration,
        packages = ['snowScatt',
                    'snowScatt.snowProperties',
                    'snowScatt.refractiveIndex',
                    'snowScatt.instrumentSimulator',],
        package_dir={'snowScatt': 'python' ,
                     'snowScatt.snowProperties': 'python/snowProperties',
                     'snowScatt.refractiveIndex': 'python/refractiveIndex',
                     'snowScatt.instrumentSimulator': 'python/instrumentSimulator'},
        package_data = {'snowScatt.snowProperties': ['*.csv'],
                        'snowScatt.refractiveIndex': ['*.dat'],},
        platforms = ['any'],
        requires = ['numpy', 'scipy', 'Cython', 'pandas', 'os', 'xarray'],
        ext_modules=cythonize([ssrgalib]))