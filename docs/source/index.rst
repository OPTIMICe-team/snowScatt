.. snowScatt documentation master file, created by
   sphinx-quickstart on Thu Mar 26 13:13:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to snowScatt's documentation!
=====================================

snowScatt is a python package that provides scattering and microphysical properties of realistically shaped snowflakes using the Self-Similar Rayleigh-Gans Approximation, hydrodynamic theory and snowflake aggregation model.

Why snowScatt?
**************

At the university of Cologne we like snow modeling and observation a lot! We have produced more than 100k snowflake 3D shapes using a physical aggregation model. We used various paramaters to generate them and we can use the structural properties to simulate scattering and fallspeed.

snowScatt provide access to our snowflake database. It can be used to obtain:

* SSRGA, and microphysical parameters such as mass-size and velocity-size power-law fits
* scattering properties simulated using the SSRGA model
* mass and terminal fallspeed
* simulated radar Doppler spectra and reflectivity

Included Libraries
******************

The core python package snowScatt provides an interface to a fast C code (ssrgalib) to compute the scattering properties of snowflakes provided their SSRGA parameters. Other utility libraries are also included and are necessary for the functioning of the main package:

* snowProperties - contains the database of snow properties and the interface for accessing them
* fallSpeed - submodule that provides various models for the simulation of snowflake terminal fallspeed
* refractiveIndex - provides an interface to various refractive index models for ice, water and mixture of substances.
* instrumentSimulator - forward simulation of microwave remote sensing instruments
* ssrga - functions to derive the SSRGA parameters of an ensemble of snowflake shapes

These submodules are installed as part of the main package, but can also be used as standalone libraries


.. toctree::
   :numbered:
   :glob:
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial
   simple_usage
   notebooks/tutorial
   snowScatt
   troubleshooting
   credits


Module Documentation
********************
I find hard times making autodoc working on readthedocs. If it fails or you find empty pages follow the instrutions on how to compile the Documentation 

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
