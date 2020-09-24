Go back to documentation `Homepage <index.html>`_

snowScatt - main package
========================

snowScatt has been designed to be used as a python module, but it also comes with a series of executable scripts that allows some functionalities to be used without the need of programming. As an example, a script that reads the snow library and compiles text lookup tables (LUT) is available within the snowScatt package. Another script mimics the behaviour of the original \emph{scatdb} [Liu2008] software in both the input and output formatting. Low-level C and FORTRAN interfaces to the snowScatt library, that would facilitate the implementation of SSRGA into microwave forward simulators, are also under development.


Schematics of the package
*************************

The structure of the snowScatt package is illustrated in Figure. snowScatt is designed to be modular and each component can be used as an independent program. The main information database is provided by the snow library which contains the snowflake microphysical properties. Together with the dielectric model for ice, the SSRGA parameters are used by the core SSRGA program to compute the single scattering properties. The mass and area parametrization can be used by the hydrodynamic model component to estimate the terminal fall speed of the snowflakes. Finally, the single scattering and microphysical properties of the snowflakes can be integrated over a particle size distribution (PSD) by the radar simulator to produce idealized synthetic Doppler radar measurements.


.. image :: img/snowScatt_scheme.png


Index of Submodules
*******************

The various components of snowScatt are illustrated here with more detail

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:

   snowProperties
   fallspeed
   refractiveIndex
   instrumentSimulator
   ssrga


