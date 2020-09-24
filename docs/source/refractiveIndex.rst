Go back to documentation `Homepage <index.html>`_

The refractiveIndex module
==========================
This is a collection of utilities to deal with the formulation of ice and ice-air mixtures refractive index in the microwave for radar and radiometer remote sensing applications


The dielectric models for ice
*****************************
This submodule provides access to implementations of refractive index models for ice in the microwave. The implemented models currently available are the Matzler 2006, Warren 2008, and Iwabuchi 2011 formulations

.. automodule:: snowScatt.refractiveIndex.ice
   :members:

The dielectric mixing formula
*****************************
This submodule provides access to dielectric mixing formula of Maxwell-Garnett, Bruggeman and Shivola

.. automodule:: snowScatt.refractiveIndex.mixing
   :members:

Convenient function for snow
****************************
This submodule provides access to convenience functions that combine the refractive index submodule for ice and the dielectric mixing submodule in order to calculate the average dielectric properties of snowflakes as homogeneous mixtures of ice and air. It converts the snow density into volume fraction of ice and air and uses a dielectric mixing model and an ice refractive index model to estimate the dielectric properties of the mixture

.. automodule:: snowScatt.refractiveIndex.snow
   :members:

Dielectric utilities
********************
This submodule provides access to conveninence function to make conversions between commonly used dielectric quatities like the complex refractive index *n*, the complex dielectric permittivity *eps*, the complex radar dielectric factor *K* (Clausius-Mossotti) and its real squared norm *K2*

.. automodule:: snowScatt.refractiveIndex.utilities
   :members:

