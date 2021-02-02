Simple Usage Examples
=====================

Some simple use-case scenarios are illustrated in the `example folder <https://github.com/OPTIMICe-team/snowScatt/tree/master/examples>`_

Here is a list of jupyter noteboks that will guide you thorugh the basic steps to effectively use snowScatt


List snowScatt Library content
******************************

In this notebook you will learn how to have a basic command on snowScatt, list its modules and the snow particle properties included in the snowLibrary basic_usage_.

.. _basic_usage: notebooks/list_content.ipynb


Microphysical properties of snow
********************************

In this example you will use snowScatt to calculate the microphysical properties of some snow types. You will make plots of mass, area and fall velocity using different hydrodynamic models microphysics_.

.. _microphysics: notebooks/microphysics.ipynb


Scattering properties of snow
*****************************

This notebook shows how to use snowscatt to compute the basic scattering properties of one type of snow. It shows how to use the default configuration and how to overload the mass information in order to match some user-defined values single_scattering_.

.. _single_scattering: notebooks/single_scattering.ipynb


Multifrequency Doppler radar simulations
****************************************

This is a basic application example on how to use snowScatt to make simple Doppler radar simulations and evaluate the multifrequency characteristic of different snow types radar_example_.

.. _radar_example: notebooks/radar_example.ipynb 


Create a Look Up Table
**********************

If you do not wish to integrate python or snowScatt in your production environment, but you still want to use snowScatt properties it might be useful for you to know how to write a Look Up Table from snowScatt to a text or a netCDF file tutorial_lut_.

.. _tutorial_lut: notebooks/tutorial.ipynb