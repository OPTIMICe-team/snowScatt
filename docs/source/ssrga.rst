Go back to documentation `Homepage <index.html>`_

ssrga
=====

This module contains two functions that are used to derive SSRGA parameters kappa, beta, gamma, zeta and alpha_eff from a set of particle shapes. Follow the `SSRGA example <https://github.com/OPTIMICe-team/snowScatt/blob/master/examples/prepare_ssrga_table.py>`_  to understand how to use it.

The first function **area_function** computes the number of mass elements in the shapefile along a specific direction defined by the zenith angle *theta*. It is assumed that the shapefiles hold measuring units (not normalized). Particles are oriented and assumed to take random orientation along the vertical *z* axis. If the direction along which the area_function is computed is off the vertical Nphi*sin(theta) number of samples are taken. As an example if theta=0.5pi (horizontal), the area_function is calculated along Nphi (default 32) equally spaced azimuth directions.

The second function **fitSSRGA** takes an array of normalized area functions calculated by the previous function and fit the SSRGA parameters. Optionally it can also produce the plots of the fits of the mean shape and the power spectrum of the fluctuations around the mean shape. By looking at the plots it is possible to have a qualitative assessment of the goodness of the fits.

.. automodule:: snowScatt.ssrga._ssrgaFit
   :members:

