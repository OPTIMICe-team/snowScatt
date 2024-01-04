=========
snowScatt
=========
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3746261.svg
   :target: https://doi.org/10.5281/zenodo.3746261

`SnowScatt <https://github.com/DaveOri/SnowScatt>`_ is an Open Source Python package to compute scattering and microphysical properties of snowflakes ensambles. Use Self-Similar Rayleigh-Gans Approximation and snowflake aggregation models

-----------
Quick Guide
-----------

Clone or download the repository. Install the listed requirements (numpy and Cython for the installation, the rest [scipy, matplotlib, xarray, pandas] for the runtime, including some example scripts). Navigate the main folder and install the package with
.. code-block::
  python -m pip .

The scripts in the example folder should give enough information on how to use the software main functionalities.
After installation you should be able to execute in a python environment
.. code-block::
  import snowScatt
  snowScatt.help()

--------------------
Online Documentation
--------------------
.. image:: https://readthedocs.org/projects/snowscatt/badge/?version=latest
:target: https://snowscatt.readthedocs.io/en/latest/?badge=latest
:alt: Documentation Status

The documentation and a usage guide are available at `https://snowScatt.readthedocs.io <https://snowScatt.readthedocs.io>`_.
If for any reason readthedocs should broke a static copy of the documentation can be accessed `here <http://gop.meteo.uni-koeln.de/~dori/build/html/index.html>`_.

-------------
Cite the code
-------------
If you use snowScatt please cite the related publication `<https://gmd.copernicus.org/articles/14/1511/2021/>`_

The package version at the time of the publication is indexed on Zenodo DOI `10.5281/zenodo.3746261 <https://doi.org/10.5281/zenodo.3746261>`_
