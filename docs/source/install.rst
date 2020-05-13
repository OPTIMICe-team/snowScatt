How to install snowScatt
========================

The installation of snowScatt requires few dependencies on modern Linux distributions. A C compiler is needed (tested with gcc). Python required packages are numpy, scipy, pandas, xarray, Cython

cd to the package directory and launch the command

.. code-block:: bash

	python3 setup.py install

If you do not have writing permission on your pythonpath you can also append the - -user flag to the installation command.

Notes on how to install on Windows using anaconda
*************************************************

SO FAR I DID NOT MANAGE TO INSTALL IT ON WINDOWS

Install Anaconda

If you do not have a C compiler you need to install one
https://visualstudio.microsoft.com/downloads
It might be that you need only MSVC 14.xxx component, but I need help with MS tools to understand that

Alternatively you can also try to install an opensource compiler such as gcc or clang

Use conda to install git and conda-buid. Open an Anaconda Prompt and type

conda install git conda-build

If you did not downloaded the repository yet, move to the directory where you want to download snowScatt package and do 

git clone https://github.com/DaveOri/snowScatt.git

Try to compile and install with

pip install 