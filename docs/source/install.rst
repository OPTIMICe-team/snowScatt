Go back to documentation `Homepage <index.html>`_

Installation instructions
=========================

The reference implementation of snowScatt is python3, but it has been used in python2 environments without problems. The developer do not actively support legacy python2, but they will try to not break back-compatibility as much as possible. If you wish to install snowScatt in a python2 environment just substitute the following instructions with the python2 equivalents (apt install python3-X -> apt install python-X; pip3 -> pip and so on)


Get the code
************

snowScatt is under active development. New snow properties are continuously uploaded to the snow library, the package is extended with new functionalities and bugs are fixed. We always recommend to clone the most recent version of the package from the official `repository <https://github.com/OPTIMICe-team/snowScatt.git>`_ .

If you want to avoid possible issues related to continuous development you can also download one of our periodic `releases <https://github.com/OPTIMICe-team/snowScatt/releases>`_

How to install snowScatt (tested on Ubuntu 16.04+)
**************************************************

The installation of snowScatt requires few dependencies on modern Linux distributions.

A C compiler is needed. Most Linux distributions natively provide gcc and snowScatt works just fine with it. Other compilers are expected to work as well, but they have not been tested.

Python required packages are numpy, scipy, pandas, xarray, Cython. On Ubuntu you can use system packages

.. code-block:: bash

	sudo apt install python3-scipy python3-numpy python3-pandas cython3

Alternatively you can install the required packages with pip

.. code-block:: bash

	pip3 install scipy numpy pandas cython

cd to the package directory and launch the command

.. code-block:: bash

	python3 setup.py install

If you do not have writing permission on your pythonpath you can also append the - -user flag to the installation command.


Compile Documentation (tested on Ubuntu 16.04+)
***********************************************

Package manual and documentation is based on sphinx and autodoc.

Install sphinx, navigate the doc folder, compile the html documentation and open it with any we browser

.. code-block:: bash

	sudo apt install python3-sphinx python3-nbsphinx
	cd docs
	make html
	firefox build/html/index.html &


Notes on how to install on Windows
**********************************

I guess it would be fairly easy to install the package under windows emulating a linux distribution with WSL (Windows subsystem for linux) available on Windows 10

I have experimented some installation attempts on Windows 8.1 using Anaconda (STILL NOT WORKING ANY HELP IS APPRECIATED):

Install Anaconda

If you do not have a C compiler you need to install one `VS <https://visualstudio.microsoft.com/downloads>`_
It might be that you need only MSVC 14.xxx component, but I need help with MS tools to understand that

Alternatively you can also try to install an opensource compiler such as gcc or clang

Use conda to install git and conda-buid. Open an Anaconda Prompt and type

conda install git conda-build

If you did not downloaded the repository yet, move to the directory where you want to download snowScatt package and do 

.. code-block:: bash

	git clone https://github.com/DaveOri/snowScatt.git

Try to compile and install with

.. code-block:: bash

	pip install
