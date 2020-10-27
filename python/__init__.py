from ._compute import calcProperties
from ._compute import backscatter
from ._compute import backscatVel
from ._compute import snowMassVelocityArea
from . import refractiveIndex
from .snowProperties import snowLibrary

help_message = "This is the snwoScatt module. \n \
snowScatt loads the following modules and functions \n\n \
calcProperties, backscatter, backscatVel and \
snowMassVelocityArea are functions defined in the _compute \
submodule. Print their respective docstrings for further \
informations or navigate the online documentation \n\n \
snowLibrary is the main object loaded by the module \n \
to print a list of the available snow particle properties \
execute \n\n snowScatt.snowLibrary.info() \n"

def help():
	print(help_message)