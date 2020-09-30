from ._compute import calcProperties
from ._compute import backscatter
from ._compute import backscatVel
from ._compute import snowMassVelocityArea
from . import refractiveIndex
from .snowProperties import snowLibrary

help_message="This is the snwoScatt module"

def help():
	print(help_message)