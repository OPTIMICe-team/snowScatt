import numpy as np
from snowScatt._constants import _g as g
from snowScatt._constants import _ice_density as rho_ice


def Boehm92(diam, rho_air, nu_air, mass, area, as_ratio=1.0):
	"""
	TODO PUT SOME REFERENCE HERE

    Parameters:
    -----------
    diam : array(Nparticles) double
    	spectrum of diameters of the particles [meters]
    rho_air : scalar double
    	air density [kilograms/meter**3]
    nu_air : scalar double
    	air kinematic viscosity [meters**2/seconds]
    mass : array(Nparticles) double
    	mass of the particles [kilograms]
    area : array(Nparticles) double
    	cross section area [meters**2]
    as_ratio : scalar double
    	Correction factor for the calculation of area ration in non-spherical
    	symmetric particles. See Karrer et al. 2020

    Returns:
    --------
    vterm_bohm : array(Nparticles) double
    	terminal fallspeed computed according to the model [meters/second]
	"""

	q = area / (np.pi/4.0 * diam**2)
	eta_air = nu_air*rho_air # dynamic viscosity
	
	alpha = np.array(as_ratio) #1.0
	X_0 = 2.8e6
	#X = 8.0*mass*grav*rho_air/(np.pi*(eta_air**2)*q**0.25)
	X = 8.0*mass*g*rho_air/(np.pi*(eta_air**2)*np.maximum(alpha,np.ones_like(alpha)*1.0)*np.maximum(q**0.25,q)) #reduced to 8.0*mtot*grav*rho/(np.pi*(eta**2)*q**(1/4) = 8.0*mtot*grav*rho/(np.pi*(eta**2)*(area / (np.pi/4.0 * diam**2))**(1/4) for alpha=1 and q<1 (which is usually the case)
	k = 1.0 #np.minimum(np.maximum(0.82+0.18*alpha,
	        #                  np.ones_like(alpha)*0.85),
	        #       0.37+0.63/alpha,
	        #       1.33/(np.maximum(np.log(alpha),
	        #                        np.ones_like(alpha)*0.0)+1.19)) #k is 1 for alpha=1
	gama_big = np.maximum(np.ones_like(alpha)*1.0, np.minimum(np.ones_like(alpha)*1.98,3.76-8.41*alpha+9.18*alpha**2-3.53*alpha**3)) #1 for alpha=1
	C_DP = np.maximum(0.292*k*gama_big,0.492-0.2/np.sqrt(alpha)) #0.292 for alpha=1
	C_DP = np.maximum(1.0,q*(1.46*q-0.46))*C_DP #0.292 for alpha=1
	C_DP_prim = C_DP*(1.0+1.6*(X/X_0)**2)/(1.0+(X/X_0)**2) #0.292 for small particles; larger for bigger particles 
	beta = np.sqrt(1.0+C_DP_prim/6.0/k*np.sqrt(X/C_DP_prim))-1
	N_Re0 = 6.0*k/C_DP_prim*beta**2
	C_DO = 4.5*k**2*np.maximum(alpha,np.ones_like(alpha)*1.0)
	gama_small = (C_DO - C_DP)/4.0/C_DP
	N_Re  = N_Re0*(1.0 + (2.0*beta*np.exp(-beta*gama_small))/((2.0+beta)*(1.0+beta)) )
	#Re = 8.5*((1.0+0.1519*X**0.5)**0.5-1.0)**2
	vterm_bohm = N_Re*eta_air/diam/rho_air
	return vterm_bohm


def Boehm89(diam, rho_air, nu_air, mass, area):
	"""
	TODO PUT SOME REFERENCE HERE

    Parameters:
    -----------
    diam : array(Nparticles) double
    	spectrum of diameters of the particles [meters]
    rho_air : scalar double
    	air density [kilograms/meter**3]
    nu_air : scalar double
    	air kinematic viscosity [meters**2/seconds]
    mass : array(Nparticles) double
    	mass of the particles [kilograms]
    area : array(Nparticles) double
    	cross section area [meters**2]

    Returns:
    --------
    vterm_bohm : array(Nparticles) double
    	terminal fallspeed computed according to the model [meters/second]
	"""

	q = area / (np.pi/4.0 * diam**2)
	eta_air = nu_air*rho_air # dynamic viscosity
	
	#alpha = np.array(as_ratio) #1.0
	#X_0 = 2.8e6
	X = 8.0*mass*g*rho_air/(np.pi*(eta_air**2)*q**0.25)
	
	#k = np.minimum(np.maximum(0.82+0.18*alpha,np.ones_like(alpha)*0.85),0.37+0.63/alpha,1.33/(np.maximum(np.log(alpha),np.ones_like(alpha)*0.0)+1.19)) #k is 1 for alpha=1
	#gama_big = np.maximum(np.ones_like(alpha)*1.0, np.minimum(np.ones_like(alpha)*1.98,3.76-8.41*alpha+9.18*alpha**2-3.53*alpha**3)) #1 for alpha=1
	#C_DP = np.maximum(0.292*k*gama_big,0.492-0.2/np.sqrt(alpha)) #0.292 for alpha=1
	#C_DP = np.maximum(1.0,q*(1.46*q-0.46))*C_DP #0.292 for alpha=1
	#C_DP_prim = C_DP*(1.0+1.6*(X/X_0)**2)/(1.0+(X/X_0)**2) #0.292 for small particles; larger for bigger particles 
	#beta = np.sqrt(1.0+C_DP_prim/6.0/k*np.sqrt(X/C_DP_prim))-1
	#N_Re0 = 6.0*k/C_DP_prim*beta**2
	#C_DO = 4.5*k**2*np.maximum(alpha,np.ones_like(alpha)*1.0)
	#gama_small = (C_DO - C_DP)/4.0/C_DP
	#N_Re  = N_Re0*(1.0 + (2.0*beta*np.exp(-beta*gama_small))/((2.0+beta)*(1.0+beta)) )
	Re = 8.5*((1.0+0.1519*X**0.5)**0.5-1.0)**2
	vterm_bohm = Re*eta_air/diam/rho_air
	return vterm_bohm


def HeymsfieldWestbrook2010(diaSpec_SI, rho_air_SI, nu_SI, mass, area, k=0.5):
	"""
	Heymsfield, A. J. & Westbrook, C. D. Advances in the Estimation of Ice Particle Fall Speeds
    Using Laboratory and Field Measurements. Journal of the Atmospheric Sciences 67, 2469–2482 (2010).
    equations 9-11

    Parameters:
    -----------
    diaSpec_SI : array(Nparticles) double
    	spectrum of diameters of the particles [meters]
    rho_air_SI : scalar double
    	air density [kilograms/meter**3]
    nu_SI : scalar double
    	air kinematic viscosity [meters**2/seconds]
    mass : array(Nparticles) double
    	mass of the particles [kilograms]
    area : array(Nparticles) double
    	cross section area [meters**2]
    k : scalar double
    	tuning coefficient for turbulent flow defaults to 0.5

    Returns:
    --------
    velSpec : array(Nparticles) double
    	terminal fallspeed computed according to the model [meters/second]
	"""
	delta_0 = 8.0
	C_0 = 0.35
	
	area_proj = area/((np.pi/4.)*diaSpec_SI**2) # area ratio
	eta = nu_SI * rho_air_SI #!now dynamic viscosity

	Xstar = 8.0*rho_air_SI*mass*g/(np.pi*area_proj**(1.0-k)*eta**2)# !eq 9
	Re=0.25*delta_0**2*((1.0+((4.0*Xstar**0.5)/(delta_0**2.0*C_0**0.5)))**0.5 - 1 )**2 #!eq10
	 
	velSpec = eta*Re/(rho_air_SI*diaSpec_SI)
	return velSpec


def KhvorostyanovCurry2005(diam, rho_air, nu_air, mass, area, smooth=False):
	"""
	TODO PUT SOME REFERENCE HERE

    Parameters:
    -----------
    diam : array(Nparticles) double
    	spectrum of diameters of the particles [meters]
    rho_air : scalar double
    	air density [kilograms/meter**3]
    nu_air : scalar double
    	air kinematic viscosity [meters**2/seconds]
    mass : array(Nparticles) double
    	mass of the particles [kilograms]
    area : array(Nparticles) double
    	cross section area [meters**2]
    smooth : scalar bool
    	Decide wheather or not use the smooth approximation for the estimation
    	of the drag coefficient from the Best number X

    Returns:
    --------
    velSpec : array(Nparticles) double
    	terminal fallspeed computed according to the model [meters/second]
	"""

	# Best number eq. (2.4b) with buoyancy
	Vb = mass/rho_ice
	Fb = rho_air * Vb * g
	eta_air = nu_air*rho_air # dynamic viscosity
	Xbest = 2. * np.abs(mass*g-Fb) * rho_air * diam**2 / (area * eta_air**2)
	if( smooth ):
	  Cd  = X2Cd_kc05smooth(Xbest)
	else:
	  Cd  = X2Cd_kc05rough(Xbest)
	return np.sqrt( 2*np.abs(mass*g - Fb)/(rho_air * area * Cd))


def X2Cd_kc05rough(Xbest):
	do_i = 5.83
	co_i = 0.6
	Ct = 1.6
	X0_i = 0.35714285714285714285e-6 # 1.0/2.8e6
	# derived constants
	c1 = 4.0 / ( do_i**2 * np.sqrt(co_i))
	c2 = 0.25 * do_i**2
	# Re-X eq. (2.5)
	bracket = np.sqrt(1.0 + c1 * np.sqrt(Xbest)) - 1.0
	# turbulent Reynold's number, eq (3.3)
	psi = (1.0+(Xbest*X0_i)**2) / (1.0+Ct*(Xbest*X0_i)**2)
	Re  = c2*bracket**2 # * np.sqrt(psi) # TODO remove psi in Re?
	# eq. (2.1) from KC05 with (3.2)
	return co_i * (1.0 + do_i / np.sqrt(Re))**2 / psi


def X2Cd_kc05smooth(Xbest):
	do_i = 9.06
	co_i = 0.292
	Ct = 1.6
	X0_i = 1.0/6.7e6
	c1 = 4.0/(do_i**2 * np.sqrt(co_i))
	c2 = 0.25 * do_i**2
	# Re-X eq. (2.5)
	bracket = np.sqrt(1.0 + c1 * np.sqrt(Xbest)) - 1.0
	# turbulent Reynold's number, eq (3.3)
	psi = (1+(Xbest*X0_i)**2) / (1+Ct*(Xbest*X0_i)**2)
	Re  = c2*bracket**2 #* np.sqrt(psi) # TODO remove psi in Re?
	# eq. (2.1) from KC05 with (3.2)
	return co_i * (1. + do_i/np.sqrt(Re))**2 / psi