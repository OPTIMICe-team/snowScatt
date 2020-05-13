from scipy.fftpack import fft
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def _calc_mean_shape(k, x):
	"""
	Calculates the mean, normalized area-function.
	Function defined in Hogan and Westbrook (2014)

	Parameters
	----------
	k : scalar - double
		kurtosis parameter kappa of the SSRGA expansion defines how peaked is
		the normalized distribution of mass within the aggregate.
		Higher values mean mass more concentrated in the core of the aggregate
		with respect to the edges
	x : array-like double
		normalized distances from the center. Should range from -0.5 to +0.5

	Returns
	-------
	shape : array(len(x)) - double
		mean area function for the specified k
	"""
	
	shape = (1.0+k/3.0)*np.cos(np.pi*x) + k*np.cos(3.0*np.pi*x)

	return shape

def _calc_kappa_lut(N=200, min_kappa=-0.15, max_kappa=0.40):
	"""
	Generates a look-up table of values of the kappa parameter and the
	corresponding statistical variance.
	To be used to estimate kappa from the variance of the distribution of the
	mean area-function.
	Since the functional form of the mean shape is just a fit not all values of
	kappa are physically reasonable. Too high value would lead to negative mean
	distributions in the edges and too low would give unphysical highly bimodal
	distributions.

	Parameters
	----------
	N : scalar double (optional defaults to 1000)
		Number of values to be included in the look-up table
	min_kappa : scalar double (optional defaults to -0.15)
		Minimum possible value for kappa
	max_kappa : scalar double (optional defaults to 0.4)
		Maximum possible value for kappa

	Returns
	-------
	kappas : array(N) double
		Array of possible values of the kappa parameter
	Avar : array(N) double
		Variances corresponding to the values of kappa
	"""

	x = np.linspace(-0.5, 0.5, 1000)
	kappas = np.linspace(min_kappa, max_kappa, N)
	A = [_calc_mean_shape(k, x) for k in kappas]
	normA = [a/a.mean() for a in A]
	Avar = [np.mean(a*x**2) for a in normA]

	return kappas, Avar


[kappas_lut, Avar_lut] = _calc_kappa_lut()
kappa_interp = interp1d(Avar_lut, kappas_lut, kind='nearest')


def _calc_vector_true_length(A):
	"""
	Gives the number of elements from the first non-zero to the last along the
	specified axis
	
	Parameters
	----------
	A : array(N)
		Vector of single-valued elements (preferably integers)

	Returns
	-------
	n : scalar integer
		True cropped length of vectors of A. N>=n
	"""

	# argwhere will give you the coordinates of every non-zero point
	true_points = np.argwhere(A)
	
	return true_points.max() - true_points.min() + 1

"""
Something that later on I might want to configure
If I have a very high resolution, 1024 might be too low.
For example the 10.24mm particle goes into a normalized grid of 20um (the grid
goes from -1 to 1 and I use only -0.5 to 0.5)
I need the normalized grid to be finer
"""

nX = 1024 # TODO make it configurable
X = np.linspace(-1, 1, nX)
index = np.abs(X) <= 0.5
Xfit = X[index]

def _normalize_area_functions(A):
	"""
	Normalize the area function into a common nX wide vector. Centers of mass
	are shifted so that they lie at the center of the vectors. nX is usually a
	power of 2 to make the FFT faster.
	TODO: Possibly for very large high resolved particles the number of bins in
	the normalized vector is not high enough. It might be that we are losing
	resolution!! Is it truly a problem? => scaling of the monomers are neglected
	anyway, but in principle the details of the particle morphology matter at
	very high frequency

	Parameters
	----------
	A : array(Nbins) integer
		Vector array of single valued elements (preferably integers).
		Contains the original area function

	Returns
	-------
	Anorm : array(nX) - double
		Vector of rescaled area function. The center of mass is shifted at the
		center of the vector. Area function is interpolated from the original
		Does not matter if it is integral conserved. The mass is calculated
		outside and every vector is normalized
	nx : scalar - integer
		Length (in number of bins) of the original area function. Basically the
		length of the particle along the propagation direction
	"""
	# Get the true length in number of bins
	nx = _calc_vector_true_length(A)
	x = np.linspace(0, len(A)/nx, len(A))
	x = x - np.sum(x*A)/np.sum(A)

	# Interpolate, fill outside with zeros
	A_full_interp = interp1d(x, A, kind='linear',
		                     bounds_error=False, fill_value=0.0)
	A_full = A_full_interp(X)
	return A_full/A_full[index].mean(), nx


def _compute_power_spectrum(Adiff):
	"""
	Compute the power spectrum of the input vector using the FFT algorithm
	Parameters
	----------
	Adiff : array - double
		One dimensional real array. Expectes the deviations of a particular
		area function sample from the average area function shape

	Returns
	-------
	P : array (2*len(Adiff)) double
		Power spectrum of the input vector

	"""
	F = fft(Adiff)
	nF = len(F)
	return np.real(F*F.conj()*2.0/(nF*nF))

# nominal_size = 0.01 # This is not needed here. 
# it can be treated externally, it is just the bin center. 
# perhaps it is more useful than Dmax when computing alpha_eff??
def fitSSRGA(A, Dmax, voxel_spacing,
	         max_index_largescale=12,
	         do_plots=False, plots_path=None):
	"""
	Fit SSRGA parameter kappa, beta, gamma, zeta

	Parameters
	----------
	A : 2D array(Nparticles*Nsamples, Nbins) - integer
		List of area functions. Linearized representation of the aggregate mass
		distribution along the propagation direction. The propagation direction
		is sampled in Nbins intervals. The array contain the integer number of
		dipoles belonging to each bin. For each particles there might be
		multiple samples (orientations when theta!=0). The particle legnth along
		the propagation direction can be inferred from the length of valid
		area functions. The bin spacing is first assumed to be equal to the
		corresponding voxel_spacing.
	Dmax : array(Nparticles) - double
		List of maximum diameter values, one per particle [meters]
	voxel_spacing : array(Nparticles) - double
		List or volume element resolutions, one per particle [meters]. Particles
		might have different resolutions
	max_index_largescale : scalar double
		Maximum order of the power spectrum taken into account for the fit of
		gamma and beta parameters
	do_plots : bool
		If True plots the average shape and the power spectrum fits
	plots_path : string
		If not None, save the plots at the specified path. Appends a meaningful
		suffix and uses .png format

	Returns
	-------
	kappa : scalar double
		Kurtosis parameter kappa of the mean shape of the aggregates
	beta : scalar double
		beta parameter. Intercept of the power spectrum of the deviations from
		the mean shape as a log power-law fit.
	gamma : scalar double
		gamma parameter. Average slope of the first portion of the power
		spectrum of the deviations from the mean shape as a log power-law fit
	zeta : scalar double
		zeta parameter. Ratio between the actual first element of the power
		spectrum of the deviations from the mean and the one calculated from the
		power-law fit
	alpha_eff : scalar double
		effective aspect ratio of the particle. Scaling between the actual size
		along the propagation direction and the characteristic size of the
		snowflake (here assumed to be Dmax).
		TODO: perhaps it is better to use the nominal size? -> surely not if the
		parameters are derived for a large population of particles (not binned)
	volume : scalar double
		mean volume occupied by the particle (mass/ice_density here computed as
		Nvoxel*voxel_spacing**3)

	Raises
	------
	AttributeError : if the list of arguments has some problems
	"""

	Nparticles = len(Dmax)
	if Nparticles != len(voxel_spacing):
		raise AttributeError('len(Dmax) != len(voxel_spacing) these must equal Nparticles')
	Nvectors = A.shape[0] # Number of vectors
	if (Nvectors % Nparticles):
		raise AttributeError('length(A) must be a multiple of Nparticles')
	Nsamples = Nvectors//Nparticles # Number of samples per particle (orientations)

	# Rescale area functions for fft processing - and
	# Compute the length of each vector and derive aspect ratio
	Anorm = np.apply_along_axis(func1d=_normalize_area_functions, axis=1, arr=A)
	nx = Anorm[:, 1]
	Anorm = np.stack(Anorm[:, 0])
	mean_nx = nx.reshape((Nparticles, Nsamples)).mean(axis=1)
	alpha_eff = (mean_nx*voxel_spacing/Dmax).mean()
	
	# Compute mean volume occupied
	volume = (A.sum(axis=1)*np.repeat(voxel_spacing, Nsamples)**3).mean()

	# Fit kappa
	Avar = np.mean(Xfit**2*Anorm.mean(axis=0)[index])
	kappa = kappa_interp(Avar)

	# Compute the fitted mean shape and the deviations
	Afit = 0.5*np.pi*_calc_mean_shape(kappa, Xfit)
	Adiff = Anorm[:, index] - np.ones([Nvectors, 1])*Afit

	# Average power spectrum
	nXfit = index.sum()
	E = np.apply_along_axis(func1d=_compute_power_spectrum, axis=1, arr=Adiff)

	sum_power_spectrum = E[:, 1:nXfit//2+1].sum(axis=0) # Ignore first element: represents the mean of the field
	# sum_log_power_spectrum = np.log(E[:, 1:nXfit//2+1]).sum(axis=0)
	power_spectrum = sum_power_spectrum / Nvectors # average
	
	index_largescale = np.arange(1, max_index_largescale, 1)
	#variance_largescale = (np.sum(power_spectrum[index_largescale])).real
	
	print('PARSEVAL', np.sum(power_spectrum)/np.var(Adiff[:]))

	j = np.arange(1, nXfit//2+1) # All the meaningful wavenumbers, start from 1
	# Fit the P = beta*(2j)**-gamma function
	gamma, beta = np.polyfit(x=np.log10(2*(index_largescale+1)),
		                     y=np.log10(power_spectrum[index_largescale]),
		                     deg=1)

	beta = 10.0**(beta)*(8.0/np.pi**2) # rescale beta
	gamma*=-1 # change sign of gamma
	power_spectrum_fit = (2*j)**-gamma # calculate the fitted power spectrum
	power_spectrum_fit = beta*power_spectrum_fit*np.pi**2/8.0 # rescale back
	zeta = power_spectrum[0]/power_spectrum_fit[0] # fix the first element

	if do_plots: # TODO make it save them somewhere if asked to
		plt.figure()
		plt.plot(X, Anorm.mean(axis=0))
		plt.plot(Xfit, Afit)

		power_spectrum_fit[0]*=zeta
		plt.figure()
		plt.scatter(j, power_spectrum)
		plt.plot(j, power_spectrum_fit)
		plt.xscale('log')
		plt.yscale('log')

	return kappa, beta, gamma, zeta, alpha_eff, volume