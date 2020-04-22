# cython: boundscheck=False
# cython: wraparound=False
# Comments above are special. Please do not remove.
cimport numpy as np  # needed for function arguments
import numpy as np  # needed for np.ones, np.empty_like ...
cimport ssrgaLib

ctypedef np.float32_t float_t
ctypedef np.float64_t double_t
ctypedef np.int32_t int_t
ctypedef np.complex128_t complex_t


def ssrga(np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Deff,
	      np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Vol,
	      np.ndarray[dtype = double_t, ndim = 1, mode = "c"] wl,
	      np.ndarray[dtype = complex_t, ndim = 1, mode = "c"] K,
	      np.ndarray[dtype = double_t, ndim = 1, mode = "c"] kappa,
	      np.ndarray[dtype = double_t, ndim = 1, mode = "c"] gamma,
	      np.ndarray[dtype = double_t, ndim = 1, mode = "c"] beta,
	      np.ndarray[dtype = double_t, ndim = 1, mode = "c"] zeta0,
	      int_t Ntheta):

	cdef int_t Npart = len(Deff)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cext
	Cext = np.empty_like(Deff, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cabs
	Cabs = np.empty_like(Deff, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Csca
	Csca = np.empty_like(Deff, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cbck
	Cbck = np.empty_like(Deff, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] asym
	asym = np.empty_like(Deff, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] phase
	phase = np.zeros(Npart*Ntheta, dtype=np.float64)
	ssrgaLib.ssrga(Npart, < double * > Deff.data, < double * > Vol.data,
		           < double * > wl.data, < double complex * > K.data, 
		           < double * > kappa.data, < double * > gamma.data,
		           < double * > beta.data, < double * > zeta0.data, Ntheta,
		           < double * > Cext.data, < double * > Cabs.data,
		           < double * > Csca.data, < double * > Cbck.data,
		           < double * > asym.data, < double * > phase.data)
	return Cext, Cabs, Csca, Cbck, asym, phase.reshape(Npart, Ntheta)


def ssrgaBack(np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Deff,
	      	  np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Vol,
	          np.ndarray[dtype = double_t, ndim = 1, mode = "c"] wl,
	          np.ndarray[dtype = complex_t, ndim = 1, mode = "c"] K,
	          np.ndarray[dtype = double_t, ndim = 1, mode = "c"] kappa,
	          np.ndarray[dtype = double_t, ndim = 1, mode = "c"] gamma,
	          np.ndarray[dtype = double_t, ndim = 1, mode = "c"] beta,
	          np.ndarray[dtype = double_t, ndim = 1, mode = "c"] zeta0,
	          int_t Ntheta):

	cdef int_t Npart = len(Deff)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cbck
	Cbck = np.empty_like(Deff, dtype=np.float64)
	ssrgaLib.ssrgaBack(Npart, < double * > Deff.data, < double * > Vol.data,
                       < double * > wl.data, < double complex * > K.data, 
		               < double * > kappa.data, < double * > gamma.data,
		               < double * > beta.data, < double * > zeta0.data,
                       < double * > Cbck.data)
	return Cbck


def hexPrismK(np.ndarray[dtype = complex_t, ndim = 1, mode = 'c'] refind,
	          np.ndarray[dtype = double_t, ndim = 1, mode = 'c'] aspect):
	cdef np.ndarray[dtype = complex_t, ndim = 1, mode = 'c'] K
	K = np.empty_like(refind)
	cdef int_t N = len(K)
	ssrgaLib.vectorK(N, < double complex * > refind.data,
		             < double * > aspect.data, < double complex * > K.data)
	return K


def ssrga_single(): # not functioning, kept for testing
	cdef double_t Deff = 1.0
	cdef double_t Vol = 2.0
	cdef double_t wl = 3.0
	cdef complex_t K = complex(1.78, 1.3e-6)
	cdef double_t kappa = 5.0
	cdef double_t gamma = 6.0
	cdef double_t beta = 7.0
	cdef double_t zeta0 = 8.0
	cdef int_t Ntheta = 10 
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cext
	Cext = 10*np.ones(1, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cabs
	Cabs = 11*np.ones(1, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Csca
	Csca = 12*np.ones(1, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] Cbck
	Cbck = 13*np.ones(1, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] asym
	asym = 14*np.ones(1, dtype=np.float64)
	cdef np.ndarray[dtype = double_t, ndim = 1, mode = "c"] phase
	phase = 15*np.ones(Ntheta, dtype=np.float64)
	phase[Ntheta-1] = -15
	print(phase)
	ssrgaLib.ssrga_single(Deff, Vol, wl, K, kappa, gamma, beta, zeta0, Ntheta,
		                  < double * > Cext.data, < double * > Cabs.data,
		                  < double * > Csca.data, < double * > Cbck.data,
		                  < double * > asym.data, < double * > phase.data)