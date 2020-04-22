"""
Copyright (C) 2020 Davide Ori - University of Cologne
This piece of code is derived from the pytmatrix.psd module originally coded
by Jussi Leinonen https://github.com/jleinonen/pytmatrix.git

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.special import gamma


class PSD(object):
    def __call__(self, D):
        if np.shape(D) == ():
            return 0.0
        else:
            return np.zeros_like(D)

    def __eq__(self, other):
        return False


class ExponentialPSD(PSD):
    """Exponential particle size distribution (PSD).
    
    Callable class to provide an exponential PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = N0 * exp(-Lambda*D)

    Attributes:
        N0: the intercept parameter.
        Lambda: the inverse scale parameter        
        D_max: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D0, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, N0=1.0, Lambda=1.0, D_max=None):
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0/Lambda if D_max is None else D_max

    def __call__(self, D):
        psd = self.N0 * np.exp(-self.Lambda*D)
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, ExponentialPSD) and \
                (self.N0 == other.N0) and (self.Lambda == other.Lambda) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False


class UnnormalizedGammaPSD(ExponentialPSD):
    """Gamma particle size distribution (PSD).
    
    Callable class to provide an gamma PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = N0 * D**mu * exp(-Lambda*D)

    Attributes:
        N0: the intercept parameter.
        Lambda: the inverse scale parameter
        mu: the shape parameter
        D_max: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D0, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """
    
    def __init__(self, N0=1.0, Lambda=1.0, mu=0.0, D_max=None):
        super(UnnormalizedGammaPSD, self).__init__(N0=N0, Lambda=Lambda, 
            D_max=D_max)
        self.mu = mu

    def __call__(self, D):
        # For large mu, this is better numerically than multiplying by D**mu
        psd = self.N0 * np.exp(self.mu*np.log(D)-self.Lambda*D)
        if np.shape(D) == ():
            if (D > self.D_max) or (D==0):
                return 0.0
        else:
            psd[(D > self.D_max) | (D == 0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return super(UnnormalizedGammaPSD, self).__eq__(other) and \
                self.mu == other.mu
        except AttributeError:
            return False
        


class GammaPSD(PSD):
    """Normalized gamma particle size distribution (PSD).
    
    Callable class to provide a normalized gamma PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = Nw * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)
    f(mu) = 6/(3.67**4) * (3.67+mu)**(mu+4)/Gamma(mu+4)

    Attributes:
        D0: the median volume diameter.
        Nw: the intercept parameter.
        mu: the shape parameter.
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, D0=1.0, Nw=1.0, mu=0.0, D_max=None):
        self.D0 = float(D0)
        self.mu = float(mu)
        self.D_max = 3.0*D0 if D_max is None else D_max
        self.Nw = float(Nw)
        self.nf = Nw * 6.0/3.67**4 * (3.67+mu)**(mu+4)/gamma(mu+4)

    def __call__(self, D):
        d = (D/self.D0)
        psd = self.nf * np.exp(self.mu*np.log(d)-(3.67+self.mu)*d)
        if np.shape(D) == ():
            if (D > self.D_max) or (D==0.0):
                return 0.0
        else:
            psd[(D > self.D_max) | (D==0.0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, GammaPSD) and (self.D0 == other.D0) and \
                (self.Nw == other.Nw) and (self.mu == other.mu) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False


class BinnedPSD(PSD):
    """Binned gamma particle size distribution (PSD).
    
    Callable class to provide a binned PSD with the given bin edges and PSD
    values.

    Args (constructor):
        The first argument to the constructor should specify n+1 bin edges, 
        and the second should specify n bin_psd values.        
        
    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters outside the bins.
    """
    
    def __init__(self, bin_edges, bin_psd):
        if len(bin_edges) != len(bin_psd)+1:
            raise ValueError("There must be n+1 bin edges for n bins.")
        
        self.bin_edges = bin_edges
        self.bin_psd = bin_psd
        
    def psd_for_D(self, D):       
        if not (self.bin_edges[0] < D <= self.bin_edges[-1]):
            return 0.0
        
        # binary search for the right bin
        start = 0
        end = len(self.bin_edges)
        while end-start > 1:
            half = (start+end)//2
            if self.bin_edges[start] < D <= self.bin_edges[half]:
                end = half
            else:
                start = half
                                
        return self.bin_psd[start]                    
        
    def __call__(self, D):
        if np.shape(D) == (): # D is a scalar
            return self.psd_for_D(D)
        else:
            return np.array([self.psd_for_D(d) for d in D])
    
    def __eq__(self, other):
        if other is None:
            return False
        return len(self.bin_edges) == len(other.bin_edges) and \
            (self.bin_edges == other.bin_edges).all() and \
            (self.bin_psd == other.bin_psd).all()