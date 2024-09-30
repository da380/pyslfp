import numpy as np
import pyshtools as pysh
from numpy import pi as pi


##############################################################
# sets Sobolev type covariance
def sobolev_covariance(L,std = 1.,s = 0.,mu = 0.,b = 1.):
    Q = np.zeros(L+1)
    norm = 0.
    for l in range(L+1):
        fac = 1.+mu*mu*l*(l+1)
        fac = fac**(-s)
        norm += (2*l+1)*fac/(4*pi)
        Q[l] = fac
    Q = b*b*std*std*Q/norm
    return Q


##############################################################
# sets heat kernel type covariance
def heat_covariance(L,std = 1.,mu = 0.,b = 1.):
    Q = np.zeros(L+1)
    norm = 0.
    for l in range(L+1):
        fac = -l*(l+1)*mu*mu
        fac = np.exp(fac)
        norm += (2*l+1)*fac/(4*pi)
        Q[l] = fac
    Q = b*b*std*std*Q/norm
    return Q

###############################################################
# generates random field with Laplacian like covariance
def random_field(Q,b = 1.):
    L = Q.size-1
    fun_lm = pysh.SHCoeffs.from_zeros(lmax=L,normalization = 'ortho')
    for l in range(L+1):
        fac = np.sqrt(Q[l])/b
        ranp = np.random.normal(size = l+1)
        rann = np.random.normal(size = l+1)
        fun_lm.coeffs[0,l,0] = fac*ranp[0]
        for m in range(1,l+1):
            fun_lm.coeffs[0,l,m] = fac*ranp[m]
            fun_lm.coeffs[1,l,m] = fac*rann[m]
    return fun_lm.expand(grid = 'GLQ')        

################################################################
# returns the two-point correlation function for a random field
# given the covariance operator
def correlation_function(Q,lat0 = 0.,lon0 = 0.,b = 1.):
    L = Q.size-1    
    ths = 90.- lat0
    phs = lon0 + 180.               
    cf_lm = pysh.SHCoeffs.from_zeros(lmax=L,normalization = 'ortho')               
    ylm = pysh.expand.spharm(cf_lm.lmax,ths,phs,normalization = 'ortho')        
    for l in range(L+1):
        fac = Q[l]/(b*b)
        cf_lm.coeffs[0,l,0] = cf_lm.coeffs[0,l,0] + ylm[0,l,0]*fac
        for m in range(1,l+1):
            cf_lm.coeffs[0,l,m] += ylm[0,l,m]*fac
            cf_lm.coeffs[1,l,m] += ylm[1,l,m]*fac
    cf = cf_lm.expand(grid = 'GLQ')
    return cf



####################################################
# returns the action of the covariance operator, Q,
# for a rotationally invariant Gaussian random field
# on an input function
def apply_covariance(Q,fun):
    grid = fun.grid
    L = fun.lmax
    fun_lm = fun.expand(normalization = 'ortho')
    for l in range(L+1):
        fun_lm.coeffs[:,l,:] *= Q[l]
    return fun_lm.expand(grid = grid)
