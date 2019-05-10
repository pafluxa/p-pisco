# coding: utf-8
import numpy as np
#import pylab
#import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.integrate import simps
from scipy.optimize import curve_fit
import pandas

sigma2fwhm = np.sqrt(8.0 * np.log(2.0))

def beam2bl(beam, theta, lmax):
    """Computes a transfer (or window) function b(l) in spherical 
    harmonic space from its circular beam profile b(theta) in real 
    space.

    Parameters
    ----------
    beam : array
        Circular beam profile b(theta).
    theta : array
        Radius at which the beam profile is given. Has to be given 
        in radians with same size as beam.
    lmax : integer
        Maximum multipole moment at which to compute b(l).

    Returns
    -------
    bl : array
        Beam window function b(l).
    """

    nx = len(theta)
    nb = len(beam)
    if nb != nx:
        print("beam and theta must have same size!")

    x = np.cos(theta)
    st = np.sin(theta)
    window = np.zeros(lmax + 1)

    p0 = np.ones(nx)
    p1 = np.copy(x)

    window[0] = simps(beam * p0 * st, theta)
    window[1] = simps(beam * p1 * st, theta)

    for l in np.arange(2, lmax + 1):
        p2 = x * p1 * (2 * l - 1) / l - p0 * (l - 1) / l
        window[l] = simps(beam * p2 * st, theta)
        p0 = p1
        p1 = p2

    window *= 2 * np.pi

    return window

def make_elliptical_beam_symmetrical_profile(fwhm_x, fwhm_y, theta):
    '''
    Returns profile of elliptical Gaussian beam integrated over phi from 0 to 2 pi
    
    inputs (in degrees)
   
        fwhm_x  : single value Full Width at Half Maximum, in the x (South-North) 
        fwhm_y  : single value Full Width at Half Maximum, in the y (East-West) 
        theta   : Array of angles for output profile

    ''' 
    sigma_x = fwhm_x / sigma2fwhm
    sigma_y = fwhm_y / sigma2fwhm
    sx2 = sigma_x * sigma_x
    sy2 = sigma_y * sigma_y
    s_sym2 = (2.0 * sx2 * sy2) / (sx2 + sy2)
    
    major = np.max( (sx2, sy2) )
    minor = np.min( (sx2, sy2) )
    e2 = 1.0 - minor / major
    th2 = theta * theta
    profile = np.exp(-0.5 * th2 / s_sym2) * i0(e2 * th2 / (4.0 * minor))
    return profile

def make_composite_elliptical_beam_window_function(fwhm_x, fwhm_y, lmax, radius=5.0, return_composite=False, pol=False):
    '''
    Returns window function of elliptical Gaussian beam integrated over phi from 0 to 2 pi
    
    inputs (in degrees)
   
        fwhm_x  : Array of Full Width at Half Maxima, in the x (South-North) 
        fwhm_y  : Array of Full Width at Half Maxima, in the y (East-West) 
        lmax    : Maximum multipole moment at which to compute window function
        radius  : Radius of composite beam profile
        pol : bool
            if False, output has size (lmax+1) and is temperature beam
            if True output has size (lmax+1, 4) with components:
            * temperature beam
            * grad/electric polarization beam
            * curl/magnetic polarization beam
            * temperature * grad beam

    ''' 

    theta = np.linspace(0.0, radius, lmax )
    composite = np.zeros(lmax)                                                   
    ndet = len(fwhm_x)
    for i in range(ndet):
        composite +=  make_elliptical_beam_symmetrical_profile(fwhm_x[i], fwhm_y[i], theta)
    composite /= ndet

    window = beam2bl(composite, np.radians(theta), lmax)
    #Equivalent Gaussian beam sigma^2
    sigma2 = window[0] / (2.0 * np.pi)
    #FWHM of equivalent Gaussian beam (degrees)
    fwhm_sym = np.degrees(np.sqrt(sigma2) * sigma2fwhm)
    if return_composite:
        return composite, fwhm_sym
    else:
        #Normalize 
        g = window / window[0]
        if not pol:  # temperature-only beam
            return g, fwhm_sym
        else:  # polarization beam
            # polarization factors [1, 2 sigma^2, 2 sigma^2, sigma^2]
            pol_factor = np.exp([0.0, 2 * sigma2, 2 * sigma2, sigma2])
            return g[:, np.newaxis] * pol_factor, fwhm_sym

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, sigma = p
    return A*np.exp(-x**2/(2.*sigma**2))

#----------------------------------------------------------------------------------------------------------#  
if __name__ == '__main__':
    # Read beam parameter file                                                                                    
    print 'reading beam parameters'                                                                               
    beam_data = pandas.read_csv( './data/array_data/qband_array_data_beam_params.csv')                                                   
#    beam_data = pandas.read_csv( 'qband_era2_pair_averaged_beam_parameters.csv')                                                   
#    beam_data = np.genfromtxt('qband_era2_pair_averaged_beam_parameters.csv', delimiter=',', names=True, dtype=None, encoding='ascii')
    feeds     = np.array( beam_data[   'Feed'] )                                                               
    azOff     = np.array( beam_data[  'AzOff'] )                                                               
    elOff     = np.array( beam_data[  'ElOff'] )                                                               
    fwhm_x    = np.array( beam_data[ 'FWHM_x'] )                                                               
    fwhm_y    = np.array( beam_data[ 'FWHM_y'] )                                                               
    rotation  = np.array( beam_data[  'Theta'] )                                                               
#----------------------------------------------------------------------------------------------------------# 
    '''
    fwhm_x = 1.5
    fwhm_y = 1.3
    fwhm_sym = np.sqrt((2.0 * fwhm_x**2 * fwhm_y**2) / (fwhm_x**2 + fwhm_y **2))
    print "FWHM_sym = ", fwhm_sym
    fwhm_x = np.asarray( [fwhm_x] ) 
    fwhm_y = np.asarray( [fwhm_y] ) 
    '''

    # Get composite beam profile
    lmax = 250
    radius = 5.0
    theta = np.linspace( 0.0, radius, lmax )
    comp_profile, fwhm_sym = make_composite_elliptical_beam_window_function( fwhm_x, fwhm_y, lmax, radius=radius, return_composite=True ) 
    print "FWHM_sym = ", fwhm_sym

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [1., 1.]

    coeff, var_matrix = curve_fit( gauss, theta, comp_profile, p0=p0 )

    print "Amp   = ", coeff[0]
    print "FWHM  = ", sigma2fwhm * coeff[1]

#    pylab.plot( theta, comp_profile )
#    pylab.show()

