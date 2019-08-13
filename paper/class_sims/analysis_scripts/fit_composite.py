# coding: utf-8
import sys
import os

import numpy
from   numpy import pi
import numpy as np

import pandas

import pylab
import matplotlib.pyplot as plt

import healpy
from   healpy.projector import MollweideProj, CartesianProj
from   healpy.pixelfunc import vec2pix

from pisco.mapping.core import matrices_to_maps

from cmb_analysis.powerspectrum.pyspice import spice

import numpy as np
from scipy.special import iv
from scipy.integrate import simps
from scipy.optimize import curve_fit

def I0( x ):

    return iv( 0, x )

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

    window *= 2 * pi

    return window

def make_elliptical_beam_symmetrical_profile(fwhm_x, fwhm_y, theta):
    '''
    Returns profile of elliptical Gaussian beam integrated over phi from 0 to 2 pi
    
    inputs (in degrees)
   
        fwhm_x  : single value Full Width at Half Maximum, in the x (South-North) 
        fwhm_y  : single value Full Width at Half Maximum, in the y (East-West) 
        theta   : Array of angles for output profile

    ''' 
    fwhm2sigma = np.sqrt(8.0 * np.log(2.0))
    sigma_x = fwhm_x / fwhm2sigma
    sigma_y = fwhm_y / fwhm2sigma
    sx2 = sigma_x * sigma_x
    sy2 = sigma_y * sigma_y
    s_sym2 = (2.0 * sx2 * sy2) / (sx2 + sy2)
    
    major = np.max( (sx2, sy2) )
    minor = np.min( (sx2, sy2) )
    e2 = 1.0 - minor / major
    th2 = theta * theta
    profile = 2.0 * np.pi * np.exp(-0.5 * th2 / s_sym2) * I0(e2 * th2 / (4.0 * minor))
    return profile

def make_composite_elliptical_beam_window_function(fwhm_x, fwhm_y, nx, radius, lmax, return_composite=False):
    '''
    Returns window function of elliptical Gaussian beam integrated over phi from 0 to 2 pi
    
    inputs (in degrees)
   
        fwhm_x  : Array of Full Width at Half Maxima, in the x (South-North) 
        fwhm_y  : Array of Full Width at Half Maxima, in the y (East-West) 
        nx      : Number of points to compute composite beam profile
        radius  : Radius of composite beam profile
        lmax    : Maximum multipole moment at which to compute window function

    ''' 

    theta = numpy.linspace(0.0, radius, nx )
    composite = np.zeros(nx)                                                   
    ndet = len(fwhm_x)
    for i in range(ndet):
        composite +=  make_elliptical_beam_symmetrical_profile(fwhm_x[i], fwhm_y[i], theta)
    composite /= ndet

    if return_composite:
        return composite
    else:
        window = beam2bl(composite, np.radians(theta), lmax)
        return window

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

#----------------------------------------------------------------------------------------------------------#  
# Read beam parameter file                                                                                    
print 'reading beam parameters'                                                                               
beam_data = pandas.read_csv( './data/array_data/qband_array_data_beam_params.csv'  )                                                   
feeds     = numpy.array( beam_data[   'Feed'] )                                                               
azOff     = numpy.array( beam_data[  'AzOff'] )                                                               
elOff     = numpy.array( beam_data[  'ElOff'] )                                                               
fwhm_x    = numpy.array( beam_data[ 'FWHM_x'] )                                                               
fwhm_y    = numpy.array( beam_data[ 'FWHM_y'] )                                                               
rotation  = numpy.array( beam_data[  'theta'] )                                                               
#----------------------------------------------------------------------------------------------------------#  

# Get composite beam profile
lmax = 250
radius = 5.0
theta = numpy.linspace( 0.0, radius, lmax )
comp_profile = make_composite_elliptical_beam_window_function( fwhm_x, fwhm_y, lmax, radius, lmax, return_composite=True ) 


# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeff, var_matrix = curve_fit( gauss, theta, comp_profile, p0=p0 )

print "sigma = ", coeff[2]
print "FWHM  = ", 2.355*coeff[2]

import pylab
pylab.plot( theta, comp_profile )
pylab.show()

