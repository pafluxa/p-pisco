# coding: utf-8
import sys
import os

import numpy
import numpy as np

from scipy.optimize import curve_fit

import pylab
import matplotlib.pyplot as plt

import healpy
from   healpy.projector import MollweideProj, CartesianProj
from   healpy.pixelfunc import vec2pix

from pisco.mapping.core import matrices_to_maps

from cmb_analysis.powerspectrum.pyspice import spice

# Load data from user input
data = numpy.load( sys.argv[1] )
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
I,Q,U,W = matrices_to_maps( NSIDE, AtA, AtD )

I[ W > 0 ] = I[ W > 0 ]/W[ W > 0 ]
Q[ W > 0 ] = Q[ W > 0 ]/W[ W > 0 ]
U[ W > 0 ] = U[ W > 0 ]/W[ W > 0 ]

# Clip bad pixels
#I = numpy.clip( I, -1e-4, 1e-4 )
#Q = numpy.clip( Q, -1e-5, 1e-5 )
#U = numpy.clip( U, -1e-5, 1e-5 )

I_orig = numpy.load( sys.argv[2] )['I']
Q_orig = numpy.load( sys.argv[2] )['Q']
U_orig = numpy.load( sys.argv[2] )['U']

# Create a mask and a window
window = numpy.copy( W )
window[ W >= 100 ] = 1.0
window[ W <  100 ] = 0.0
'''
mask = numpy.zeros_like( W, dtype='bool' ) 
mask[ W < 3  ] = True
I_orig[ mask ] = healpy.UNSEEN
Q_orig[ mask ] = healpy.UNSEEN
U_orig[ mask ] = healpy.UNSEEN

I     [ mask ] = healpy.UNSEEN
Q     [ mask ] = healpy.UNSEEN
U     [ mask ] = healpy.UNSEEN
'''
# Load beam FWHM from user input, in degrees
beam_fwhm = (float)( sys.argv[3] )

# Set LMAX to the maximum possible value /2
lmax = 250# (int)( 180.0/beam_fwhm ) + 100

# Smooth original maps for comparison
I_o_s, Q_o_s, U_o_s = healpy.smoothing( (I_orig, Q_orig, U_orig), fwhm=numpy.radians(beam_fwhm), pol=True )

# Get map Pixel Window function
pixwin_temp, pixwin_pol = healpy.pixwin( NSIDE, pol=True )
pixwin_temp = pixwin_temp[0:lmax+1]
pixwin_pol  = pixwin_pol [0:lmax+1]

# Get Gaussian window function
glTT, glEE, glBB, glTE = healpy.sphtfunc.gauss_beam( numpy.radians(beam_fwhm), pol=True, lmax=lmax ).T
wl_TT = (glTT**2 * pixwin_temp )
wl_EE = (glEE**2 * pixwin_pol  )
wl_BB = (glBB**2 * pixwin_pol  )

# Define spherical projection axis
vec2pix_func = lambda x,y,z: vec2pix( NSIDE,x,y,z,nest=False)
fig  = pylab.figure( figsize=(12,20) )

# Setup projection axes
ax_I = fig.add_subplot(231)
ax_Q = fig.add_subplot(232)
ax_U = fig.add_subplot(233)

ax_rI = fig.add_subplot(234)
ax_rQ = fig.add_subplot(235)
ax_rU = fig.add_subplot(236)

proj = MollweideProj()

# Perform projection
I_proj = proj.projmap( I, vec2pix_func )
Q_proj = proj.projmap( Q, vec2pix_func )
U_proj = proj.projmap( U, vec2pix_func )

rI_proj = proj.projmap( I - I_o_s, vec2pix_func )
rQ_proj = proj.projmap( Q - Q_o_s, vec2pix_func )
rU_proj = proj.projmap( U - U_o_s, vec2pix_func )

# Show images
ax_I.set_title( 'PISCO recovered Stokes I' )
imageI = ax_I.imshow( I_proj,
                   vmin=-5e-5, vmax=5e-5,
                   cmap=plt.get_cmap('gray'),
                   origin='lower')
pylab.colorbar( imageI, ax=ax_I, orientation='horizontal', ticks=[-5E-5,0,5E-5] , format='%+2.2e' ) 

ax_Q.set_title( 'PISCO recovered Stokes Q' )
imageQ = ax_Q.imshow( Q_proj,
                   vmin=-5e-7, vmax=5e-7,
                   cmap=plt.get_cmap('gray'),
                   origin='lower')
pylab.colorbar( imageQ, ax=ax_Q, orientation='horizontal', ticks=[-5E-7,0,5E-7], format='%+2.2e' )

ax_U.set_title( 'PISCO recovered Stokes U' )
imageU = ax_U.imshow( U_proj,
                   vmin=-5e-7, vmax=5e-7,
                   cmap=plt.get_cmap('gray'),
                   origin='lower')
pylab.colorbar( imageU, ax=ax_U, orientation='horizontal', ticks=[-5E-7,0,5E-7], format='%+2.2e' )

ax_rI.set_title( 'residual I' )
imagerI = ax_I.imshow( rI_proj,
                   vmin=-5e-6, vmax=5e-6,
                   cmap=plt.get_cmap('gray'),
                   origin='lower')
pylab.colorbar( imagerI, ax=ax_rI, orientation='horizontal', ticks=[-5E-5,0,5E-5] , format='%+2.2e' ) 

ax_rI.set_title( 'residual I' )
imagerI = ax_I.imshow( rI_proj,
                   vmin=-5e-6, vmax=5e-6,
                   cmap=plt.get_cmap('gray'),
                   origin='lower')
pylab.colorbar( imagerI, ax=ax_rI, orientation='horizontal', ticks=[-5E-5,0,5E-5] , format='%+2.2e' ) 

ax_rI.set_title( 'residual I' )
imagerI = ax_I.imshow( rI_proj,
                   vmin=-5e-6, vmax=5e-6,
                   cmap=plt.get_cmap('gray'),
                   origin='lower')
pylab.colorbar( imagerI, ax=ax_rI, orientation='horizontal', ticks=[-5E-5,0,5E-5] , format='%+2.2e' ) 

pylab.show()


