# coding: utf-8
import sys
import os

import numpy
import numpy as np

import pylab
import matplotlib.pyplot as plt

import healpy
from   healpy.projector import MollweideProj, CartesianProj
from   healpy.pixelfunc import vec2pix

from pisco.mapping.core import matrices_to_maps

from cmb_analysis.powerspectrum.pyspice import spice

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# Load data from user input
data = numpy.load( sys.argv[1] )
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
# temporary fix
NSIDE = 128
I,Q,U,W = matrices_to_maps( NSIDE, AtA, AtD )

I_orig = numpy.load( sys.argv[2] )['I']
Q_orig = numpy.load( sys.argv[2] )['Q']
U_orig = numpy.load( sys.argv[2] )['U']

# Create a window
window = numpy.zeros_like( W , dtype='bool' )
# Mask pixels with bad coverage
window[ W <  3 ] = True
# Mask the maps
I[ window ] = healpy.UNSEEN
Q[ window ] = healpy.UNSEEN
U[ window ] = healpy.UNSEEN

# Load beam FWHM from user input, in degrees
beam_fwhm = (float)( sys.argv[3] )

# Set LMAX to the maximum possible value /2
lmax = 250# (int)( 180.0/beam_fwhm ) + 100

# Smooth original maps for comparison
I_o_s, Q_o_s, U_o_s = healpy.smoothing( (I_orig, Q_orig, U_orig), fwhm=numpy.radians(beam_fwhm), pol=True )

# Mask the maps
I_o_s[ window ] = healpy.UNSEEN
Q_o_s[ window ] = healpy.UNSEEN
U_o_s[ window ] = healpy.UNSEEN

# Define spherical projection axis
vec2pix_func = lambda x,y,z: vec2pix( NSIDE,x,y,z,nest=False)

fig  = pylab.figure( figsize=(12,20) )

healpy.mollview( I        , sub=(2,3,1) )
healpy.mollview( I - I_o_s, sub=(2,3,4) )

healpy.mollview( Q        , sub=(2,3,2) )
healpy.mollview( Q - Q_o_s, sub=(2,3,5) )

healpy.mollview( U        , sub=(2,3,3) )
healpy.mollview( U - U_o_s, sub=(2,3,6) )

pylab.show()

rTT, rEE, rBB, rTE, rEB, rTB = healpy.anafast( (I-I_o_s,Q-Q_o_s,U-U_o_s), pol=True )

pylab.subplot(131)
pylab.plot( rTT )

pylab.subplot(132)
pylab.plot( rEE )

pylab.subplot(133)
pylab.plot( rBB )

pylab.show()

'''
# Setup projection axes
ax_I  = fig.add_subplot(231)
ax_Q  = fig.add_subplot(232)
ax_U  = fig.add_subplot(233)

ax_Ir = fig.add_subplot(234)
ax_Qr = fig.add_subplot(235)
ax_Ur = fig.add_subplot(236)

proj = MollweideProj()

# Mask the maps to zero to avoid bug in `projmap`
I[ window ] = 0
Q[ window ] = 0
U[ window ] = 0
# Perform projection
I_proj = proj.projmap( I, vec2pix_func )
Q_proj = proj.projmap( Q, vec2pix_func )
U_proj = proj.projmap( U, vec2pix_func )


# Mask the maps to zero to avoid bug in `projmap`
I_o_s[ window ] = 0
Q_o_s[ window ] = 0
U_o_s[ window ] = 0
# Perform projection
I_r = proj.projmap( I - I_o_s, vec2pix_func )
Q_r = proj.projmap( Q - Q_o_s, vec2pix_func )
U_r = proj.projmap( U - U_o_s, vec2pix_func )

# Show images
ax_Ir.set_title( 'Residual I' )
imageIr = ax_Ir.imshow( I_r,
                   vmin=-5e-6, vmax=5e-6,
                   origin='lower')
cbarIr = add_colorbar( imageIr,  ticks=[-5e-6,0,5E-6] , format='%+2.2e' )

ax_Qr.set_title( 'Residual Q' )
imageQr = ax_Qr.imshow( Q_r,
                   vmin=-5e-9, vmax=5e-9,
                   origin='lower')
cbarQr = add_colorbar( imageQr,  ticks=[-5E-9,0,5E-9] , format='%+2.2e' )

ax_Ur.set_title( 'Residual U' )
imageUr = ax_Ur.imshow( U_r,
                   vmin=-5e-9, vmax=5e-9,
                   origin='lower')
cbarUr = add_colorbar( imageUr,  ticks=[-5E-9,0,5E-9] , format='%+2.2e' )

# Show images
ax_I.set_title( 'PISCO recovered Stokes I' )
imageI = ax_I.imshow( I_proj,
                   vmin=-5e-5, vmax=5e-5,
                   origin='lower')
cbarI = add_colorbar( imageI,  ticks=[-5E-5,0,5E-5] , format='%+2.2e', orientation='horizontal' )

ax_Q.set_title( 'PISCO recovered Stokes Q' )
imageQ = ax_Q.imshow( Q_proj,
                   vmin=-5e-7, vmax=5e-7,
                   origin='lower')
cbarQ = add_colorbar( imageQ,  ticks=[-5E-7,0,5E-7] , format='%+2.2e' )

ax_U.set_title( 'PISCO recovered Stokes U' )
imageU = ax_U.imshow( U_proj,
                   vmin=-5e-7, vmax=5e-7,
                   origin='lower')
cbarU = add_colorbar( imageU,  ticks=[-5E-7,0,5E-7] , format='%+2.2e' )

pylab.show()
'''
