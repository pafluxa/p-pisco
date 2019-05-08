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

# Load data from user input
data = numpy.load( sys.argv[1] )
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
I,Q,U,W = matrices_to_maps( NSIDE, AtA, AtD )

I_orig = numpy.load( sys.argv[2] )['I']
Q_orig = numpy.load( sys.argv[2] )['Q']
U_orig = numpy.load( sys.argv[2] )['U']

# Create a window
window = numpy.zeros_like( W , dtype='bool' )
# Mask pixels with bad coverage
window[ W <  10 ] = True
# Cut low and high declinations
tht,phi = healpy.pix2ang( NSIDE, numpy.arange( I.size ) )
low_cut = -65
hgh_cut =  25
window[ (numpy.pi/2.0 - tht) < low_cut ] = True
window[ (numpy.pi/2.0 - tht) > hgh_cut ] = True

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

'''Compute power specra'''
TT, EE, BB, TE, EB, TB       = healpy.anafast( (I    ,Q    ,U    ), pol=True )
TTo, EEo, BBo, TEo, EBo, TBo = healpy.anafast( (I_o_s,Q_o_s,U_o_s), pol=True )
'''Adjust to the lmax parameter'''
TTo = TTo[0:lmax+1]
EEo = EEo[0:lmax+1]
BBo = BBo[0:lmax+1]
TT  = TT [0:lmax+1]
EE  = EE [0:lmax+1]
BB  = BB [0:lmax+1]

# Define l numbers
ell = numpy.arange( TT.size )
ell2 = ell * (ell+1)/(2*numpy.pi)

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

fig  = pylab.figure( figsize=(12,5) )

ax_TT = fig.add_subplot( 131 )
ax_TT.set_title( r'TT Power Spectra' )
ax_TT.set_xlabel( r'$\ell$' )
ax_TT.set_ylabel( r'K$^2$' )
ax_TT.set_ylim( (1e-20,1e-9) ) 
ax_TT.set_xlim( (2,lmax) )
ax_TT.set_yscale( 'log' )
ax_TT.set_xscale( 'log' )

ax_EE = fig.add_subplot( 132 )
ax_EE.set_title( r'EE Power Spectra' )
ax_EE.set_xlabel( r'$\ell$' )
ax_EE.set_xlim( (2,lmax) )
ax_EE.set_ylim( (1e-20,1e-9) )
ax_EE.set_xscale( 'log' )

ax_BB = fig.add_subplot( 133 )
ax_BB.set_title( r'BB Power Spectra' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_xlim( (2,lmax) )
ax_BB.set_ylim( (1e-20,1e-9) )
ax_BB.set_xscale( 'log' )

ax_TT.plot( ell2*TTo/wl_TT, label='' ) 
ax_TT.plot( ell2*TT/wl_TT , label='PISCO rec. $C_{\ell}^{TT}$' ) 
ax_TT.legend()

ax_EE.plot( ell2*EEo/wl_EE, label='' ) 
ax_EE.plot( ell2*EE/wl_EE , label='PISCO rec. $C_{\ell}^{EE}$' ) 
ax_EE.legend()

ax_BB.plot( ell2*BBo/wl_BB, label='') 
ax_BB.plot( ell2*BB/wl_BB , label='PISCO rec. $C_{\ell}^{BB}$' ) 
#ax_BB.plot( ell2*BB/wl_BB , label='PISCO recovered BB' ) 
ax_BB.legend()

pylab.show()


