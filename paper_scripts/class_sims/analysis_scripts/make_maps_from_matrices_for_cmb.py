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

# Load beam FWHM from user input, in degrees
beam_fwhm = (float)( sys.argv[3] )

# Set LMAX to the maximum possible value /2
lmax = 350# (int)( 180.0/beam_fwhm ) + 100

# Smooth original maps for comparison
I_o_s, Q_o_s, U_o_s     = healpy.smoothing( (I_orig, Q_orig, U_orig), fwhm=numpy.radians(beam_fwhm), pol=True )
_TTo, _EEo, _BBo, _,_,_ = spice( (I_orig,Q_orig,U_orig) )

'''Compute power specra'''
TT, EE, BB, TE, EB, TB       = spice( (I    ,Q    ,U    ) )
TTo, EEo, BBo, TEo, EBo, TBo = spice( (I_o_s,Q_o_s,U_o_s) )

'''Adjust to the lmax parameter'''
_TTo = _TTo[0:lmax+1]
_EEo = _EEo[0:lmax+1]
_BBo = _BBo[0:lmax+1]

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

fig  = pylab.figure( figsize=(12,20) )
# Setup projection axes
ax_I = fig.add_subplot(231)
ax_Q = fig.add_subplot(232)
ax_U = fig.add_subplot(233)
proj = CartesianProj()

# Perform projection
I_proj = proj.projmap( I, vec2pix_func )
Q_proj = proj.projmap( Q, vec2pix_func )
U_proj = proj.projmap( U, vec2pix_func )

ax_TT = fig.add_subplot( 234 )
ax_TT.set_title( r'TT Power Spectra' )
ax_TT.set_xlabel( r'$\ell$' )
ax_TT.set_ylabel( r'K$^2$' )
ax_TT.set_ylim( (1e-12,1e-8) ) 
ax_TT.set_xlim( (2,lmax) )
ax_TT.set_yscale( 'log' )
ax_TT.set_xscale( 'log' )

ax_EE = fig.add_subplot( 235 )
ax_EE.set_title( r'EE Power Spectra' )
ax_EE.set_xlabel( r'$\ell$' )
ax_EE.set_xlim( (2,lmax) )
ax_EE.set_ylim( (1e-20,1e-9) )
ax_EE.set_xscale( 'log' )
ax_EE.set_yscale( 'log' )

ax_BB = fig.add_subplot( 236 )
ax_BB.set_title( r'BB Power Spectra' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_xlim( (2,lmax) )
ax_BB.set_ylim( (1e-18,1e-14) )
ax_BB.set_xscale( 'log' )
ax_BB.set_yscale( 'log' )
#ax_BB.set_yscale( 'symlog', linthreshy=1e-16 )

ax_TT.plot( ell2*_TTo, label='input $C_\ell^{\mathrm{TT}}$' ) 
ax_TT.plot( ell2*TTo/wl_TT, label='healpy $C_\ell^{\mathrm{TT}}$' ) 
ax_TT.plot( ell2*TT/wl_EE , label='PISCO $C_\ell^{\mathrm{TT}}$' ) 
ax_TT.legend()

ax_EE.plot( ell2*_EEo, label='input $C_\ell^{\mathrm{EE}}$' ) 
ax_EE.plot( ell2*EEo/wl_EE, label='healpy $C_\ell^{\mathrm{EE}}$' ) 
ax_EE.plot( ell2*EE/wl_EE , label='PISCO $C_\ell^{\mathrm{EE}}$' ) 
ax_EE.legend()

ax_BB.plot( ell2*_BBo, label='input $C_\ell^{\mathrm{BB}}$') 
ax_BB.plot( ell2*BBo/wl_BB, label='healpy $C_\ell^{\mathrm{BB}}$') 
ax_BB.plot( ell2*BB/wl_BB , label='PISCO $C_\ell^{\mathrm{BB}}$' ) 
ax_BB.legend()

# Show images
ax_I.set_title( 'PISCO recovered Stokes I' )
imageI = ax_I.imshow( I_proj,
               vmin=-5e-5, vmax=5e-5,
               cmap=plt.get_cmap('gray'),
               extent=( (-180,180,-90,90) ),
               origin='lower')
pylab.colorbar( imageI, ax=ax_I, orientation='horizontal', ticks=[-5E-5,0,5E-5] , format='%+2.2e' ) 

ax_Q.set_title( 'PISCO recovered Stokes Q' )
imageQ = ax_Q.imshow( Q_proj,
               vmin=-5e-7, vmax=5e-7,
               cmap=plt.get_cmap('gray'),
               extent=( (-180,180,-90,90) ),
               origin='lower')
pylab.colorbar( imageQ, ax=ax_Q, orientation='horizontal', ticks=[-5E-7,0,5E-7], format='%+2.2e' )

ax_U.set_title( 'PISCO recovered Stokes U' )
imageU = ax_U.imshow( U_proj,
               vmin=-5e-7, vmax=5e-7,
               cmap=plt.get_cmap('gray'),
               extent=( (-180,180,-90,90) ),
               origin='lower')
pylab.colorbar( imageU, ax=ax_U, orientation='horizontal', ticks=[-5E-7,0,5E-7], format='%+2.2e' )
pylab.show()

