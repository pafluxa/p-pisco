# coding: utf-8
import sys
import os

import numpy
from   numpy import pi
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

lmax = 250

# Smooth original maps for comparison
I_o_s, Q_o_s, U_o_s = healpy.smoothing( (I_orig, Q_orig, U_orig), fwhm=numpy.radians(beam_fwhm), pol=True )

# Create a window from hit map. Mask all pixels with less than 10 hits.
# No apodization.
mask = numpy.where( W < 10 )
w    = numpy.ones_like( W, dtype='float' )
w[ mask ] = 0.0

# DO NOT read CLASS window because is messes up PS
#w = healpy.read_map( './data/mask.fits' )

healpy.mollview( (I)     * w, min=I_o_s.min(), max=I_o_s.max(), title='PISCO I', sub=(2,3,1) )
healpy.mollview( (I_o_s) * w, min=I_o_s.min(), max=I_o_s.max(), title='healpy I', sub=(2,3,4) )

healpy.mollview( (Q)     * w, min=Q_o_s.min(), max=Q_o_s.max(), title='PISCO Q', sub=(2,3,2) )
healpy.mollview( (Q_o_s) * w, min=Q_o_s.min(), max=Q_o_s.max(), title='healpy Q', sub=(2,3,5) )

healpy.mollview( (U)     * w, min=U_o_s.min(), max=U_o_s.max(), title='PISCO U', sub=(2,3,3) )
healpy.mollview( (U_o_s) * w, min=U_o_s.min(), max=U_o_s.max(), title='healpy U', sub=(2,3,6) )

pylab.show()

'''Compute power specra'''
TT, EE, BB, TE, EB, TB       = spice( (I    ,Q    ,U    ), window=w, decouple=True ) 
TTo, EEo, BBo, TEo, EBo, TBo = spice( (I_o_s,Q_o_s,U_o_s), window=w, decouple=True )

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
wl_TT = (glTT**2)# * pixwin_temp ) 
wl_EE = (glEE**2)# * pixwin_pol  )
wl_BB = (glBB**2)# * pixwin_pol  )

# Load BB with r=0.1 for comparison
ls_in,cl_TT_in,cl_EE_in,cl_BB_in,cl_TE_in = numpy.loadtxt( 'data/cls/lcdm_cls_r=0.1000.dat' )

# Define spherical projection axis
vec2pix_func = lambda x,y,z: vec2pix( NSIDE,x,y,z,nest=False)

fig  = pylab.figure( figsize=(20,5) )

ax_TT = fig.add_subplot( 131 )
ax_TT.set_title( r'TT Power Spectra' )
ax_TT.set_xlabel( r'$\ell$' )
ax_TT.set_ylabel( r'K$^2$' )
ax_TT.set_ylim( (-1e-11,5e-10) ) 
ax_TT.set_xlim( (2,lmax) )
ax_TT.set_yscale('symlog', linthreshy=1e-11)
ax_TT.set_xscale('log')

ax_EE = fig.add_subplot( 132 )
ax_EE.set_title( r'EE Power Spectra' )
ax_EE.set_xlabel( r'$\ell$' )
ax_EE.set_xlim( (2,lmax) )
ax_EE.set_ylim( (-1e-15,1e-13) )
ax_EE.set_yscale('symlog', linthreshy=1e-15)
ax_EE.set_xscale('log')

ax_BB = fig.add_subplot( 133 )
ax_BB.set_title( r'BB Power Spectra' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_xlim( (2,lmax) )
ax_BB.set_ylim( (-1e-16,1e-14) )
ax_BB.set_yscale('symlog', linthreshy=1e-16)
#ax_BB.set_yscale('log')
ax_BB.set_xscale('log')

'''
ax_TT.plot( ell2*TTo/wl_TT, label='' ) 
ax_TT.plot( ell2*TT/wl_TT , label='PISCO rec. $C_{\ell}^{TT}$' ) 
ax_TT.legend()
'''

ax_EE.plot( ell2*EEo/wl_EE, label='' ) 
ax_EE.plot( ell2*EE/wl_EE , label='PISCO rec. $C_{\ell}^{EE}$' ) 
ax_EE.plot( cl_EE_in[0:351]/(2*pi) , label='' )
ax_EE.legend()

ax_BB.plot( ell2*BBo/wl_BB, label='') 
ax_BB.plot( ell2*BB /wl_BB, label='PISCO rec. $C_{\ell}^{BB}$' ) 
ax_BB.plot( cl_BB_in[0:351]/(2*pi) , label='' )
ax_BB.legend()

pylab.show()


