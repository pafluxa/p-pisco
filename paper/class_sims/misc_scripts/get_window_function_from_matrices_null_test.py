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

# Define lmax =250
lmax = 250

# Create a window
window = numpy.zeros_like( W , dtype='bool' )
'''
# Mask pixels with bad coverage
window[ W <  100 ] = True
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
'''
'''Compute power specra'''
TT, EE, BB, TE, EB, TB       = healpy.anafast( (I     ,Q     ,U     ), pol=True, lmax=lmax )
TTo, EEo, BBo, TEo, EBo, TBo = healpy.anafast( (I_orig,Q_orig,U_orig), pol=True, lmax=lmax )

# Define l numbers
ell = numpy.arange( TT.size )
ell2 = ell * (ell+1)/(2*numpy.pi)

# Compute window function of circularly symmetric Gaussian beam of 1.5 degrees FWHM
beam_fwhm = 1.5
glTT, glEE, glBB, glTE = healpy.sphtfunc.gauss_beam( numpy.radians(beam_fwhm), pol=True, lmax=lmax ).T
wl_TTo = (glTT**2)
wl_EEo = (glEE**2)
wl_BBo = (glBB**2)

# Compute window function of PISCO by dividing the input power spectra by the one obtained from PISCO maps
wl_TT = (TT/TTo)
wl_EE = (EE/EEo)
wl_BB = (BB/BBo)

fig  = pylab.figure( figsize=(12,4) )

ax_TT = fig.add_subplot( 131 )
ax_TT.set_title( r'TT window function' )
ax_TT.set_xlabel( r'$\ell$' )
ax_TT.set_ylabel( r'K$^2$' )
ax_TT.set_ylim( (0,1) )
ax_TT.set_xlim( (2,lmax) )

ax_EE = fig.add_subplot( 132, sharey=ax_TT )
ax_EE.set_title( r'EE window function' )
ax_EE.set_xlabel( r'$\ell$' )
ax_EE.set_xlim( (2,lmax) )
ax_EE.set_ylim( (0,1) )

ax_BB = fig.add_subplot( 133 , sharey=ax_TT )
ax_BB.set_title( r'BB window function' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_xlim( (2,lmax) )
ax_BB.set_ylim( (0,1) )

ax_TT.plot( ell, wl_TT, label='PISCO $w^{\mathrm{TT}}_\ell$') 
ax_TT.plot( ell, wl_TTo, label='Analytical $w^{\mathrm{TT}}_\ell$') 
ax_TT.legend()

ax_EE.plot( ell, wl_EE, label='PISCO $w^{\mathrm{EE}}_\ell$') 
ax_EE.plot( ell, wl_EEo, label='Analytical $w^{\mathrm{EE}}_\ell$') 

ax_EE.legend()

ax_BB.plot( ell, wl_BB, label='PISCO $w^{\mathrm{BB}}_\ell$') 
ax_BB.plot( ell, wl_BBo, label='Analytical $w^{\mathrm{BB}}_\ell$') 
ax_BB.legend()

pylab.show()


