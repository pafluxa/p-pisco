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
Q_orig = numpy.load( sys.argv[2] )['Q']*0 
U_orig = numpy.load( sys.argv[2] )['U']*0

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

'''Compute power specra'''
TT, EE, BB, TE, EB, TB       = spice( (I    ,Q    ,U    ), window='./data/mask.fits', decouple=True ) 
TTo, EEo, BBo, TEo, EBo, TBo = spice( (I_o_s,Q_o_s,U_o_s), window='./data/mask.fits', decouple=True )

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

# dump everything as numpy array
numpy.savez( 'class_pisco_sim_ps_T_only.npz',
             ell=ell,                 # ell number
             wl_pix_TT=pixwin_temp,   # temperature pixel window function
             wl_pix_EE=pixwin_pol,     # E-mode pixel window function
             wl_pix_BB=pixwin_pol,     # B-mode pixel window function
             wl_TT=wl_TT,             # T window function
             wl_EE=wl_EE,             # E window function
             wl_BB=wl_BB,             # B window function
             ps_TT_sim=TT,
             ps_EE_sim=EE,
             ps_BB_sim=BB,
             ps_TT_in=TTo,
             ps_EE_in=EEo,
             ps_BB_in=BBo )

