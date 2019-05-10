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

# Do this shit right with argparser
'''
import argparse                                                                                               
parser = argparse.ArgumentParser(description='Compute power spectra from output of run_fp_sim_with_(...).')
                                                                                                              
parser.add_argument( '-path_to_matrices_file' , action='store', type=str, dest='pathToMatrices',
                      help='Base path to matrices. Must be in format <path>/<filename>.npz' )

parser.add_argument( '-path_to_input_map' , action='store', type=float, dest='pathToInputMap',
                      help='Value of r for the input map' )                                                   
                                                                                                              
parser.add_argument( '-' , action='store', type=str, dest='tag',                                           
                      help='' )                                                                               
                                                                                                              
parser.add_argument( '-pointing', action='store', type=str, dest='pointing',                                  
                      help='NPZ file with the pointing of the season. See generete_pointing.py for more details.' )
                                                                                                              
parser.add_argument( '-array_data', action='store', type=str, dest='array_data',                              
                      help='CSV file with the array specificiations.' )                                       
                                                                                                              
parser.add_argument( '-beam_par_file',  action='store', type=str, dest='beam_par_file',                       
                      help='File with specifications of the beam parameters for each detector.')              
                                                                                                              
args = parser.parse_args() 
'''
# Across the code, there is o<variable name>, u<variable name> and p<variable name>.
# The meaning is
#
# o : original, no smoothing
# p : PISCO
# s : "smoothed" (convolved using smoothing)
#

# Load data from user input
data = numpy.load( sys.argv[1] )

# Load beam FWHM from user input, in degrees
beam_fwhm = (float)( sys.argv[3] )

lmax = 250

# Compute PISCO maps
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
pI,pQ,pU,pW = matrices_to_maps( NSIDE, AtA, AtD )

# Read original maps from disk
oI = numpy.load( sys.argv[2] )['I']
oQ = numpy.load( sys.argv[2] )['Q'] 
oU = numpy.load( sys.argv[2] )['U'] 

# Convolve original maps using smoothing
sI, sQ, sU = healpy.smoothing( (oI, oQ, oU), fwhm=numpy.radians(beam_fwhm), pol=True )

# Create a window from hit map. Mask all pixels with less than 10 hits.
mask = numpy.where( pW < 10 )
w    = numpy.ones_like( pW, dtype='float' )
w[ mask ] = 0.0

# DO NOT read CLASS window because is messes up PS
#w = healpy.read_map( './data/mask.fits' )

# Compute power spectra
oTT, oEE, oBB, _, _, _ = spice( (oI, oQ, oU ), window=w, decouple=True )
sTT, sEE, sBB, _, _, _ = spice( (sI, sQ, sU ), window=w, decouple=True )
pTT, pEE, pBB, _, _, _ = spice( (pI, pQ, pU ), window=w, decouple=True )

# Adjust to the lmax parameter, because Spice doesn't have that option
oTT = oTT[0:lmax+1]
oEE = oEE[0:lmax+1]
oBB = oBB[0:lmax+1]

sTT = sTT[0:lmax+1]
sEE = sEE[0:lmax+1]
sBB = sBB[0:lmax+1]

pTT = pTT[0:lmax+1]
pEE = pEE[0:lmax+1]
pBB = pBB[0:lmax+1]

# Define l numbers
ell = numpy.arange( oTT.size )
ell2 = ell * (ell+1)/(2*numpy.pi)

# Get map Pixel Window function
# I am not entirely sure how to use this, but I was told is just another window function
# that multiplies the one from the beam
pixwin_temp, pixwin_pol = healpy.pixwin( NSIDE, pol=True )
pixwin_temp = pixwin_temp[0:lmax+1]
pixwin_pol  = pixwin_pol [0:lmax+1]

# Get window function from a circularly symmetric Gaussian beam.
# Note we are getting both polarization and temperature ones.
glTT, glEE, glBB, glTE = healpy.sphtfunc.gauss_beam( numpy.radians(beam_fwhm), pol=True, lmax=lmax ).T
# I am not (and never was, because I am not sure about it) including the effect of pixwin here.
wl_TT = (glTT**2)# * pixwin_temp ) 
wl_EE = (glEE**2)# * pixwin_pol  )
wl_BB = (glBB**2)# * pixwin_pol  )

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
ax_BB.set_xscale('log')

# Plot C_\ell^{TT} for original maps
ax_TT.plot( ell2*oTT, c='black', alpha=0.6,
            label='$C_{\ell}^{TT}$ of input CMB.' ) 

# Plot C_\ell^{TT} of PISCO output, divided by a Circularly Symmetric Gaussian (csg) window function
ax_TT.plot( ell2*pTT/wl_TT, c='blue', alpha=0.6,
            label='$C_{\ell}^{TT} / w_{\ell}^{\mathrm{csg}}$ of PISCO output.' ) 

# Plot C_\ell^{TT} of smoothed original maps, divided by a Circularly Symmetric Gaussian (csg) window function
ax_TT.plot( ell2*sTT/wl_TT, c='red', alpha=0.6,
            label='$C_{\ell}^{TT} / w_{\ell}^{\mathrm{csg}}$ of smoothed input CMB.' ) 

# Plot C_\ell^{EE} for original maps
ax_EE.plot( ell2*oEE, c='black', alpha=0.6,
            label='$C_{\ell}^{EE}$ of input CMB.' ) 
# Plot C_\ell^{EE} of PISCO output, divided by a Circularly Symmetric Gaussian (csg) window function
ax_EE.plot( ell2*pEE/wl_EE, c='blue', alpha=0.6,
            label='$C_{\ell}^{EE} / w_{\ell}^{\mathrm{csg}}$ of PISCO output.' ) 

# Plot C_\ell^{EE} of smoothed original maps, divided by a Circularly Symmetric Gaussian (csg) window function
ax_EE.plot( ell2*sEE/wl_EE, c='red', alpha=0.6,
            label='$C_{\ell}^{EE} / w_{\ell}^{\mathrm{csg}}$ of smoothed input CMB.' ) 
# Plot C_\ell^{BB} for original maps
ax_BB.plot( ell2*oBB, c='black', alpha=0.6,
            label='$C_{\ell}^{BB}$ of input CMB.' ) 
# Plot C_\ell^{BB} of PISCO output, divided by a Circularly Symmetric Gaussian (csg) window function
ax_BB.plot( ell2*pBB/wl_BB, c='blue', alpha=0.6,
            label='$C_{\ell}^{BB} / w_{\ell}^{\mathrm{csg}}$ of PISCO output.' ) 

# Plot C_\ell^{BB} of smoothed original maps, divided by a Circularly Symmetric Gaussian (csg) window function
ax_BB.plot( ell2*sBB/wl_BB, c='red', alpha=0.6,
            label='$C_{\ell}^{BB} / w_{\ell}^{\mathrm{csg}}$ of smoothd input CMB.' ) 
ax_TT.legend()
ax_EE.legend()
ax_BB.legend()

pylab.show()


