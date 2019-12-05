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
import healpy as hp

from pisco.mapping.core import matrices_to_maps

from composite_elliptical_beam_window_function import make_composite_elliptical_beam_window_function

import os

# PolSpice wrapper from Duncan Watts (2019)
#
def spice_wrap2(m1, m2, weights=None, lmax=300):

    clfile = 'temp_cls.txt'
    mapfile1 = 'temp_m1.fits'
    mapfile2 = 'temp_m2.fits'
    weightfile = 'temp_mask.fits'

    hp.write_map(mapfile1, m1, overwrite=True)
    hp.write_map(mapfile2, m2, overwrite=True)
    hp.write_map(weightfile, weights, overwrite=True)

    command = [
        '/home/pafluxa/software/PolSpice_v03-05-01/build/spice',
        '-mapfile', mapfile1,
        '-mapfile2', mapfile2,
        '-verbosity', '0',
        '-weightfile', weightfile,
        '-nlmax', str(lmax),
        '-overwrite', 'NO',
        '-polarization', 'YES',
        '-decouple', 'YES',
        '-symmetric_cl', 'YES', # averages T_map1*E_map2 and T_map2*E_map1, etc.
        '-clfile', clfile,
        '-tolerance', '1e-7'
        ]

    os.system(' '.join(command))
    ell, TT, EE, BB, TE, TB, EB = np.loadtxt(clfile,unpack=True)
    os.system('rm temp*')

    return ell,TT,EE,BB,TE,TB,EB

# Across the code, there is o<variable name>, u<variable name> and p<variable name>.
# The meaning is
#
# o : original, no smoothing
# p : PISCO
# s : "smoothed" (convolved using smoothing)
#

# Load data from user input
data = numpy.load( sys.argv[1] )

#----------------------------------------------------------------------------------------------------------#
# Read beam parameter file
print 'reading beam parameters'
beam_data = pandas.read_csv( sys.argv[2] )
feeds     = numpy.array( beam_data[   'Feed'] )
azOff     = numpy.array( beam_data[  'AzOff'] )
elOff     = numpy.array( beam_data[  'ElOff'] )
fwhm_x    = numpy.array( beam_data[ 'FWHM_x'] )
fwhm_y    = numpy.array( beam_data[ 'FWHM_y'] )
rotation  = numpy.array( beam_data[  'Theta'] )
#----------------------------------------------------------------------------------------------------------#

# Compute PISCO maps
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
pI,pQ,pU,pW = matrices_to_maps( NSIDE, AtA, AtD )
print( NSIDE )

# create a coverage mask
covMask = numpy.zeros_like( pW, dtype=bool )
tht,_ = healpy.pix2ang( NSIDE, numpy.arange( pW.size ) )
dec   = 90 - numpy.rad2deg( tht )
covMask = numpy.where( numpy.logical_and( dec < 30, dec > -65 ) )
seenPixels = pW[ covMask ]
#print numpy.sum( seenPixels < 1 ), seenPixels.size

# Read original maps from disk
oI = numpy.load( sys.argv[3] )['I']
oQ = numpy.load( sys.argv[3] )['Q']
oU = numpy.load( sys.argv[3] )['U']

lmax = 750
# Define l numbers                                                                                            
ell = numpy.arange( lmax + 1 )                                                                                
ell2 = ell * (ell+1)/(2*numpy.pi)   

# Get window function considering all beams in the focal plane                                                

print fwhm_x, fwhm_y
w, beam_fwhm = make_composite_elliptical_beam_window_function( fwhm_x, fwhm_y, lmax, pol=True )               
glTT_mkb, glEE_mkb, glBB_mkb, glTE_mkb = w                                                                    
wl_TT_mkb = (glTT_mkb**2)# * pixwin_temp ) 
wl_EE_mkb = (glEE_mkb**2)# * pixwin_pol  )                                                                    
wl_BB_mkb = (glBB_mkb**2)# * pixwin_pol  )                                                                    
                                                                                                           
# Get window function of equivalent Gaussian beam
#beam_fwhm = 1.5
glTT, glEE, glBB, glTE = healpy.sphtfunc.gauss_beam( numpy.radians(beam_fwhm), pol=True, lmax=lmax ).T

ell2 = 1#ell2[2:lmax+1]

glTT = glTT[2:lmax+1]
glEE = glEE[2:lmax+1]
glBB = glBB[2:lmax+1]

pixwin_temp, pixwin_pol = healpy.pixwin( NSIDE, pol=True )
pixwin_temp = pixwin_temp[2:lmax+1]
pixwin_pol  = pixwin_pol [2:lmax+1]

wl_TT_mkb = (glTT**2) * pixwin_temp 
wl_EE_mkb = (glEE**2) * pixwin_pol                                                                     
wl_BB_mkb = (glBB**2) * pixwin_pol                                                                     

wl_TT = (glTT**2) * pixwin_temp
wl_EE = (glEE**2)
wl_BB = (glBB**2)

# Convolve original maps using smoothing
sI, sQ, sU = healpy.smoothing( (oI, oQ, oU), fwhm=numpy.radians(beam_fwhm), pol=True )

healpy.mollview( pI - sI, min=-1e-7, max=1e-7 )
plt.show()

w = numpy.ones_like( pI )

_,oTT,oEE,oBB,oTE,oTB,oEB = spice_wrap2( (oI,oQ,oU), (oI,oQ,oU), weights=w, lmax=lmax )
_,sTT,sEE,sBB,sTE,sTB,sEB = spice_wrap2( (sI,sQ,sU), (sI,sQ,sU), weights=w, lmax=lmax )
_,pTT,pEE,pBB,pTE,pTB,pEB = spice_wrap2( (pI,pQ,pU), (pI,pQ,pU), weights=w, lmax=lmax )


# Adjust to the lmax parameter, because Spice doesn't have that option
oTT = oTT[2:lmax+1]
oEE = oEE[2:lmax+1]
oBB = oBB[2:lmax+1]

sTT = sTT[2:lmax+1]
sEE = sEE[2:lmax+1]
sBB = sBB[2:lmax+1]

pTT = pTT[2:lmax+1]
pEE = pEE[2:lmax+1]
pBB = pBB[2:lmax+1]

fig  = pylab.figure( figsize=(20,5) )

ax_TT = fig.add_subplot( 131 )
ax_TT.set_title( r'TT Power Spectra' )
ax_TT.set_xlabel( r'$\ell$' )
ax_TT.set_ylabel( r'K$^2$' )
ax_TT.set_ylim( (1e-10,1e-6) )
ax_TT.set_xlim( (2,lmax) )
ax_TT.set_yscale('symlog', linthreshy=1e-11)
ax_TT.set_xscale('log')

ax_EE = fig.add_subplot( 132 )
ax_EE.set_title( r'EE Power Spectra' )
ax_EE.set_xlabel( r'$\ell$' )
ax_EE.set_xlim( (2,lmax) )
ax_EE.set_ylim( (-1e-16,1e-13) )
ax_EE.set_yscale('symlog', linthreshy=1e-16)
ax_EE.set_xscale('log')

ax_BB = fig.add_subplot( 133 )
ax_BB.set_title( r'BB Power Spectra' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_xlim( (2,lmax) )
ax_BB.set_ylim( (-1e-16,1e-13) )
ax_BB.set_yscale('symlog', linthreshy=1e-16)
ax_BB.set_xscale('log')

# Plot C_\ell^{TT} for original maps
ax_TT.plot( ell2*oTT, c='black', alpha=1.0,
            label='$C_{\ell}^{TT}$ of input CMB.' )
# Plot C_\ell^{TT} of PISCO output, divided by a Circularly Symmetric Gaussian (csg) window function from MKB
ax_TT.plot( ell2*pTT/(wl_TT_mkb), c='blue', alpha=0.6,
            label='$C_{\ell}^{TT} / w_{\ell}^{\mathrm{mkb}}$ of PISCO output.' )
# Plot C_\ell^{TT} of smoothed original maps, divided by a Circularly Symmetric Gaussian (csg) window function
ax_TT.plot( ell2*sTT/wl_TT, c='red', alpha=0.6,
            label='$C_{\ell}^{TT} / w_{\ell}^{\mathrm{csg}}$ of smoothed input CMB.' )

# Plot C_\ell^{EE} for original maps
ax_EE.plot( oEE, c='black', alpha=1.0,
            label='$C_{\ell}^{EE}$ of input CMB.' )
# Plot C_\ell^{EE} of PISCO output, divided by a Circularly Symmetric Gaussian (csg) window function
ax_EE.plot( pEE/(wl_EE_mkb), c='blue', alpha=0.6,
            label='$C_{\ell}^{EE} / w_{\ell}^{\mathrm{mkb}}$ of PISCO output.' )
# Plot C_\ell^{EE} of smoothed original maps, divided by a Circularly Symmetric Gaussian (csg) window function
ax_EE.plot( sEE/wl_EE, c='red', alpha=0.6,
            label='$C_{\ell}^{EE} / w_{\ell}^{\mathrm{csg}}$ of smoothed input CMB.' )

# Plot C_\ell^{BB} for original maps
ax_BB.plot( oBB, c='black', alpha=1.0,
            label='$C_{\ell}^{BB}$ of input CMB.' )
# Plot C_\ell^{BB} of PISCO output, divided by a Circularly Symmetric Gaussian (csg) window function
ax_BB.plot( pBB/(wl_BB_mkb), c='blue', alpha=0.6,
            label='$C_{\ell}^{BB} / w_{\ell}^{\mathrm{mkb}}$ of PISCO output.' )
# Plot C_\ell^{BB} of smoothed original maps, divided by a Circularly Symmetric Gaussian (csg) window function
ax_BB.plot( sBB/wl_BB, c='red', alpha=0.6,
            label='$C_{\ell}^{BB} / w_{\ell}^{\mathrm{csg}}$ of smoothed input CMB.' )
ax_TT.legend()
ax_EE.legend()
ax_BB.legend()

pylab.show()

