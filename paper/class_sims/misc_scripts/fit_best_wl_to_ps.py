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

from scipy.optimize import minimize

def ps_diff( beam_fwhm_deg, BB_pisco, I_o, Q_o, U_o, lmax ):
    
    # Define l numbers
    ell = numpy.linspace( 0, lmax, lmax+1 )[2::]
    ell2 = ell * (ell+1)/(2*numpy.pi)

    beam_fwhm_deg = (float)( beam_fwhm_deg )

    I_o, Q_o, U_o = healpy.smoothing( (I_o,Q_o,U_o), fwhm=numpy.radians(beam_fwhm_deg), pol=True )
    TT_o, EE_o, BB_o, _, _, _ = spice( (I_o,Q_o,U_o), window='./data/mask.fits', decouple=True )
    
    BB_o = BB_o[2:lmax+1]
    
    cl_BB_pisco = ell2 * BB_pisco
    cl_BB_o     = ell2 * BB_o
   
    res = numpy.sum( (cl_BB_pisco - cl_BB_o)**2 )

    return res
    
lmax = 250

# Load data from user input
data          = numpy.load( sys.argv[1] )
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
I,Q,U,W       = matrices_to_maps( NSIDE, AtA, AtD )

# read CLASS map window
w = healpy.read_map( './data/mask.fits' )

'''Compute power specra'''
TT_s, EE_s, BB_s, _, _, _ = spice( (I,Q,U), window='./data/mask.fits', decouple=False ) 
TT_p = TT_s[1:lmax+1]
EE_p = EE_s[2:lmax+1]
BB_p = BB_s[2:lmax+1]

# Define l numbers
ell = numpy.linspace( 0, lmax, lmax+1 )[2::]
ell2 = ell * (ell+1)/(2*numpy.pi)

# Setup plotting
fig   = pylab.figure()
ax_BB = fig.add_subplot( 111 )
ax_BB.set_title( r'$C_\ell^{BB}$' )
ax_BB.set_xscale( 'log' )
ax_BB.set_xlabel( r'$\ell$' )
ax_BB.set_xlim( (2,lmax) )


# load original maps
I_o = numpy.load( sys.argv[2] )['I']
Q_o = numpy.load( sys.argv[2] )['Q'] 
U_o = numpy.load( sys.argv[2] )['U'] 
    
#TT_o, EE_o, BB_o, _, _, _ = spice( (I_o,Q_o,U_o), window='./data/mask.fits', decouple=True )

cl_BB_p = ell2 * BB_p * 1e18
ax_BB.plot( cl_BB_p, label='$C_{\ell}^{BB} (PISCO)$' )

# smooth input maps by some gaussian beam
# = minimize( ps_diff, [1.0], (BB_s, I_o, Q_o, U_o, lmax), method='BFGS' )
#print r
#beam_fwhm_deg = float( r['x'] )

for b in [1.0,1.5,2.0,2.5]:
    
    beam_fwhm_deg = b
    I_o, Q_o, U_o = healpy.smoothing( (I_o,Q_o,U_o), fwhm=numpy.radians(beam_fwhm_deg), pol=True )
    TT_o, EE_o, BB_o, _, _, _ = spice( (I_o,Q_o,U_o), decouple=False, window='./data/mask.fits' )
    #bl_TT, bl_EE, bl_BB, bl_TE = healpy.sphtfunc.gauss_beam( numpy.radians(beam_fwhm_deg), lmax=lmax, pol=True ).T
    #BB_o_wl = BB_s / (bl_BB[2:lmax+1])**2
    
    _BB_o = BB_o[2:lmax+1]
    cl_BB_o = ell2 * _BB_o * 1e18

    ax_BB.plot( cl_BB_o, label='$C_{\ell}^{BB} \times w_\ell^2 (%1.1f)^\circ$' % (b) )
    #ax_BB.plot( cl_BB_o, label='fwhm=%2.2f' % (b) ) 

ax_BB.legend()
pylab.show()
