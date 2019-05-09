#!/usr/bin/env python
# coding: utf-8
import pisco
from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.mapping.core import *

import sys
import time
import numpy
from numpy import pi
import healpy
import pylab

# Load LCDM model with r = 0
map_data  = numpy.load( sys.argv[1] )
map_nside = map_data['nside'][()]

I_map   = map_data['I']
Q_map   = map_data['Q']*0
U_map   = map_data['U']*0 
V_map   = map_data['V']*0

# Setup a simple Gaussian beam of 10 arcmin
beam_nside  = 51
beam_fwhm   = 5.0
beam_extn   = numpy.radians(20.0) # Extension of 5 degrees

# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )
beam0_cx  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.00)

beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )
beam90_cx = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.00 )

# Setup ring scan
# Setup ring scan
scan_pixels = numpy.arange( healpy.nside2npix( map_nside ) )
theta, phi  = healpy.pix2ang( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta

# perform deprojection aliged with the source

AtA,AtD = 0.0,0.0
# Repeat the above scan with random detector angles, and add up the mapping matrices
for scan, scan_angle in enumerate( [0.0,pi/8.0,pi/4.0] ):

    print scan
    # Setup parallactic angle to be PI/4.0
    pa   = numpy.zeros_like( ra ) + scan_angle*numpy.random.random(1)
    det_pol_angle = scan_angle
    tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    scan_angle,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )
    
    arr_pa   = numpy.asarray( [pa]   )
    arr_ra   = numpy.asarray( [ ra]   )
    arr_dec  = numpy.asarray( [dec]   )
    arr_pol_angles = numpy.asarray( [ det_pol_angle ] )
    arr_pisco_tod = numpy.asarray( [tod] )
    ata, atd = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside ) 
    AtA += ata
    AtD += atd
    
# Build PISCO convolved maps
I_out,Q_out,U_out,W_out = matrices_to_maps( map_nside, AtA, AtD )
I = I_out
Q = Q_out
U = U_out

# Get alm space smoothed input maps
I_map,Q_map,U_map = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(beam_fwhm), pol=True )

healpy.mollview( I, sub=(3,2,1) )
healpy.mollview( I_map , sub=(3,2,2) )

healpy.mollview( Q, sub=(3,2,3) )
healpy.mollview( Q_map , sub=(3,2,4) )

healpy.mollview( U , sub=(3,2,5) )
healpy.mollview( U_map , sub=(3,2,6) )
pylab.show()

c = 1.0

# Get PS from PISCO maps
cls = healpy.anafast( (I*c,Q*c,U*c), pol=True )
TT, EE, BB, TE, EB, TB = cls

# Get PS from input maps
TT_alm, EE_alm, BB_alm, TE_alm, EB_alm, TB_alm = healpy.anafast(
    (I_map,
     Q_map,
     U_map), pol=True )



# Make some noise
ls = numpy.arange( TT.size )
fig = pylab.figure( figsize=(10,4) )
ax0 = fig.add_subplot( 131 )
ax0.plot( ls, ls*(ls+1)*TT     , label='TT pisco')
ax0.plot( ls, ls*(ls+1)*TT_alm , label='TT alm')
ax0.set_yscale( 'log' )
ax0.set_xlim( 2, 180.0/(beam_fwhm)*2 )
pylab.legend()

ax1 = fig.add_subplot( 132, sharey=ax0 )
ax1.plot( ls, ls*(ls+1)*EE         , label='EE pisco')
ax1.plot( ls, ls*(ls+1)*EE_alm     , label='EE alm')
ax1.set_xlim( 2, 180.0/(beam_fwhm)*2 )
pylab.legend()

ax2 = fig.add_subplot( 133, sharey=ax0 )
ax2.plot( ls, ls*(ls+1)*BB    , label='BB pisco')
ax2.plot( ls, ls*(ls+1)*BB_alm, label='BB_alm')
ax2.set_xlim( 2, 180.0/(beam_fwhm)*2 )

pylab.legend()
pylab.show()
