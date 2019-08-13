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
Q_map   = map_data['Q']
U_map   = map_data['U']
V_map   = map_data['V']

# Setup a simple Gaussian beam of 10 arcmin
beam_nside  = 101
beam_fwhm   = 1.5
beam_extn   = numpy.radians(7.0) # Extension of 5 degrees

# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )
# Normalize maps by the beam sum
beam0_cx  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.00)

beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )
beam90_cx = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.00 )

# Get alm space smoothed input maps
I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(beam_fwhm), pol=True )

# Setup ring scan
scan_pixels = numpy.arange( healpy.nside2npix( map_nside ) )
theta, phi  = healpy.pix2ang( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta
pa   = numpy.zeros_like( ra )

# perform deprojection aliged with the source
AtA_pa,AtD_pa = 0.0,0.0
TOD_pa = []
scan_angles = numpy.linspace( 0, pi, 4 )
det_angles  = numpy.zeros( 4 )
for scan_angle, det_angle in zip( scan_angles, det_angles ):

    tod = deproject_sky_for_feedhorn(
        ra, dec, pa + scan_angle,
        det_angle,
        (I_map,Q_map,U_map,V_map),
        beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx , gpu_dev=1)
    
    ata, atd = update_matrices( arr_ra, arr_dec, arr_pa + scan_angle, arr_pol_angles, arr_pisco_tod, map_nside ) 
    AtA_pa += ata
    AtD_pa += atd

AtA_pa_m = AtA_pa.reshape( (-1,3,3) ) 
AtD_pa_v = AtD_pa.reshape( (-1,3) ) 
TOD_pa = numpy.asarray( TOD_pa )

# Build PISCO convolved maps
I_pa,Q_pa,U_pa,W_pa = matrices_to_maps( map_nside, AtA_pa, AtD_pa )

# Filter out high frequency noise
I_pa,Q_pa,U_pa = healpy.smoothing( (I_pa,Q_pa,U_pa), sigma=numpy.radians(180.0/200.0/3.0), pol=False )

# perform deprojection aliged with the source
AtA_det,AtD_det = 0.0,0.0
TOD_det = []
scan_angles  = numpy.zeros( 4 )
det_angles = numpy.linspace( 0, pi, 4 )
for scan_angle, det_angle in zip( scan_angles, det_angles ):

    pa   = numpy.zeros_like( ra ) + scan_angle
    tod = deproject_sky_for_feedhorn(
        ra, dec, pa,
        det_angle,
        (I_map,Q_map,U_map,V_map),
        beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx , gpu_dev=1)
    TOD_det.append( tod )
    arr_pa   = numpy.asarray( [ pa]   )
    arr_ra   = numpy.asarray( [ ra]   )
    arr_dec  = numpy.asarray( [dec]   )
    arr_pol_angles = numpy.asarray( [ det_angle ] )
    arr_pisco_tod = numpy.asarray( [tod] )
    ata, atd = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside ) 
    AtA_det += ata
    AtD_det += atd

AtA_det_m = AtA_det.reshape( (-1,3,3) ) 
AtD_det_v = AtD_det.reshape( (-1,3) ) 
TOD_det   = numpy.asarray( TOD_det )
'''
pylab.figure()
healpy.mollview( I_pa     , sub=(3,2,1) )
healpy.mollview( I_smooth , sub=(3,2,2) )

healpy.mollview( Q_pa     , sub=(3,2,3) )
healpy.mollview( Q_smooth , sub=(3,2,4) )

healpy.mollview( U_pa     , sub=(3,2,5) )
healpy.mollview( U_smooth , sub=(3,2,6) )
'''
'''
# Get PS from PISCO maps
c = 1.0
cls = healpy.anafast( (I_pa*c,Q_pa*c,U_pa*c), pol=True )
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
'''
#pylab.show()

