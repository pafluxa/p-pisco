#!/usr/bin/env python
# coding: utf-8
import pisco
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.beam_analysis.utils import *
from pisco.mapping.core import *

import sys
import time
import numpy
from numpy import pi
import healpy

map_nside = 256
npix      = healpy.nside2npix( map_nside )
I_map     = numpy.zeros( npix )
Q_map     = numpy.zeros( npix )
U_map     = numpy.zeros( npix )
V_map     = numpy.zeros( npix )

# Setup a pint source
ra_s  =  45.0
dec_s =  45.0

print ra_s, dec_s

tht_s = numpy.radians( 90 - dec_s )
phi_s = numpy.radians( ra_s )
v     = healpy.ang2vec( tht_s, phi_s )
source_pixels = healpy.vec2pix( map_nside, v[0], v[1], v[2] )
theta, phi  = healpy.pix2ang   ( map_nside, source_pixels )
v     = healpy.ang2vec( theta, phi  )

outputName = ""
if sys.argv[1] == 'I':
    
    I_map[ source_pixels ] = 1.0
    Q_map[ source_pixels ] = 0.0
    U_map[ source_pixels ] = 0.0
    outputName = 'stokes_I.npz'
    
elif sys.argv[1] == 'Q':
    
    I_map[ source_pixels ] = 1.0
    Q_map[ source_pixels ] = 1.0
    U_map[ source_pixels ] = 0.0
    outputName = 'stokes_Q.npz'

elif sys.argv[1] == 'U':
    
    I_map[ source_pixels ] = 1.0
    Q_map[ source_pixels ] = 0.0
    U_map[ source_pixels ] = 1.0
    outputName = 'stokes_U.npz'

else:
    
    raise RuntimeError( "" )

# Setup a simple Gaussian beam of 1.5 degrees FWHM
beam_nside  = 1024
beam_fwhm   = 1.5
# Get alm space smoothed input maps
I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(beam_fwhm), pol=True )

# Beam for feedhorn
beam0_co  = make_gaussian_beam( beam_nside, beam_fwhm )
beam0_cx  = numpy.zeros_like( beam0_co )

# Beam for feedhorn 2
beam90_co = make_gaussian_beam( beam_nside, beam_fwhm )
beam90_cx = numpy.zeros_like( beam90_co )

# Setup scan
nscans      = 3
scan_pixels = healpy.query_disc( map_nside, v, numpy.radians(5.0) )
theta, phi  = healpy.pix2ang   ( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta
pa   = numpy.zeros_like( ra )

# perform deprojection aliged with the source
AtA_pa,AtD_pa = 0.0,0.0
det_angle   = 0.0

# IN RADIANS GOD DAMN IT!!
scan_angles = numpy.radians( numpy.linspace( -45,45, nscans ) )
for i,scan_angle in enumerate(scan_angles):
    
    print i+1

    tod = deproject_sky_for_feedhorn(
        ra, dec, pa + scan_angle,
        det_angle,
        (I_map,Q_map,U_map,V_map),
        beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx, gpu_dev=0 )
    

    arr_pa   = numpy.asarray( [ pa] + scan_angle   )
    arr_ra   = numpy.asarray( [ ra]   )
    arr_dec  = numpy.asarray( [dec]   )
    arr_pol_angles = numpy.asarray( det_angle )
    arr_pisco_tod = numpy.asarray( [tod] )
    ata, atd = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside ) 
    AtA_pa += ata
    AtD_pa += atd

# Build PISCO convolved maps
I_pa,Q_pa,U_pa,W_pa = matrices_to_maps( map_nside, AtA_pa, AtD_pa )

source_pixels = healpy.query_disc( map_nside, v, numpy.radians(5.0) )

print( "#DRA DDEC I_pisco Q_pisco U_pisco I_smoothing Q_smoothing U_smoothing" )
print( "#--------------------------------------------------------------------" )
for pix in source_pixels:
    
    d,r = numpy.degrees( healpy.pix2ang( map_nside, pix ) )
    d   = 90 - d
    
    if numpy.abs( d - dec_s ) < 0.5:
        print( "%2.2f %2.2f %2.4E %2.4E %2.4E %2.4E %2.4E %2.4E" % 
               (r - ra_s, d - dec_s, 
                I_pa[pix], Q_pa[pix], U_pa[pix],
                I_smooth[pix], Q_smooth[pix], U_smooth[pix] ) )
