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

pa0 = numpy.radians( float(sys.argv[1]) )
eps = float(sys.argv[2])
ba  = numpy.radians( float(sys.argv[3]) )

# Setup an input map with a disk at (0,0) in healpix
map_nside = 256
I_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
Q_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
U_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )

# Paint a point source at the middle of the map
v = numpy.asarray( ( 1.0,-0.05,0) )
point_source_pixels = healpy.query_disc( map_nside, v, numpy.radians(0.5) )
I_map[ point_source_pixels ] = 1
Q_map[ point_source_pixels ] = 1

v = numpy.asarray( ( 1.0,0.05,0) )
point_source_pixels = healpy.query_disc( map_nside, v, numpy.radians(0.5) )
I_map[ point_source_pixels ] = 1
Q_map[ point_source_pixels ] = 1

# Setup a simple Gaussian beam
beam_nside  =  map_nside*8
beam_fwhm   =  3.0
beam_extn   =  9.0
# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm*eps, beam_extn, theta=ba, amplitude=1.0 ) 
beam0_cx  = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )

beam90_co = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm*eps, beam_extn, theta=ba, amplitude=1.0 )
beam90_cx = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )

# Setup map matrices
AtA = None
AtD = None

# Setup raster scan
v = numpy.asarray( (1,0,0) )
scan_pixels = healpy.query_disc( map_nside, v, numpy.radians(10) )
tht,phi     = healpy.pix2ang   ( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - tht
pa   = numpy.zeros_like(ra).ravel() + pa0

arr_ra   = numpy.asarray( [ ra] )
arr_dec  = numpy.asarray( [dec] )
arr_pa   = numpy.asarray( [ pa] )

start = time.time()
# Run deprojection of raster scan
# Setup detector polarization angles
det_pol_angles = numpy.asarray( [0.0] )
det1 = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam_extn, beam0_co,beam0_cx,beam90_co,beam90_cx,
    gpu_dev=0 )
end = time.time()
print 'GPU execution time : ', end - start

arr_data  = numpy.asarray( [det1] )
arr_pol_angles = numpy.asarray( [ det_pol_angles ] )
# Map scan
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_data, map_nside, AtA=AtA, AtD=AtD )

# Setup detector polarization angles
det_pol_angles = numpy.asarray( [pi/2.0] )

start = time.time()
# Run deprojection of raster scan
det1 = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx,beam90_co,beam90_cx,
    gpu_dev=0 )
end = time.time()

print 'GPU execution time : ', end - start
arr_data  = numpy.asarray( [det1] )
arr_pol_angles = numpy.asarray( [ det_pol_angles ] )
# Map scan
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_data, map_nside, AtA=AtA, AtD=AtD )

# Setup detector polarization angles
det_pol_angles = numpy.asarray( [-pi/2] )

start = time.time()
# Run deprojection of raster scan
det1 = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx,beam90_co,beam90_cx,
    gpu_dev=0 )
end = time.time()

print 'GPU execution time : ', end - start
arr_data  = numpy.asarray( [det1] )
arr_pol_angles = numpy.asarray( [ det_pol_angles ] )

# Map scan
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_data, map_nside, AtA=AtA, AtD=AtD )

# Project matrices to maps
I,Q,U,W = matrices_to_maps( map_nside, AtA, AtD )

# Show maps
tht_s, phi_s = 90,0
healpy.gnomview( I , xsize=100, ysize=100, reso=10 , sub=(2,3,1), rot=(phi_s, 90-tht_s) , title='Iconv')
healpy.gnomview( Q , xsize=100, ysize=100, reso=10 , sub=(2,3,2), rot=(phi_s, 90-tht_s) , title='Qconv')
healpy.gnomview( U , xsize=100, ysize=100, reso=10 , sub=(2,3,3), rot=(phi_s, 90-tht_s) , title='Uconv' )

healpy.gnomview( I_map , xsize=100, ysize=100, reso=10 , sub=(2,3,4), rot=(phi_s, 90-tht_s) , title='I in')
healpy.gnomview( Q_map , xsize=100, ysize=100, reso=10 , sub=(2,3,5), rot=(phi_s, 90-tht_s) , title='Q in')
healpy.gnomview( U_map , xsize=100, ysize=100, reso=10 , sub=(2,3,6), rot=(phi_s, 90-tht_s) , title='U in')
pylab.show()
