#!/usr/bin/env python
# coding: utf-8
import pisco
from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.mapping.core import * 

import time
import numpy
import healpy
import pylab


# Setup scan properties
nsamples_per_scan = 600
nscans = 200
nsamples = nsamples_per_scan * nscans

# Setup an input map with a disk at (0,0) in healpix
map_nside = 256 
I_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
Q_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
U_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
# Paint a point source in the middle
origin  = healpy.ang2pix( map_nside, numpy.pi/2.0, 0 )
I_map[ origin ] = 1.0

# Setup a simple Gaussian beam 
beam_nside  = 512
beam_fwhm   = 1.5
beam_extn   = 10
# Beam for feedhorn 1
beam1_co = 1.0*make_gaussian_beam( beam_nside, beam_fwhm, max_theta=beam_extn )
beam1_cx = 0.1*make_gaussian_crosspolar_beam( beam_nside, beam_fwhm, max_theta=beam_extn )

# Beam for feedhorn 2
beam2_co = 1.0*make_gaussian_beam( beam_nside, beam_fwhm, max_theta=beam_extn )
beam2_cx = 0.1*make_gaussian_crosspolar_beam( beam_nside, beam_fwhm, max_theta=beam_extn )

# Setup map matrices
AtA = None
AtD = None

# Setup first raster Scan
ra   = numpy.linspace( -0.1, 0.1, nsamples_per_scan )
dec  = numpy.linspace( -0.1, 0.1, nsamples_per_scan )
pa   = numpy.zeros   ( nsamples_per_scan )
ra   = numpy.repeat( ra   , nscans ).ravel()
dec  = numpy.tile  ( dec  , nscans ).ravel()
pa   = numpy.repeat( pa   , nscans ).ravel()

arr_ra   = numpy.asarray( [ ra]   )
arr_dec  = numpy.asarray( [dec]   )
arr_pa   = numpy.asarray( [ pa]   )

start = time.time()
# Run deprojection of raster scan
# Setup detector polarization angles
det_pol_angles = numpy.asarray( [0.0] )
det1 = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, (beam1_co,beam1_cx), (beam2_co, beam2_cx),
    gpu_dev=0 )
end = time.time()

print 'GPU execution time : ', end - start
arr_data  = numpy.asarray( [det1] )
arr_pol_angles = numpy.asarray( [ det_pol_angles ] )

# Map scan 
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_data, map_nside, AtA=AtA, AtD=AtD ) 

# Setup detector polarization angles
det_pol_angles = numpy.asarray( [numpy.pi/4.0] )

start = time.time()
# Run deprojection of raster scan
det1 = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, (beam1_co,beam1_cx), (beam2_co, beam2_cx),
    gpu_dev=0 )
end = time.time()

print 'GPU execution time : ', end - start
arr_data  = numpy.asarray( [det1] )
arr_pol_angles = numpy.asarray( [ det_pol_angles ] )
# Map scan
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_data, map_nside, AtA=AtA, AtD=AtD ) 

# Setup detector polarization angles
det_pol_angles = numpy.asarray( [-numpy.pi/4.0] )

start = time.time()
# Run deprojection of raster scan
det1 = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, (beam1_co,beam1_cx), (beam2_co, beam2_cx),
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
healpy.gnomview( I , xsize=100, ysize=100, reso=10, min= 0.0, max=1.0 , sub=(1,4,1) )                                                    
healpy.gnomview( Q , xsize=100, ysize=100, reso=10, min=-0.1, max=0.1 , sub=(1,4,2) )                                                    
healpy.gnomview( U , xsize=100, ysize=100, reso=10, min=-0.1, max=0.1 , sub=(1,4,3) )                                                    
healpy.gnomview( W , xsize=100, ysize=100, reso=10,                     sub=(1,4,4) )      
pylab.show()                                                                                                  
