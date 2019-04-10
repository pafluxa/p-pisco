#!/usr/bin/env python
# coding: utf-8
from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn

import time
import numpy
import healpy
import pylab

# Setup scan properties
nsamples_per_scan = 100
nscans = 100
nsamples = nsamples_per_scan * nscans

# Setup an input map with a disk at (0,0) in healpix
map_nside = 128
I_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
Q_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
U_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros ( healpy.nside2npix( map_nside ) )

# Paint a point source at ra=0 dec=0
origin  = healpy.ang2pix( map_nside, numpy.pi/2.0, 0 )
I_map[ origin ] = 1.0

# Setup a simple Gaussian beam 
beam_nside  = 128
beam_fwhm   = 1.0
# Beam for feedhorn 1
beam0_co  = 1.0*make_gaussian_beam( beam_nside, beam_fwhm )
beam0_cx  = 0.0*make_gaussian_beam( beam_nside, beam_fwhm )

# Beam for feedhorn 2
beam90_co = 1.0*make_gaussian_beam( beam_nside, beam_fwhm )
beam90_cx = 0.0*make_gaussian_beam( beam_nside, beam_fwhm )

# Setup detector polarization angles
det_pol_angles = numpy.asarray( [0.0] )

# Setup a scan that visits all pixels
tht,ra = healpy.pix2ang( map_nside, numpy.arange(0,I_map.size,1) )
dec    = numpy.pi/2.0 - tht
pa     = numpy.zeros_like( ra )

# Run deprojection of raster scan
start = time.time()
tod  = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx,
    gpu_dev=0, grid_size=0.5 )
end = time.time()

print 'execution time : ', end - start

