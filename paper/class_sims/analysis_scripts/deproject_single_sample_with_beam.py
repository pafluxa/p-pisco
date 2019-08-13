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


map_nside = 128
I_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
Q_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
U_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros( healpy.nside2npix( map_nside ) )

# Setup a pure Q source
source_pixels = numpy.random.randint(0, healpy.nside2npix( map_nside ) )
S_in = (1,0,1,0)
I_map[ source_pixels ] = S_in[0]
Q_map[ source_pixels ] = S_in[1]
U_map[ source_pixels ] = S_in[2]

# Setup a simple Gaussian beam
beam_nside  = map_nside
beam_fwhm   =  4.0
beam_extn   = 12.0
# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.0, theta=-45 )
beam0_cx  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.0)

beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.0, theta=-45 )
beam90_cx = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.0 )

# Setup ring scan
theta, phi  = healpy.pix2ang( map_nside, source_pixels )
ra   = numpy.array( (phi, phi) )
dec  = numpy.array( (pi/2.0 - theta, pi/2.0-theta) )
pa   = numpy.zeros_like( ra )

print "S_in = ", S_in
# perform deprojection aliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )
print 'Deprojection at 0 degrees' , pisco_tod

# perform deprojection at 90 degrees from the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa + pi/2.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )
print 'Deprojection at 90 degrees' , pisco_tod

# perform deprojection aliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa + pi/4.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

print 'Deprojection at 45 degrees' , pisco_tod

# perform deprojection misaliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa - pi/4.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

print 'Deprojection at -45 degrees' , pisco_tod
