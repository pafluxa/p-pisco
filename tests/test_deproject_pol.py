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


map_nside = 256
I_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
Q_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
U_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros( healpy.nside2npix( map_nside ) )

# Setup a pure Q source
v = numpy.asarray( (1,0,0 ))
source_pixels = healpy.query_disc( map_nside, v, numpy.radians(0.5) )
I_map[ source_pixels ] = 1.0
U_map[ source_pixels ] = 1.0

# Setup a simple Gaussian beam
beam_nside  = map_nside
beam_fwhm   =  1.0
beam_extn   =  6.0
# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=1.0 )
beam0_cx  = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )

beam90_co = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=1.0 )
beam90_cx = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )

# Setup ring scan
scan_pixels = numpy.arange( healpy.nside2npix( map_nside ) )
theta, phi  = healpy.pix2ang( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta
pa   = numpy.zeros_like( ra )

# Perform deprojection aliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

# Plot the TOD
pylab.figure()
pylab.subplot( 211 )
pylab.title( "Pure Q, S_in = (1,0,1,0) scan" )
pylab.plot( pisco_tod, label='scan at 0 degrees' )
#pylab.ylim( 0.0, 2 )
pylab.legend()

# Perform deprojection 90 degrees aligned. Should get zero in the TOD
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa + pi/2.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

# Plot the TOD
pylab.subplot( 212 )
pylab.plot( pisco_tod, label='scan at 90 degrees' )
#pylab.ylim( 0.0, 2 )
pylab.legend()

# Perform deprojection aliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa - pi/4.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

# Plot the TOD
pylab.figure()
pylab.subplot( 211 )
pylab.title( "Pure Q, S_in = (1,0,1,0) scan" )
pylab.plot( pisco_tod, label='scan at -45 degrees' )
#pylab.ylim( 0.0, 2 )
pylab.legend()

# Perform deprojection 90 degrees aligned. Should get zero in the TOD
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa + pi/4.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

# Plot the TOD
pylab.subplot( 212 )
pylab.plot( pisco_tod, label='scan at 45 degrees' )
#pylab.ylim( 0.0, 2 )
pylab.legend()
pylab.show()
