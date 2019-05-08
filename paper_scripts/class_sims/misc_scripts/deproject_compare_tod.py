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
# Load LCDM model with r = 0
map_data  = numpy.load( sys.argv[1] )
map_nside = map_data['nside'][()]

I_map   = map_data['U']
Q_map   = map_data['Q']*0 
U_map   = map_data['U']
V_map   = map_data['V']*0

map_nside = 512
I_map = healpy.ud_grade( I_map, 64 )
Q_map = healpy.ud_grade( Q_map, 64 )
U_map = healpy.ud_grade( U_map, 64 )
V_map = healpy.ud_grade( V_map, 64 )

# Setup a simple Gaussian beam of 10 arcmin
beam_nside  = 61
beam_fwhm   = 2.0
beam_extn   = numpy.radians(10.0) # Extension of 5 degrees

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
pa   = numpy.zeros_like( ra )

det_angle  = pi/4.0
scan_angle = 0.0
tod1 = deproject_sky_for_feedhorn(
    ra, dec, pa + scan_angle,
    det_angle,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

'''
# perform deprojection aliged with the source
det_angle  = pi/4.0
scan_angle = 0.0
tod2 = deproject_sky_for_feedhorn(
    ra, dec, pa + scan_angle,
    det_angle,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

pylab.subplot( 211 )
pylab.plot( tod1, alpha=0.5 )
pylab.plot( tod2, alpha=0.5 )
pylab.subplot( 212 )

pylab.plot( tod1 - tod2 )

pylab.show()
'''
