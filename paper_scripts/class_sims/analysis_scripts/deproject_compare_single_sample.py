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
#map_data  = numpy.load( sys.argv[1] )
#map_nside = map_data['nside'][()]

numpy.random.seed(1)

map_nside = 2048
I_map   = numpy.random.random( healpy.nside2npix( map_nside ) )
Q_map   =-I_map
U_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros( healpy.nside2npix( map_nside ) )

# Setup a simple Gaussian beam of 10 arcmin
beam_nside  = 1501
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
numpy.random.seed(23249234)
pix  = numpy.random.randint( 0, healpy.nside2npix( map_nside ) )
ra   = phi[pix:pix+1]
dec  = numpy.pi/2.0 - theta[pix:pix+1]
pa   = numpy.zeros_like( ra ) 

print ra, dec

# perform deprojection aliged with the source
det_angle  =  0.0
scan_angle =  pi/2.0
tod1 = deproject_sky_for_feedhorn(
    ra, dec, pa + scan_angle,
    det_angle,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

det_angle  =  pi/2.0
scan_angle =  0.0
tod2 = deproject_sky_for_feedhorn(
    ra, dec, pa + scan_angle,
    det_angle,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

print numpy.degrees( scan_angle ),tod1, tod2

'''
pylab.subplot( 211 )
pylab.plot( tod1, alpha=0.5 )
pylab.plot( tod2, alpha=0.5 )

pylab.subplot( 212 )
pylab.plot( tod1 - tod2 )


pylab.show()
'''
