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
                                                                                                              
I_map   = map_data['Q']                                                                                       
Q_map   = map_data['Q']                                                                                       
U_map   = map_data['U']                                                                                       
V_map   = map_data['V']

# Setup a pure Q source
source_pixels = 400000 #healpy.query_disc( map_nside, (1,0,0), numpy.radians(1) )
S_in = (1,0,1,0)
I_map[ source_pixels ] = S_in[0]
Q_map[ source_pixels ] = S_in[1]
U_map[ source_pixels ] = S_in[2]

# Setup a simple Gaussian beam
beam_nside  = map_nside
beam_fwhm   =  4.0
beam_extn   = numpy.radians(16.0)
# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )
beam0_co[0] = 1.0
beam0_cx  = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )

beam90_co = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )
beam90_co[0] = 1.0
beam90_cx = make_gaussian_beam( beam_nside, beam_fwhm, beam_fwhm, beam_extn, amplitude=0.0 )

# Setup ring scan
scan_pixels = numpy.arange( healpy.nside2npix( map_nside ) )
theta, phi  = healpy.pix2ang( map_nside, source_pixels )
ra   = numpy.asarray( [theta,theta] )
dec  = numpy.asarray( [phi,phi] )
pa   = numpy.zeros_like( ra )

# perform deprojection aliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )

