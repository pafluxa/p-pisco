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


map_nside = 512
I_map   = numpy.zeros( healpy.nside2npix( map_nside ) ) 
Q_map   = numpy.zeros( healpy.nside2npix( map_nside ) ) 
U_map   = numpy.zeros( healpy.nside2npix( map_nside ) )
V_map   = numpy.zeros( healpy.nside2npix( map_nside ) )

# Setup a pure Q source
tm,pm = healpy.pix2ang( map_nside, numpy.arange( healpy.nside2npix( map_nside ) ) )
source_pixels = numpy.where( (pi/2.0-tm)**2 + pm < 0.01 )
S_in = (1,0,1,0)
I_map[source_pixels]  += S_in[0]
Q_map[source_pixels]  += S_in[1]
U_map[source_pixels]  += S_in[2]

healpy.mollview( U_map )
pylab.show()

# Setup a simple Gaussian beam
beam_nside  = 121
beam_fwhm   = 0.01
beam_extn   = numpy.radians(10.0)
# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.0 )
beam0_cx  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.0 )

beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.0 )
beam90_cx = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.0 )

# Setup ring scan
ra   = numpy.array( (0.0,0.0) )
dec  = numpy.array( (0.0,0.0) )
pa   = numpy.zeros_like( ra )

# perform deprojection aliged with the source
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa + pi/2.0,
    0.0,
    (I_map,Q_map,U_map,V_map),
    beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )
