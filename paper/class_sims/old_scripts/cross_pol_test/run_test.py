#!/usr/bin/env python
# coding: utf-8
import sys

import pisco
from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.mapping.core import *
from pisco.pointing import *
from pisco.pointing.core import *
from pisco.tod import *

import astropy.units as u
from astropy.coordinates import EarthLocation

import pandas
import time
import numpy
import healpy
import pylab

tag = sys.argv[1]

# Load maps from input
maps = numpy.load( sys.argv[2] )
map_nside = maps['nside'][()]
I_map     = maps['I']
Q_map     = maps['Q']
U_map     = maps['U']
V_map     = maps['V']

#----------------------------------------------------------------------------------------------------------#
# Setup focal plane with a single detector aligned with the boresight
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
receiver = Receiver()
receiver.initialize(  [0],  # uid of detector
                      [0],  # dx of detector
                      [0],  # dy of detector,
                      [0],) # polarization angle of detector
#----------------------------------------------------------------------------------------------------------#

# create Gaussian beam of 1.0 degree FWHM
beam_nside = 512
fwhm       = 1.0
beam_co    = make_gaussian_elliptical_beam( beam_nside, fwhm, fwhm  , theta=0.0 )

# Setup cross-polar beam
beam_cx    = make_gaussian_elliptical_beam( beam_nside, fwhm, fwhm*2, theta= 45 )
beam_cx   -= make_gaussian_elliptical_beam( beam_nside, fwhm, fwhm*2, theta=-45 )
beam_cx   /= beam_cx.max()
beam_cx   *= 0.01
fwhm       = 0.5

healpy.gnomview( beam_co, rot=(0,90), sub=(1,2,1), xsize=300, title='co-polar beam pattern' )
healpy.gnomview( beam_cx, rot=(0,90), sub=(1,2,2), xsize=300, title='co-polar beam pattern' )
pylab.show()

# Setup co and cross polar beams. Null test only use co-polar
beam_co_x   = numpy.copy( beam_co )
beam_cx_x   = numpy.copy( beam_cx )

beam_co_y   = numpy.copy( beam_co )
beam_cx_y   = numpy.copy( beam_cx )

# Setup map projection matrices
AtA = 0.0
AtD = 0.0

# Setup all sky scanning strategy
sky_pixels = numpy.arange( healpy.nside2npix( map_nside ) )
tht, ra    = healpy.pix2ang( map_nside, sky_pixels )
# HEALPY returns co-latitude, so convert to declination
dec        = numpy.pi/2.0 - tht
# Set parallactic angle to be zero + scan_angle
pa         = numpy.zeros_like( ra )

# Run scanning. Make three passes at different parallactic angles of -45, 0 and 45 degrees.
for scan_angle in numpy.linspace(-45,45,3):
    
    print scan_angle

    data = deproject_sky_for_feedhorn( ra, dec, pa + numpy.radians( scan_angle ),
                                      receiver.pol_angles[0],
                                      (I_map,Q_map,U_map,V_map),
                                      beam_nside, beam_co_x, beam_cx_x, beam_co_y, beam_cx_y,
                                      gpu_dev=0, maxmem=1024 )

    # Reshape stuff to fool PISCO to think it is processing a regular TOD
    fra   =  ra.reshape( (1,-1) )
    fdec  = dec.reshape( (1,-1) )
    fpa   =  pa.reshape( (1,-1) )
    
    # Same for the detector data
    fdata = data.reshape( (1,-1) )

    ata, atd = update_matrices(
                     fra, fdec, fpa + numpy.radians( scan_angle ),
                     receiver.pol_angles,
                     fdata,
                     map_nside )
    AtA += ata
    AtD += atd
    
    # Save matrices
    numpy.savez( './runs/matrices_%s.npz' % (tag), AtA=AtA, AtD=AtD, nside=map_nside )

I,Q,U,W = matrices_to_maps( map_nside, AtA, AtD )

I[ W==0 ] = healpy.UNSEEN
Q[ W==0 ] = healpy.UNSEEN
U[ W==0 ] = healpy.UNSEEN
W[ W==0 ] = healpy.UNSEEN

healpy.mollview( I , sub=(1,3,1) , title='I')
healpy.mollview( Q , sub=(1,3,2) , title='Q')
healpy.mollview( U , sub=(1,3,3) , title='U')

pylab.show()
#pylab.savefig( 'maps.pdf' )
