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
Q_map     = maps['Q']*0
U_map     = maps['U']*0
V_map     = maps['V']*0

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

# Load beam from detector 57, central-ish one
beam_file  = sys.argv[3]
beam_data  = numpy.load( beam_file )
beam_nside = beam_data['nside'][()]
beam_co    = beam_data['E_co']
beam_cx    = beam_data['E_cx']

healpy.gnomview( beam_co**2, rot=(0,90), xsize=400, reso=5, sub=(1,2,1), title='co polar beam' )
healpy.gnomview( beam_cx**2, rot=(0,90), xsize=400, reso=5, sub=(1,2,2), title='cross polar beam' )
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
for scan_angle in numpy.linspace(-180,180,7):
    
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
