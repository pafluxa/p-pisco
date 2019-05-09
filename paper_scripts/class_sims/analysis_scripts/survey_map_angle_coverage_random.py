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
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord,  AltAz
from astropy.utils import iers
from astropy.coordinates import get_sun

import pandas
import time
import numpy
import healpy
import pylab

from numpy import pi

# Setup experiment location
location = EarthLocation( lat= -22.959748*u.deg, lon=-67.787260*u.deg, height=5200*u.m )

#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( './data/array_data/qband_rcvr.csv' )
#----------------------------------------------------------------------------------------------------------#
receiver = Receiver()
receiver.initialize(  focalPlane['uid'],
                     -focalPlane['az_off'],
                     -focalPlane['el_off'],
                      numpy.radians( focalPlane['rot'] ) )
#----------------------------------------------------------------------------------------------------------#

# Get valid detectors                                                                                         
#----------------------------------------------------------------------------------------------------------#  
invalid_dets = numpy.argwhere( numpy.isnan( focalPlane['az_off'] ) )                                          
# Set all detectors to be off                                                                                 
receiver.toggle_detectors( invalid_dets )

# Setup mock scan

day    = 86400
month  = 86400*28
t00    = 1492732800
sps    = 1.0
T      = 600.0
nsamples  = int( T * sps ) # Sampling rate of 200 Hz
nscans    = int( 86400 / T ) + 1

map_nside = 256
angle_maps = numpy.zeros( (16, healpy.nside2npix(map_nside) ) )

for month_of_the_year in range(12):
    
    for day_of_the_week, boresight_rotation in enumerate( numpy.radians( [0,15,30,45,-45,-30,-15] ) ):
        
        print numpy.degrees( boresight_rotation )

        t0 = t00 + month_of_the_year*month + day_of_the_week*day
        
        print (t0 - t00)/86400
        
        for scan in range(0,nscans):
                
            ctime     = numpy.linspace( t0, t0 + T, nsamples )
            if scan%2 == 0:
                azimuth   = numpy.linspace( 0, 2*numpy.pi, nsamples )
            else:
                azimuth   = numpy.linspace(-2*numpy.pi, 0, nsamples )
            
            elevation = numpy.zeros( nsamples ) + numpy.pi/4.0
            rotation  = numpy.zeros( nsamples ) + boresight_rotation

            # Setup TOD
            tod = TOD()
            tod.initialize( ctime, azimuth, elevation, rotation )
            
            # Compute ICRS coordiinates of the feedhorns
            feed_ra, feed_dec, feed_pa = compute_receiver_to_ICRS( tod, receiver, location )
           
            for det,pa_det in enumerate( feed_pa ):
                
                if det in invalid_dets: 
                    continue
                # bin pa_det
                bins = numpy.linspace(-pi,pi,16)
                digitized = numpy.digitize( pa_det + receiver.pol_angles[det], bins )
                for b,angle_map in enumerate(angle_maps):
                    
                    bin_mask = [ digitized == b ]

                    pixels = healpy.ang2pix( map_nside, pi/2.0-feed_dec[det][bin_mask], feed_ra[det][bin_mask] )
                    
                    angle_map[ pixels ] += 1

            t0 += T

bins = numpy.linspace(-pi,pi,16)
for i in range( 16 ):
    healpy.mollview( angle_maps[i], sub=(4,4,i+1), title="pa_det ~ %+3.3f\n" % numpy.degrees(bins[i]) )

pylab.show()
