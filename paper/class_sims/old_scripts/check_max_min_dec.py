#!/usr/bin/env python
# coding: utf-8
import sys

import pisco
from pisco.pointing import *
from pisco.pointing.core import *
from pisco.tod import *

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

import pandas
import time
import numpy
import pylab

from blist import blist


# Setup experiment location
location = EarthLocation( lat= -22.959748*u.deg, lon=-67.787260*u.deg, height=5200*u.m )

#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( './data/array_data/qband.csv' )
#----------------------------------------------------------------------------------------------------------#
receiver = Receiver()
receiver.initialize(  focalPlane['uid'],
                      focalPlane['az_off'],
                      focalPlane['el_off'],
                      numpy.radians( focalPlane['rot'] ) )
#----------------------------------------------------------------------------------------------------------#

# Get valid detectors                                                                                         
#----------------------------------------------------------------------------------------------------------#  
invalid_dets = numpy.argwhere( numpy.isnan( focalPlane['az_off'] ) )                                          
# Set all detectors to be off                                                                                 
receiver.toggle_detectors( invalid_dets )

# Setup mock scan
ndays          = 1 # One month of data
ts             = 1451606400
te             = ts+ndays*86400
sps            = 1
scan_length    = 600

# Define the time range
_ctime    = numpy.arange( ts, te, 1./sps )

# Chunk the ctime in `scan_length` seconds chunks
ctime_chunks  = numpy.array_split( _ctime, (te - ts)/scan_length )
nscans = len( ctime_chunks )

boresights = [45, 15, 30, 45, -45, -30, -15 ]

boresight_counter = 0
scan_counter = 0

min_feed = -1
max_feed = -1
min_dec = 1e10
max_dec =-1e10

global_time = 0
for ctime in ctime_chunks:
    
    print scan_counter, 'out of', nscans

    global_time += scan_length
    nsamples     = scan_length * sps
    
    if scan_counter % 2 == 0:
        azimuth = numpy.radians( numpy.linspace( -180, 540, (nsamples) ) )
    else:
        azimuth = numpy.radians( numpy.linspace(  540,-180, (nsamples) ) ) 

    elevation = numpy.zeros( nsamples ) + numpy.radians( 45.0 )
    
    boresight = boresight_counter % 7
    rotation  = numpy.zeros( nsamples ) + boresights[ boresight ]
    
    if global_time > 86400:
        global_time = 0
        boresight_counter += 1
    
    tod = TOD()
    tod.initialize( ctime, azimuth, elevation, rotation )
    ra, dec, pa = compute_receiver_to_ICRS( tod, receiver, location )
   
    for uid in receiver.uid:
        
        if uid in invalid_dets:
            continue
        
        # Find minima and maxima of declination
        det_min_dec = numpy.min( dec[uid] )
        det_max_dec = numpy.max( dec[uid] )
        
        if det_min_dec < min_dec:
            min_dec = det_min_dec
            # Find feed corresponding to the uid
            min_feed = focalPlane[ 'feed' ][ uid ]

        if det_max_dec > max_dec:
            max_dec = det_max_dec
            # Find feed corresponding to the uid
            max_feed = focalPlane[ 'feed' ][ uid ]
    
    scan_counter += 1

print numpy.degrees(max_dec), max_feed
print numpy.degrees(min_dec), min_feed
