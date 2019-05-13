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

# Setup mock scan
ndays          = 1 # One day of data
ts             = 1451606400
te             = ts+ndays*86400
sps            = 1
scan_length    = 600

# Define the time range
global_ctime   = numpy.arange( ts, te, 1./sps )

# Chunk the ctime in `scan_length` seconds chunks
ctime_chunks  = numpy.array_split( global_ctime, (te - ts)/scan_length )
nscans = len( ctime_chunks )

boresights = [0, 15, 30, 45, -15, -30, -45]

boresight_counter = 0
scan_counter = 0

azm_timespan = numpy.empty(1) 
alt_timespan = numpy.empty(1) 
rot_timespan = numpy.empty(1) 

global_time = 0
for ctime in ctime_chunks:
    
    print scan_counter, 'out of', nscans

    global_time += ctime[-1] - ctime[0] 
    nsamples     = ctime.size
    
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
    
    azm_timespan = numpy.concatenate( (azm_timespan, azimuth  ) )
    alt_timespan = numpy.concatenate( (alt_timespan, elevation) )
    rot_timespan = numpy.concatenate( (rot_timespan, rotation ) )
     
    scan_counter += 1                                                                                         
                                                                                                              
numpy.savez( './data/pointing/class_qband_ndays_%d_%d_Hz_pointing_horizontal.npz' % (
        (int)(ndays), (int)(sps) ),        
        ctime = global_ctime, 
        azm   = azm_timespan[1::], 
        alt   = alt_timespan[1::], 
        rot   = rot_timespan[1::] ) 
