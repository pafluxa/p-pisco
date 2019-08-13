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
ndays          = 1.0/24.0 # One hour of data
ts             = 1451606400
te             = ts+ndays*86400
sps            = 1
nscans         = 100

# Define the time range
global_ctime   = numpy.arange( ts, te, 1./sps )

# Chunk the ctime in `scan_length` seconds chunks
ctime_chunks  = numpy.array_split( global_ctime, nscans )

elevations = numpy.linspace( 35, 55, nscans )
scan_counter = 0

azm_timespan = numpy.empty(1) 
alt_timespan = numpy.empty(1) 
rot_timespan = numpy.empty(1) 

for ctime in ctime_chunks:
    
    print scan_counter, 'out of', nscans

    nsamples     = ctime.size
    
    if scan_counter % 2 == 0:
        azimuth = numpy.radians( numpy.linspace( -20, 20, (nsamples) ) )
    else:
        azimuth = numpy.radians( numpy.linspace(  20,-20, (nsamples) ) ) 

    elevation = numpy.zeros( nsamples ) + numpy.radians( elevations[ scan_counter ] )
    
    rotation  = numpy.zeros( nsamples ) 
    
    azm_timespan = numpy.concatenate( (azm_timespan, azimuth  ) )
    alt_timespan = numpy.concatenate( (alt_timespan, elevation) )
    rot_timespan = numpy.concatenate( (rot_timespan, rotation ) )
     
    scan_counter += 1                                                                                         

pylab.plot( numpy.degrees( azm_timespan[1::] ), numpy.degrees( alt_timespan[1::] ) )
pylab.show()

numpy.savez( './data/pointing/class_qband_ndays_%d_%d_Hz_pointing_sun_scan.npz' % (
        (int)(ndays), (int)(sps) ),        
        ctime = global_ctime, 
        azm   = azm_timespan[1::], 
        alt   = alt_timespan[1::], 
        rot   = rot_timespan[1::] ) 
