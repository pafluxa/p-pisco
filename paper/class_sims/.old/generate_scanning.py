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

# Setup experiment location
location = EarthLocation( lat= -22.959748*u.deg, lon=-67.787260*u.deg, height=5200*u.m )

#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( './data/array_data/qband.csv' )
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
ndays          = 1
ts             = 1451606400
te             = ts + 7*86400 # This is one week
#te             = 1454284800 # This is one month of data
sps            = 1
T              = 600

# Define the TOD
ctime     = numpy.arange( ts, te, 1./sps )
elevation = numpy.zeros_like( ctime ) + numpy.pi/4.0
azimuth   = numpy.zeros_like( ctime )
rotation  = numpy.zeros_like( ctime )

# Compute the number of scans as the duration of a scan 
nscans    = int( numpy.ceil( (te - ts)/T ) )
ndays     = int( numpy.ceil( (te - ts)/86400 ) )

print 'synth', nscans, 'per', ndays

for scan in range( nscans ):
    
    if scan%2 == 0:
        azimuth[ scan*T*sps: (scan+1)*T*sps ] = numpy.radians( numpy.linspace( -180, 540, (T*sps) ) )
    else:
        azimuth[ scan*T*sps: (scan+1)*T*sps ] = numpy.radians( numpy.linspace(  540,-180, (T*sps) ) )

boresights = [-45,-30,-15-0,15,30,45]
for day in range( ndays ):

    bday = day % 6

    rotation[ day * 86400 * sps: (day+1) * 86400 * sps ] += numpy.radians( boresights[ bday ] )

tod = TOD()
tod.initialize( ctime, azimuth, elevation, rotation )

ra, dec, pa = compute_receiver_to_ICRS( tod, receiver, location )

numpy.savez( './data/pointing/class_qband_ndays_%d_nscans_%d_%d_Hz_pointing.npz' % ( 
        (int)(ndays), 
        (int)(nscans),
        (int)(sps) ), 
        ra=ra, dec=dec, pa=pa, invalid_dets=invalid_dets )
