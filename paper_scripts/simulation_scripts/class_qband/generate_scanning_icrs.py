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
#----------------------------------------------------------------------------------------------------------#  
# Read beam parameter file                                                                                    
print 'reading beam parameters'                                                                               
beam_data = pandas.read_csv( './data/array_data/qband_array_data_beam_params.csv' )                           
feeds     = numpy.array( beam_data[ 'Feed']  )                                                                              
azOff     = numpy.array( beam_data[ 'AzOff'] )                                                                              
elOff     = numpy.array( beam_data[ 'ElOff'] )                                                                             
fwhm_x    = numpy.array( beam_data[ 'FWHM_x'] )                                                                            
fwhm_y    = numpy.array( beam_data[ 'FWHM_y'] )                                                                             
rotation  = numpy.array( beam_data[ 'theta'] )                                                                               
#----------------------------------------------------------------------------------------------------------#  
                                                                                                              
#----------------------------------------------------------------------------------------------------------#  
# Build focal plane                                                                                           
#----------------------------------------------------------------------------------------------------------#  
uids   = numpy.arange( feeds.size * 2 )                                                                        
feeds  = numpy.repeat( feeds, 2 )                                                                             
azOff  = numpy.repeat( azOff, 2 )                                                                             
elOff  = numpy.repeat( elOff, 2 )                                                                             
polang = numpy.ones_like( uids ) * 45                                                                         
polang[ uids % 2 == 1 ] = -45                                                                                 

azOff  = numpy.deg2rad(  azOff )                                                                              
elOff  = numpy.deg2rad(  elOff )                                                                              
polang = numpy.deg2rad( polang )                                                                              

receiver = Receiver()                                                                                         
receiver.initialize( uids, azOff, elOff, polang )                                                              
#----------------------------------------------------------------------------------------------------------#  

# Setup mock scan
ndays          = 7
ts             = 1451606400
te             = ts + ndays*86400 # This is one week
sps            = 20
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
        ra=ra, dec=dec, pa=pa )
