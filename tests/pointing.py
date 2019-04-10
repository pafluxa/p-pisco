#!/usr/bin/env python 
# coding: utf-8
import astropy
import time
import numpy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.utils import iers

import pisco
from pisco.pointing.core import compute_receiver_to_ICRS
from pisco.pointing import Receiver
from pisco.tod import TOD

import pylab
'''
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
'''

# Setup experiment location using AstroPy
experiment_location = EarthLocation(
    lat=-22.959748*u.deg, 
    lon=-67.787260*u.deg, 
    height=5180*u.m )

# Setup a dummy receiver with 1 detector at the center of the focal plane
recv = Receiver()
ndets = 1
uids       = [0]
dx         = [0]
dy         = [0]
pol_angles = [0]
recv.initialize( uids, dx, dy, pol_angles )

# Setup a dummy TOD with special coordinates
# Sampling rate : 200 Hz
# TOD extension: 600 seconds
sps = 600
ctime = numpy.arange( 0,1.0, 1./sps ) + 1458561600.0
az    = numpy.zeros_like( ctime )
alt   = numpy.zeros_like( ctime )
rot   = numpy.zeros_like( ctime )
tod   = TOD()
tod.initialize( ctime, az, alt, rot )

# Compute coordinates using PISCO: should return ra/dec ~ 0.0
ra_pisco, dec_pisco, pa_pisco = compute_receiver_to_ICRS( tod, recv, experiment_location )

# Compute coordinates using Astro-py
astropy_time = Time( ctime, format='unix' )
AltAz_coords = SkyCoord(
    az=az*u.deg, 
    alt=alt*u.deg, 
    obstime=astropy_time, 
    location=experiment_location,
    frame='altaz')

ra_astropy = numpy.radians( AltAz_coords.icrs.ra.value )
dec_astropy = numpy.radians( AltAz_coords.icrs.dec.value )

x_pisco = numpy.array(
    [ numpy.cos(ra_pisco.ravel()) * numpy.cos(dec_pisco.ravel()),
      numpy.sin(ra_pisco.ravel()) * numpy.cos(dec_pisco.ravel()),
      numpy.sin(dec_pisco.ravel()) ] )

x_astropy = numpy.array(
    [ numpy.cos(ra_astropy.ravel()) * numpy.cos(dec_astropy.ravel()),
      numpy.sin(ra_astropy.ravel()) * numpy.cos(dec_astropy.ravel()),
      numpy.sin(dec_astropy.ravel()) ] )

distance = numpy.linalg.norm( x_pisco - x_astropy, axis=0 )

pylab.title( 'Pointing difference between PISCO and Astropy' )
pylab.subplot( 111 )
pylab.plot( ctime - ctime[0], numpy.degrees(distance)*3600 )
pylab.ylabel( 'x_pisco - x_astropy (arcseconds)' )
pylab.xlabel( 'time (seconds)' )
pylab.ylim( -0.001, 0.001 )
pylab.show()
