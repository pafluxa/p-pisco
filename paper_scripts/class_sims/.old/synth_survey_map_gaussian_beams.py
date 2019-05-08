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

import os
import pandas
import time
import numpy
import healpy
import pylab

# Get filename
filename = os.path.basename( sys.argv[1] )
print filename

# Setup experiment location
location = EarthLocation( lat= -22.959748*u.deg, lon=-67.787260*u.deg, height=5200*u.m )

# Load maps from input
maps = numpy.load( sys.argv[1] )
map_nside = maps['nside'][()]
I_map     = maps['I']
Q_map     = maps['Q']
U_map     = maps['U']
V_map     = maps['V']*0

#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( './data/array_data/qband.csv' )
#----------------------------------------------------------------------------------------------------------#
receiver = Receiver()
receiver.initialize(  focalPlane['uid'],
                     -focalPlane['az_off'],
                      focalPlane['el_off'],
                      numpy.radians( focalPlane['rot'] ) )
#----------------------------------------------------------------------------------------------------------#

# Get valid detectors
#----------------------------------------------------------------------------------------------------------#
invalid_dets = numpy.argwhere( numpy.isnan( focalPlane['az_off'] ) )
# Set all detectors to be off
receiver.toggle_detectors( invalid_dets )

# Create beams in pairs
beams = [ None ] * 88
dtype = { 'names': ('is_pair', 'det_uid_V', 'det_uid_H'),
          'formats': ( '|S8', numpy.int, numpy.int ) }
is_pair, det_uid_V, det_uid_H = numpy.loadtxt( './data/array_data/pairs.txt', unpack=True , dtype=dtype )

beam_nside =  512
beam_fwhm  =  1.5
beam_extn  = 10.0
# Read beams in pairs
for is_paired, V_uid, H_uid in zip(is_pair, det_uid_H, det_uid_V):

    # Create perfect Gaussian beams without cross polarization
    gaussian_beam  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.0 )

    beams[ V_uid ] = { 'co': gaussian_beam , 'cx': gaussian_beam*0.0, 'nside': beam_nside }
    beams[ H_uid ] = { 'co': gaussian_beam , 'cx': gaussian_beam*0.0, 'nside': beam_nside }


# Setup map matrices
AtA = 0.0
AtD = 0.0

# Setup mock scan
month  = 1517443200 - 1514764800
t00    = 1451606400
sps = 1.0
T   = 600.0
nsamples  = int( T * sps ) # Sampling rate of 200 Hz
nscans    = 200

for offset in range(12):

    t0 = t00 + offset*month
    print t0

    for boresight_rotation in numpy.radians( [0,15,30,45,-45,-30,-15] ):

        t0 += 86400.0
        for scan in range(0,nscans):

            print scan, '/', nscans

            # Setup zero boresight, 45 degree elevation azimuth scan starting from now
            ctime     = numpy.linspace( t0, t0 + T, nsamples )
            if scan%2 == 0:
                azimuth   = numpy.linspace( 0, 2*numpy.pi, nsamples )
            else:
                azimuth   = numpy.linspace(-2*numpy.pi, 0, nsamples )

            elevation = numpy.zeros( nsamples ) + numpy.pi/4.0
            rotation  = numpy.zeros( nsamples ) + boresight_rotation
            t0 = t0 + T

            # Setup TOD
            tod = TOD()
            tod.nsamples = nsamples
            tod.ndets = receiver.ndets
            tod.ctime = ctime.astype( 'float64' )
            tod.az = azimuth.astype( 'float64' )
            tod.alt = elevation.astype( 'float64' )
            tod.rot = rotation.astype( 'float64' )
            tod.pointing_mask = numpy.zeros_like( ctime, dtype='int32' )
            tod.detdata = numpy.zeros( (receiver.ndets, tod.nsamples), dtype='float32' )

            data_mask = numpy.zeros( (receiver.ndets, tod.nsamples ), dtype='int32' )
            # Set data_mask to 1 for invalid detectors
            for bad_det in invalid_dets:
                data_mask[ bad_det ] += 1

            # Compute ICRS coordiinates of the feedhorns
            feed_ra, feed_dec, feed_pa = compute_receiver_to_ICRS( tod, receiver, location )

            # The above routine gives psi as parallactic_angle + detector_angle. We only need
            # Run deprojection in pairs.
            for V_uid, H_uid in zip(det_uid_H, det_uid_V):

                # If one of the detector in the pair is invalid, set the pair stream to zero
                if V_uid in invalid_dets or H_uid in invalid_dets:
                    data_mask[ V_uid ] += 1
                    data_mask[ H_uid ] += 1
                    continue

                # Get polarization angle of the pair
                pair_pol_angles = numpy.asarray( (receiver.pol_angles[V_uid], receiver.pol_angles[H_uid] ) )

                # Get beam
                beam_nside        = beams[ V_uid ]['nside']
                beam1_co,beam1_cx = numpy.copy(beams[ V_uid ]['co']), numpy.copy(beams[V_uid]['cx'])
                beam2_co,beam2_cx = numpy.copy(beams[ H_uid ]['co']), numpy.copy(beams[H_uid]['cx'])

                # Normalize
                beam1_peak = ( beam1_co + beam1_cx ).max()
                beam2_peak = ( beam2_co + beam2_cx ).max()

                beam1_co /= beam1_peak
                beam1_cx /= beam1_peak
                beam2_co /= beam2_peak
                beam2_cx /= beam2_peak
                
                tic = time.time()
                # Run deprojection using the beam
                det_stream_1 = deproject_sky_for_feedhorn(
                    feed_ra[V_uid], feed_dec[V_uid], feed_pa[V_uid],
                    pair_pol_angles[0],
                    (I_map,Q_map,U_map,V_map),
                    beam_extn, beam_nside,beam1_co,beam1_cx, beam2_co, beam2_cx,
                    gpu_dev=1 )

                det_stream_2 = deproject_sky_for_feedhorn(
                    feed_ra[H_uid], feed_dec[H_uid], feed_pa[H_uid],
                    pair_pol_angles[1],
                    (I_map,Q_map,U_map,V_map),
                    beam_extn, beam_nside, beam1_co,beam1_cx, beam2_co, beam2_cx,
                    gpu_dev=1 )
                tod.detdata[ V_uid ] = det_stream_1
                tod.detdata[ H_uid ] = det_stream_2
                toc = time.time()

                print 'Detector deprojection took:', toc-tic, 'sec'
            ata, atd = update_matrices(
                             feed_ra, feed_dec, feed_pa, receiver.pol_angles,
                             tod.detdata,
                             map_nside,
                             data_mask=data_mask )
            AtA += ata
            AtD += atd
            # Save matrices
            numpy.savez( './runs/matrices_gaussian_beams_gridnx_%d_%s' % (beam_nside,filename), AtA=AtA, AtD=AtD, nside=map_nside )

I,Q,U,W = matrices_to_maps( map_nside, AtA, AtD )

I[ W==0 ] = healpy.UNSEEN
Q[ W==0 ] = healpy.UNSEEN
U[ W==0 ] = healpy.UNSEEN
W[ W==0 ] = healpy.UNSEEN

healpy.mollview( I , sub=(2,2,1) )
healpy.mollview( Q , sub=(2,2,2) )
healpy.mollview( U , sub=(2,2,3) )
healpy.mollview( W , sub=(2,2,4) )

#pylab.show()
pylab.savefig( 'maps.pdf' )
