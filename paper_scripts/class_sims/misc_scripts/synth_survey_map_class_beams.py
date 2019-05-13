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
                     -focalPlane['el_off'],
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

# Read beam parameter file
uids, theta, sx, sy, e = numpy.loadtxt( './data/array_data/class_beam_params.txt', unpack=True )
for i,uid in enumerate(uids):
    
    uid = (int)( uid )
    
    nx = 512       # 501 pixels per side
    grid_side = 10 # grid size of 10 degrees
    
    beam_co   = make_gaussian_elliptical_beam( sx[i]*2.355, sy[i]*2.355, grid_side, nx, theta=theta[i], amplitude=1.0 )
    beam_cx   = numpy.zeros_like( beam_co )
    
    '''
    pylab.subplot( 121 )
    pylab.imshow( numpy.abs( beam_co.reshape( (nx,nx) ) )**2 )
    pylab.subplot( 122 )
    pylab.imshow( numpy.abs( beam_cx.reshape( (nx,nx) ) )**2 )
    pylab.show()
    '''
    beams[ uid ] = { 'co': beam_co , 'cx': beam_cx, 'nx': nx, 'grid_side':grid_side }


# Setup map matrices
AtA = 0.0
AtD = 0.0

# Setup mock scan
day    = 86400
month  = 86400*28
t00    = 1492732800
sps    = 1.0
T      = 600.0
nsamples  = int( T * sps ) # Sampling rate of 200 Hz
nscans    = int( 86400 / T ) + 1

months = numpy.arange( 12 )
for month_of_the_year in numpy.random.choice( months, size=2, replace=False ):

    for day_of_the_week, boresight_rotation in enumerate( numpy.radians( [0,15,30,45,-45,-30,-15] ) ):

        print numpy.degrees( boresight_rotation )

        t0 = t00 + month_of_the_year*month + day_of_the_week*day

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

            # Setup TOD
            tod = TOD()
            tod.initialize( ctime, azimuth, elevation, rotation )
            tod.detdata = numpy.zeros( (receiver.ndets, tod.nsamples), dtype='float64' )            
            tod.datamask = numpy.zeros( (receiver.ndets, tod.nsamples), dtype='int32' )
            # Flag bad detectors
            for inv_det in invalid_dets:        
                tod.datamask[ inv_det ] += 1

            # Compute ICRS coordiinates of the feedhorns
            feed_ra, feed_dec, feed_pa = compute_receiver_to_ICRS( tod, receiver, location )

            # The above routine gives psi as parallactic_angle + detector_angle. We only need
            # Run deprojection in pairs.
            for V_uid, H_uid in zip(det_uid_H, det_uid_V):

                # If one of the detector in the pair is invalid, set the pair stream to zero
                if V_uid in invalid_dets or H_uid in invalid_dets:
                    print 'marking', V_uid, 'and', H_uid, 'as invalid'
                    continue
                
                print 'deprojecting detectors', V_uid, 'and', H_uid

                # Get polarization angle of the pair
                pair_pol_angles = numpy.asarray( (receiver.pol_angles[V_uid], receiver.pol_angles[H_uid] ) )

                # Get beam
                grid_nx   = beams[ V_uid ]['nx']
                grid_side = beams[ V_uid ]['grid_side']
                beam1_co,beam1_cx = numpy.copy(beams[ V_uid ]['co']), numpy.copy(beams[V_uid]['cx'])
                beam2_co,beam2_cx = numpy.copy(beams[ H_uid ]['co']), numpy.copy(beams[H_uid]['cx'])

                # Run deprojection using the beam
                det_stream_1 = deproject_sky_for_feedhorn(
                    feed_ra[V_uid], feed_dec[V_uid], feed_pa[V_uid],
                    pair_pol_angles[0],
                    (I_map,Q_map,U_map,V_map),
                    grid_side, grid_nx,beam1_co,beam1_cx, beam2_co, beam2_cx,
                    gpu_dev=0 )
                
                det_stream_2 = deproject_sky_for_feedhorn(
                    feed_ra[H_uid], feed_dec[H_uid], feed_pa[H_uid],
                    pair_pol_angles[1],
                    (I_map,Q_map,U_map,V_map),
                    grid_side, grid_nx, beam1_co,beam1_cx, beam2_co, beam2_cx,
                    gpu_dev=0 )

                tod.detdata[ V_uid ] = det_stream_1
                tod.detdata[ H_uid ] = det_stream_2
                
            ata, atd = update_matrices(
                             feed_ra, feed_dec, feed_pa, receiver.pol_angles,
                             tod.detdata,
                             map_nside,
                             data_mask=tod.datamask )
            AtA += ata
            AtD += atd
            
            # Save matrices
            numpy.savez( './runs/matrices_classbeams_input_lcdmr0d10_id_0000.npz', AtA=AtA, AtD=AtD, nside=map_nside )
            
            t0 = t0 + T

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
