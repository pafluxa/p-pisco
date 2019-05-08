#!/usr/bin/env python
# coding: utf-8
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import os
import glob
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
from numpy import pi
import healpy
import pylab

# Do this shit right with argparser
import argparse
parser = argparse.ArgumentParser(description='Simulate CLASS maps using provided pointing.')
parser.add_argument( '-maps' , action='store', type=str, dest='maps', 
                      help='Input map to deproject' )
parser.add_argument( '-tag' , action='store', type=str, dest='tag',
                      help='Name of the output. Output will be matrices_tag.npz' )
parser.add_argument( '-array_data', action='store', type=str, dest='array_data',
                      help='CSV file with the array specificiations.' )
parser.add_argument( '-beam_folder',  action='store', type=str, dest='beam_folder',
                      help='File with specifications of the beam parameters for each detector.')
args = parser.parse_args()

# Setup simulation tag
tag = args.tag


#----------------------------------------------------------------------------------------------------------#
# Setup experiment location
location = EarthLocation( lat= -22.959748*u.deg, lon=-67.787260*u.deg, height=5200*u.m )
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
# Load maps from input
maps = numpy.load( args.maps )
map_nside = maps['nside'][()]
I_map     = maps['I'] 
Q_map     = maps['Q'] 
U_map     = maps['U'] 
V_map     = maps['V']*0
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( args.array_data )
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

#----------------------------------------------------------------------------------------------------------#
# Assemble beams across the focal plane
print 'assembling focal plane beams'
uids       = focalPlane['uid']
beam_nside = 128
beams      = [None] * receiver.ndets
for feed in range( 1,37 ):
    
    if feed == 0:
        continue
    
    # Load the beam corresponding the feeds
    try:
        V_uid, H_uid = focalPlane['uid'][ focalPlane['feed'] == feed ]
    except:
        continue

    beam_V = numpy.load( os.path.join( args.beam_folder, 'detector%d.npz' % (57) ) )
    beam_H = numpy.load( os.path.join( args.beam_folder, 'detector%d.npz' % (57) ) )
   
    V_Eco       = beam_V['Ex_co']
    V_Ecx       = beam_V['Ex_cx']
    mdata_V     = beam_V['mdata']
    xs,xe,ys,ye = numpy.radians( mdata_V[1][0]['limits'] )
    nx,ny       = mdata_V[1][0]['nx'], mdata_V[1][0]['ny']
    grid_size   = ye - ys

    V_co = azel_grid_to_healpix_beam( V_Eco, nx, ny, grid_size, beam_nside )
    V_cx = azel_grid_to_healpix_beam( V_Ecx, nx, ny, grid_size, beam_nside )
    
    H_Eco       = beam_H['Ex_co']
    H_Ecx       = beam_H['Ex_cx']
    mdata_H     = beam_H['mdata']
    xs,xe,ys,ye = numpy.radians( mdata_H[1][0]['limits'] )
    nx,ny       = mdata_H[1][0]['nx'], mdata_H[1][0]['ny']
    grid_size   = ye - ys

    H_co = azel_grid_to_healpix_beam( H_Eco, nx, ny, grid_size, beam_nside )
    H_cx = azel_grid_to_healpix_beam( H_Ecx, nx, ny, grid_size, beam_nside )
    
    beams[ V_uid ] = { 'co': V_co , 'cx': V_cx, 'nside': beam_nside }
    beams[ H_uid ] = { 'co': H_co , 'cx': H_cx, 'nside': beam_nside }

#----------------------------------------------------------------------------------------------------------#

# De-projection procedure
# Setup map matrices
AtA = 0.0
AtD = 0.0

# Setup mock scan
ts       = 1451606400
day      = 86400
month    = day * 30
sps      = 1.0
T        = 600.0

nscans_per_day = 150


d = 1
months = numpy.arange( 12 )
for month_of_the_year in months:

    for day_of_the_week, boresight_rotation in enumerate( numpy.radians( [0] ) ):

        t0 = ts + month_of_the_year*month + day_of_the_week*day
        tic = time.time()
        for scan in range(0,nscans_per_day):

            nsamples = int( T * sps )
            
            ctime     = numpy.linspace( t0, t0 + T, nsamples )
            if scan%2 == 0:
                azimuth   = numpy.linspace( -180,  540, nsamples )
            else:
                azimuth   = numpy.linspace(  540, -180, nsamples )

            azimuth   = numpy.radians( azimuth )
            elevation = numpy.zeros( nsamples ) + numpy.pi/4.0
            rotation  = numpy.zeros( nsamples ) + boresight_rotation

            # Setup TOD
            tod = TOD()
            tod.initialize( ctime, azimuth, elevation, rotation )
            tod.detdata = numpy.empty( (receiver.ndets, tod.nsamples) )
            tod.data_mask = numpy.zeros_like( tod.detdata, dtype='int32' )
            # Compute ICRS coordiinates of the feedhorns
            feed_ra, feed_dec, feed_pa = compute_receiver_to_ICRS( tod, receiver, location )
            
            # Set data_mask to 1 for invalid detectors
            for bad_det in invalid_dets:
                tod.data_mask[ bad_det ] += 1
            
            t0 += T

            # Run deprojection by feedhorn.
            for feed in range( 1, 37 ):
                
                print 'deprojecting feed', feed

                # Load the beam corresponding the feeds
                try:
                    V_uid, H_uid = focalPlane['uid'][ focalPlane['feed'] == feed ]
                except:
                    continue

                # If one of the detector in the pair is invalid, set the pair stream to zero
                if V_uid in invalid_dets or H_uid in invalid_dets:
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

                # Run deprojection using the beam
                det_stream_1 = deproject_sky_for_feedhorn(
                    feed_ra[V_uid], feed_dec[V_uid], feed_pa[V_uid],
                    pair_pol_angles[0],
                    (I_map,Q_map,U_map,V_map),
                    beam_nside, beam1_co,beam1_cx, beam2_co, beam2_cx,
                    gpu_dev=0 )

                det_stream_2 = deproject_sky_for_feedhorn(
                    feed_ra[H_uid], feed_dec[H_uid], feed_pa[H_uid],
                    pair_pol_angles[1],
                    (I_map,Q_map,U_map,V_map),
                    beam_nside, beam1_co,beam1_cx, beam2_co, beam2_cx,
                    gpu_dev=0 )

                tod.detdata[ V_uid ] = det_stream_1
                tod.detdata[ H_uid ] = det_stream_2
                
            ata, atd = update_matrices(
                             feed_ra, feed_dec, feed_pa, receiver.pol_angles,
                             tod.detdata,
                             map_nside,
                             data_mask=tod.data_mask )

            print numpy.where( numpy.isnan( ata ) )
            print numpy.where( numpy.isnan( atd ) )

            AtA += ata
            AtD += atd

            # Save matrices
            numpy.savez( './runs/matrices_%s.npz' % (tag), AtA=AtA, AtD=AtD, nside=map_nside )


