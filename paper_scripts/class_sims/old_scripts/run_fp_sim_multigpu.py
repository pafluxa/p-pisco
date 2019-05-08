#!/usr/bin/env python
# coding: utf-8
import array_split
from array_split import array_split, shape_split

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import sys
import argparse

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

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    
    # Do this shit right with argparser
    parser = argparse.ArgumentParser(description='Simulate CLASS maps using provided pointing.')
    parser.add_argument( '-maps' , action='store', type=str, dest='maps', 
                          help='Input map to deproject' )
    parser.add_argument( '-tag' , action='store', type=str, dest='tag',
                          help='Name of the output. Output will be matrices_tag.npz' )
    parser.add_argument( '-pointing', action='store', type=str, dest='pointing',
                          help='NPZ file with the pointing of the season. See generete_pointing.py for more details.' )
    parser.add_argument( '-array_data', action='store', type=str, dest='array_data',
                          help='CSV file with the array specificiations.' )
    parser.add_argument( '-beam_par_file',  action='store', type=str, dest='beam_par_file',
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
    #----------------------------------------------------------------------------------------------------------#
    
    pointing_file = args.pointing
    pointing = numpy.load( pointing_file , mmap_mode='r')
    pointing_shape = pointing['shape']
    inv_dets = pointing['invalid_dets']

    #----------------------------------------------------------------------------------------------------------#
    # Read beam parameter file
    print 'reading beam parameters'
    beam_data = pandas.read_csv( args.beam_par_file )
    uids      = beam_data[ 'uid'] 
    feeds     = beam_data[ 'feed'] 
    fwhm_x    = beam_data[ 'fwhm_x'] 
    fwhm_y    = beam_data[ 'fwhm_y'] 
    rotation  = beam_data[ 'rot'] 
    #----------------------------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------------------------#
    # Assemble beams across the focal plane
    print 'assembling focal plane beams'
    beam_nside = 128
    beams      = [None] * 88
    for i,uid in enumerate( uids ):

        #beam_co   = make_gaussian_elliptical_beam( beam_nside, fwhm_x[i], fwhm_y[i], theta=rotation[i] )
        beam_co   = make_gaussian_beam( beam_nside, 1.5 )
        beam_cx   = numpy.zeros_like( beam_co )
        
        beams[uid] = { 'co': beam_co , 'cx': beam_cx, 'nside': beam_nside }
    #----------------------------------------------------------------------------------------------------------#

else:

    beams          = None
    inv_dets       = None
    location       = None
    map_nside      = None
    I_map          = None
    Q_map          = None
    U_map          = None
    V_map          = None
    focalPlane     = None
    tag            = None
    pointing_file  = None
    pointing_shape = None

beams    = comm.bcast( beams, root=0 )
inv_dets = comm.bcast( inv_dets, root=0 )
location = comm.bcast( location, root=0 )
map_nside = comm.bcast( map_nside, root=0 )
I_map    = comm.bcast( I_map, root=0 )
Q_map    = comm.bcast( Q_map, root=0 )
U_map    = comm.bcast( U_map, root=0 )
V_map    = comm.bcast( V_map, root=0 )
focalPlane = comm.bcast( focalPlane, root=0 )
tag        = comm.bcast( tag, root=0 )
pointing_file = comm.bcast( pointing_file, root=0 )
pointing_shape = comm.bcast( pointing_shape, root=0 )
comm.Barrier()

#----------------------------------------------------------------------------------------------------------#
# Read in pointing stream chunck
#----------------------------------------------------------------------------------------------------------#
#print 'loading pointing stream chunk %d' % (rank)
data_shape = shape_split( pointing_shape, size, axis=1 )[0][ rank ]
pointing   = numpy.load( pointing_file )
feed_ra    = pointing['ra'] [ data_shape ]
feed_dec   = pointing['dec'][ data_shape ]
feed_pa    = pointing['pa'] [ data_shape ]
data_shape = feed_ra.shape
print data_shape
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
# Setup data mask to avoid projecting invalid detectors
#----------------------------------------------------------------------------------------------------------#
data_mask = numpy.zeros( data_shape , dtype='int32' )
for invalid_detector in inv_dets: 
    data_mask[ invalid_detector ] += 1
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Setup detector data buffer
#----------------------------------------------------------------------------------------------------------#
detector_data = numpy.zeros( data_shape, dtype='float64' )
#----------------------------------------------------------------------------------------------------------#

# Commence deprojection procedure!!
for det in focalPlane['uid']:
   
    if det in inv_dets:
        print 'detector %d was flagged as invalid. Skipping.' % (det)
        continue
    
    # Find companion detector in the feedhorn
    det_feed = (int)( focalPlane['feed'][ focalPlane['uid'] == det ] )
    # Iterate over feedhorn until the matching detector is found
    for uid in focalPlane['uid']:
        
        if focalPlane['feed'][ uid ] == det_feed and uid != det:
            pair_det = uid
            break
            
    print 'deprojecting detector %d' % (det)
    data = deproject_sky_for_feedhorn( 
                  feed_ra[ det ], feed_dec[ det ], feed_pa[ det ],
                  numpy.radians( focalPlane['rot'][ det ] ),
                  (I_map,Q_map,U_map,V_map),
                  beams[ det ]['nside'],
                  numpy.copy(beams[ det      ]['co']), 
                  numpy.copy(beams[ det      ]['cx']), 
                  numpy.copy(beams[ pair_det ]['co']), 
                  numpy.copy(beams[ pair_det ]['cx']),
                  gpu_dev=rank, maxmem=1024 )
    
    print data.shape

    detector_data[ det ] = data

AtA, AtD   = update_matrices(
             feed_ra, feed_dec, feed_pa,
             numpy.radians( focalPlane['rot'] ),
             detector_data,
             map_nside,
             data_mask=data_mask )

# Save matrices
numpy.savez( './runs/matrices_%s_rank_%d.npz' % (tag, rank), rank=rank, AtA=AtA, AtD=AtD, nside=map_nside )
