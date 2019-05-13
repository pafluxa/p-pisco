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
parser.add_argument( '-pointing', action='store', type=str, dest='pointing',
                      help='NPZ file with the pointing of the season. See generete_pointing.py for more details.' )
parser.add_argument( '-array_data', action='store', type=str, dest='array_data',
                      help='CSV file with the array specificiations.' )
args = parser.parse_args()

# Setup simulation tag
tag = args.tag

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
                      focalPlane['az_off'],
                      focalPlane['el_off'],
                      numpy.radians( focalPlane['rot'] ) )
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Read in pointing stream
#----------------------------------------------------------------------------------------------------------#
print 'loading pointing'
pointing = numpy.load( args.pointing , mmap_mode='r')
feed_ra  = pointing['ra']
feed_dec = pointing['dec']
feed_pa  = pointing['pa']
inv_dets = pointing['invalid_dets']
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Assemble beams across the focal plane
print 'assembling focal plane beams'
beam_nside = 512
beams      = [None] * 88
for uid in receiver.uid:
    
    #if uid in inv_dets:
    #    continue

    beam_co   = make_gaussian_beam( beam_nside, 1.5 )
    beam_cx   = numpy.zeros_like( beam_co )
     
    beams[uid] = { 'co': beam_co , 'cx': beam_cx, 'nside': beam_nside }
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Setup data mask to avoid projecting invalid detectors
#----------------------------------------------------------------------------------------------------------#
data_mask = numpy.zeros( feed_ra.shape , dtype='int32' )
for invalid_detector in inv_dets: 
    data_mask[ invalid_detector ] += 1
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Setup detector data buffer
#----------------------------------------------------------------------------------------------------------#
detector_data = numpy.zeros_like( feed_ra )
#----------------------------------------------------------------------------------------------------------#

min_dec = 1e10
max_dec =-1e10
# Commence deprojection procedure!!
for det in receiver.uid:
   
    if det in inv_dets:
        print 'detector %d was flagged as invalid. Skipping.' % (det)
        continue
    
    # Find companion detector in the feedhorn
    det_feed = (int)( focalPlane['feed'][ focalPlane['uid'] == det ] )
    # Iterate over feedhorn until the matching detector is found
    for uid in receiver.uid:
        
        if focalPlane['feed'][ uid ] == det_feed and uid != det:
            pair_det = uid
            break
    
    #if pair_det in inv_dets:
    #    print 'detector %d was flagged as invalid. Skipping.' % (pair_det)
    #    continue
        
    
    print 'deprojecting detector %d' % (det)
    data = deproject_sky_for_feedhorn( 
                  feed_ra[ det ], feed_dec[ det ], feed_pa[ det ],
                  receiver.pol_angles[ det ],
                  (I_map,Q_map,U_map,V_map),
                  beams[ det ]['nside'],
                  numpy.copy(beams[ det      ]['co']), 
                  numpy.copy(beams[ det      ]['cx']), 
                  numpy.copy(beams[ pair_det ]['co']), 
                  numpy.copy(beams[ pair_det ]['cx']),
                  gpu_dev=2, maxmem=7000 )
     
    detector_data[ det ] = data

    for uid in receiver.uid:                                                                                  
                                                                                                          
        if uid in inv_dets:                                                                               
            continue                                                                                          
                                                                                                              
        # Find minima and maxima of declination                                                               
        det_min_dec = numpy.min( feed_dec[uid] )                                                                   
        det_max_dec = numpy.max( feed_dec[uid] )                                                                   
                                                                                                              
        if det_min_dec < min_dec:                                                                             
            min_dec = det_min_dec                                                                             
            # Find feed corresponding to the uid                                                              
            min_feed = focalPlane[ 'feed' ][ uid ]                                                            
                                                                                                              
        if det_max_dec > max_dec:                                                                             
            max_dec = det_max_dec                                                                             
            # Find feed corresponding to the uid                                                              
            max_feed = focalPlane[ 'feed' ][ uid ]  

print numpy.degrees( max_dec ), max_feed
print numpy.degrees( min_dec ), min_feed

AtA, AtD = update_matrices(
             feed_ra, feed_dec, -feed_pa,
             receiver.pol_angles,
             detector_data,
             map_nside,
             data_mask=data_mask )

# Save matrices
numpy.savez( './runs/matrices_%s.npz' % (tag), AtA=AtA, AtD=AtD, nside=map_nside )








