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
parser.add_argument( '-pointing', action='store', type=str, dest='pointing',
                      help='NPZ file with the pointing of the season. See generete_pointing.py for more details.' )
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

#----------------------------------------------------------------------------------------------------------#
# Read in pointing stream
#----------------------------------------------------------------------------------------------------------#
print 'loading pointing'
pointing = numpy.load( args.pointing )
feed_ra  = pointing['ra']
feed_dec = pointing['dec']
feed_pa  = pointing['pa']
inv_dets = pointing['invalid_dets']
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Read beam parameter file
print 'reading beams'
beam_files = glob.glob( os.path.join( args.beam_folder , '*.npz') )
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Assemble beams across the focal plane
print 'assembling focal plane beams'
uids       = focalPlane['uid']
beam_nside = 1024
beams      = [None] * ( receiver.ndets + 1 )
for feed in range( receiver.ndets + 1 ):
    
    if feed != 15 :
        continue

    # Load the beam corresponding the feeds
    try:
        uid_V, uid_H = focalPlane['uid'][ focalPlane['feed'] == feed ]
    except:
        continue

    # Print for double checking
    print 'feed', feed, 'corresponds to uids', uid_V, uid_H

    V_co = make_gaussian_beam( beam_nside, 1.5 )
    V_cx = numpy.zeros_like( V_co )
    H_co = make_gaussian_beam( beam_nside, 1.5 )
    H_cx = numpy.zeros_like( H_co )

    beams[ feed ] = { 'V_co': V_co , 'V_cx': V_cx, 'H_co':H_co, 'H_cx':H_cx, 'nside': beam_nside }
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

# Commence deprojection procedure!!
for feed in range( receiver.ndets + 1 ):
   
    if feed != 15:
        continue

    try:
        uid_V, uid_H = focalPlane['uid'][ focalPlane['feed'] == feed ]
    except:
        continue

    print 'deprojecting detector %d' % (uid_V)
    if uid_V in inv_dets:
        print 'detector %d was flagged as invalid. Skipping.' % (uid_V)
        continue
    
    data = deproject_sky_for_feedhorn( 
                  feed_ra[ uid_V ], feed_dec[ uid_V ], feed_pa[ uid_V ],
                  receiver.pol_angles[ uid_V ],
                  (I_map,Q_map,U_map,V_map),
                  beams[ feed ]['nside'],
                  numpy.copy(beams[ feed ]['V_co']), 
                  numpy.copy(beams[ feed ]['V_cx']), 
                  numpy.copy(beams[ feed ]['H_co']), 
                  numpy.copy(beams[ feed ]['H_cx']),
                  gpu_dev=0, maxmem=4096 )
    
    detector_data[ uid_V ] = data
    
    print 'deprojecting detector %d' % (uid_H)
    if uid_H in inv_dets:
        print 'detector %d was flagged as invalid. Skipping.' % (uid_H)
        continue
    
    data = deproject_sky_for_feedhorn( 
                  feed_ra[ uid_H ], feed_dec[ uid_H ], feed_pa[ uid_H ],
                  receiver.pol_angles[ uid_H ],
                  (I_map,Q_map,U_map,V_map),
                  beams[ feed ]['nside'],
                  numpy.copy(beams[ feed ]['V_co']), 
                  numpy.copy(beams[ feed ]['V_cx']), 
                  numpy.copy(beams[ feed ]['H_co']), 
                  numpy.copy(beams[ feed ]['H_cx']),
                  gpu_dev=0, maxmem=4096 )
    
    detector_data[ uid_H ] = data

AtA, AtD = update_matrices(
             feed_ra, feed_dec, feed_pa,
             receiver.pol_angles,
             detector_data,
             map_nside,
             data_mask=data_mask )

# Save matrices
numpy.savez( './runs/matrices_%s.npz' % (tag), AtA=AtA, AtD=AtD, nside=map_nside )

