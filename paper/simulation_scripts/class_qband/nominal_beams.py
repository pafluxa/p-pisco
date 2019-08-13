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

import os
import pandas
import time
import numpy
from numpy import pi
import healpy
import pylab

# Do this shit right with argparser
import argparse
parser = argparse.ArgumentParser(description='Simulate CLASS Q-band experiment using elliptical Gaussian beams.')

parser.add_argument( '-map_base_path' , action='store', type=str, dest='map_base_path', 
                      help='Base path to maps. Must be in format <path>/rXdXX/n_%04d_nside=%03d.npz' )

parser.add_argument( '-r' , action='store', type=float, dest='r', 
                      help='Value of r for the input map' )

parser.add_argument( '-tag' , action='store', type=str, dest='tag',
                      help='' )

parser.add_argument( '-pointing', action='store', type=str, dest='pointing',
                      help='NPZ file with the pointing of the season. See generete_pointing.py for more details.' )

parser.add_argument( '-array_data', action='store', type=str, dest='array_data',
                      help='CSV file with the array specificiations.' )

parser.add_argument( '-beam_par_file',  action='store', type=str, dest='beam_par_file',
                      help='File with specifications of the beam parameters for each detector.')

args = parser.parse_args()

# Setup simulation tag
tag = args.tag

# Build map path. WARNING: HARD CODED VALUES!!
map_path = args.map_base_path
map_path = os.path.join( map_path, 'r%01.2f' % (args.r ), 'n_%04d_nside=%d.npz' % ( 0, 128 ) )
print map_path

# Build paths for output
tags   = tag.split( '_' )
props  = tags[0::2]
values = tags[1::2]

# append r value
props.append ( 'r' )
values.append( "%1.2f" % (args.r) )

# WARNING: HARDCODED VALUE!!
out_path = './runs/'
for p,v in zip( props, values ):
    out_path = os.path.join( out_path, str(p), str(v) )
print out_path

output_file = os.path.join( out_path, 'matrices.npz' )
print output_file

# check if path exists and ask what's up with that
os.makedirs( out_path )

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
# Setup experiment location
location = EarthLocation( lat= -22.959748*u.deg, lon=-67.787260*u.deg, height=5200*u.m )
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
# Assemble beams across the focal plane
print 'assembling focal plane beams'
beam_nside = 512
beams      = [None] * receiver.ndets
for i,uid in enumerate( uids ):
    
    # Note fwhm_x and fwhm_y are inverted. This is because they come from fitting the beam
    # in az-alt coordinates, where x points East-West and Y North South.
    # This function uses HEALPix convention, where x points North-South and y East-West.
    beam_co   = make_gaussian_elliptical_beam( beam_nside, 1.5, 1.5, phi_0=0.0 )
    beam_cx   = numpy.zeros_like( beam_co )
   
    beams[uid] = { 'co': beam_co , 'cx': beam_cx, 'nside': beam_nside }
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Load maps from input
maps = numpy.load( map_path )
map_nside = maps['nside'][()]
I_map     = maps['I'] 
Q_map     = maps['Q'] 
U_map     = maps['U']

# Set this guy to zero for now.
V_map     = maps['V']*0
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Read in pointing stream
#----------------------------------------------------------------------------------------------------------#
print 'loading pointing'
print args.pointing

pointing = numpy.load( args.pointing , mmap_mode='r')
feed_ra  = pointing['ra']
feed_dec = pointing['dec']
feed_pa  = pointing['pa']
inv_dets = pointing['invalid_dets']
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
                  gpu_dev=0, maxmem=7000 )
    
    detector_data[ det ] = data

AtA, AtD = update_matrices(
             feed_ra, feed_dec, feed_pa,
             receiver.pol_angles,
             detector_data,
             map_nside,
             data_mask=data_mask )

# Save matrices
numpy.savez( output_file, AtA=AtA, AtD=AtD, nside=map_nside )
