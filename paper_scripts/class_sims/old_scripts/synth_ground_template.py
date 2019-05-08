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

from pypoint.core import focalplane_to_horizontal
from pypoint.core import focalplane_to_equatorial

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
parser = argparse.ArgumentParser(description='Simulate CLASS maps using provided pointing.')
parser.add_argument( '-maps' , action='store', type=str, dest='maps', 
                      help='Input map to deproject' )
parser.add_argument( '-tag' , action='store', type=str, dest='tag',
                      help='Name of the output. Output will be matrices_tag.npz' )
parser.add_argument( '-pointing', action='store', type=str, dest='pointing',
                      help='NPZ file with the pointing of the season. See generete_pointing.py for more details.' )
parser.add_argument( '-array_data', action='store', type=str, dest='array_data',
                      help='CSV file with the array specificiations.' )
parser.add_argument( '-beams',  action='store', type=str, dest='beams_path',
                      help='Path to detector beams in HEAPIX format, packed as npz files.')
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
Q_map     = maps['Q']*0 
U_map     = maps['U']*0
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
# Get invalid detetors
#inv_dets = list( receiver.uid )
#inv_dets.remove( 57 )
inv_dets = numpy.argwhere( numpy.isnan( receiver.dx ) ).T[0]
print inv_dets

#----------------------------------------------------------------------------------------------------------#


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
# Read beam parameter file
print 'reading beam parameters'
beam_data = pandas.read_csv( args.beam_par_file )
uids      = beam_data[ 'uid']
feeds     = beam_data[ 'feed']
fwhm_x    = beam_data[ 'fwhm_x']
fwhm_y    = beam_data[ 'fwhm_y']
rotation  = beam_data[ 'rot']
#----------------------------------------------------------------------------------------------------------#

# Read in pointing
#----------------------------------------------------------------------------------------------------------#
pointing = numpy.load( args.pointing )
ctime    = pointing['ctime']
az       = pointing[  'azm']
alt      = pointing[  'alt']
rot      = pointing[  'rot']
#----------------------------------------------------------------------------------------------------------#

# Convert boresight pointing into horizontal pointing for all detectors
az_det, alt_det, rot_det = focalplane_to_horizontal(
    az, alt, rot,
    receiver.dx, receiver.dy, receiver.pol_angles*0.0 )

print az_det.shape

# Convert boresight pointing into equatorial coordinates using a dummy receiver
boresight = Receiver()
boresight.initialize( [0.0], [0.0], [0.0], [0.0] )

tod = TOD()
tod.initialize( ctime, az, alt, rot )
ra_bor, dec_bor, pa_bor = compute_receiver_to_ICRS( tod, boresight, location )

ra_det, dec_det, pa_det = focalplane_to_equatorial(
    ra_bor, dec_bor, pa_bor,
    receiver.dx, receiver.dy, receiver.pol_angles )

#----------------------------------------------------------------------------------------------------------#
# Setup detector data buffer
#----------------------------------------------------------------------------------------------------------#
nsamples      = tod.nsamples
print nsamples
detector_data = numpy.zeros( (receiver.ndets, nsamples) , dtype='float64' )
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Setup data mask to avoid projecting invalid detectors
#----------------------------------------------------------------------------------------------------------#
data_mask = numpy.zeros( (receiver.ndets, nsamples) , dtype='int32' )
for invalid_detector in inv_dets: 
    data_mask[ invalid_detector ] += 1
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Assemble beams in the focal plane
print 'assembling focal plane beams'
beams      = [None] * receiver.ndets
for uid in uids:
    
    print 'loading', os.path.join( args.beams_path ,'detector%d.npz' % (uid) )
    
    data = numpy.load( os.path.join( args.beams_path ,'detector%d.npz' % (uid) ) )
    
    beam_nside = data['nside'][()] 
    E_co, E_cx = data['E_co'], data['E_cx']
    
    beam_co   = numpy.abs(E_co)
    beam_cx   = numpy.abs(E_cx)
    
    beams[uid] = { 'co': beam_co , 'cx': beam_cx, 'nside': beam_nside }
#----------------------------------------------------------------------------------------------------------#

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
                  az_det[ det ], alt_det[ det ], rot_det[ det ],
                  receiver.pol_angles[ det ],
                  (I_map,Q_map,U_map,V_map),
                  beams[ det ]['nside'],
                  numpy.copy(beams[ det      ]['co']), 
                  numpy.copy(beams[ det      ]['cx']), 
                  numpy.copy(beams[ pair_det ]['co']), 
                  numpy.copy(beams[ pair_det ]['cx']),
                  gpu_dev=0, maxmem=7000, grid_size=1.0 )
    
    pylab.plot( data.T, alpha=0.5 )
    pylab.show()

    detector_data[ det ] = data
    
# Reproject data to the sky

# Create a detector mask
det_mask = numpy.zeros_like( receiver.uid , dtype='int32' )
det_mask[ inv_dets ] = 1

AtA, AtD = update_matrices( 
    ra_det, dec_det, pa_det,
    receiver.pol_angles*0.0,
    detector_data,
    128,
    data_mask=data_mask,
    det_mask=det_mask)

numpy.savez( 'matrices_%s.npz' % (args.tag), AtA=AtA, AtD=AtD, nside=128, det_mask=det_mask )





