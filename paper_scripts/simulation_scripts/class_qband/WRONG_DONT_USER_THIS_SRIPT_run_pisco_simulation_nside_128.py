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

from config_parser import parse_config_file

# Do this shit right with argparser
import argparse
parser = argparse.ArgumentParser(description='Simulate CLASS Q-band experiment using elliptical Gaussian beams.')

parser.add_argument( '-config' , action='store', type=str, dest='configFile', 
                      help='Path to configuration file.' )

args = parser.parse_args()

# Setup simulation tag
config = parse_config_file( args.configFile )

# Build map path. WARNING: HARD CODED VALUES!!
map_path = config['inputMapPath']
print map_path

output_file = config['outputFile']
print output_file

uids   = config['focalplane']['uids']
feeds  = config['focalplane']['feeds']
azOff  = config['focalplane']['azOff']
elOff  = config['focalplane']['elOff']
detPol = config['focalplane']['detPol']
isOn   = config['focalplane']['on']
isOff  = (isOn + 1) % 2 
print isOff

receiver = Receiver()
receiver.initialize( uids, azOff, elOff, detPol )
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Assemble beams across the focal plane
fwhm_x   = config['beams']['fwhmX']
fwhm_y   = config['beams']['fwhmY']
rotation = config['beams']['theta']
print 'assembling focal plane beams'
# TODO: put this in the config file
beam_nside = 512
beams      = [None] * receiver.ndets
for idx in range( receiver.ndets ):

    # Note fwhm_x and fwhm_y are inverted. This is because they come from fitting the beam
    # in az-alt coordinates, where x points East-West and Y North South.
    # This function uses HEALPix convention, where x points North-South and y East-West
    
    print fwhm_x[idx], fwhm_y[idx], rotation[idx]

    beam_co   = make_gaussian_elliptical_beam( beam_nside, fwhm_y[idx], fwhm_x[idx], phi_0=-rotation[idx] )
    beam_cx   = numpy.zeros_like( beam_co )
    
    beams[idx] = { 'co': beam_co , 'cx': beam_cx, 'nside': beam_nside }
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
print 'loading pointing from', config['pointingFile']
pointing = numpy.load( config['pointingFile'] , mmap_mode='r')
det_ra  = pointing['ra']
det_dec = pointing['dec']
det_pa  = pointing['pa']
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Setup detector data buffer
#----------------------------------------------------------------------------------------------------------#
detector_data = numpy.zeros_like( det_ra )
#----------------------------------------------------------------------------------------------------------#

# Generate TOD data 
for feed in numpy.unique( feeds ):
    
    # Find all uids that share the same feed
    pairs = uids[ feeds == feed ] 
    
    for i,det in zip(range(2), pairs):
        
        pair = pairs[ (i+1) % 2 ]
        idx0 = numpy.argwhere( uids ==  det )[0][0]
        idx1 = numpy.argwhere( uids == pair )[0][0]

        print "detector", det, "with index", idx0, "is in feed", feed, \
               "and is paired to detector", pair, "which has index", idx1
        

        data = None

        if isOn[ idx0 ] == 1:

            data = deproject_sky_for_feedhorn( 
                      # detector pointing
                      det_ra [ idx0 ], 
                      det_dec[ idx0 ], 
                      det_pa [ idx0 ],
                      receiver.pol_angles[ idx0 ],
                        
                      # input sky
                      (I_map,Q_map,U_map,V_map),
                      
                      # beams, to build beam tensor
                      beams[ idx0 ]['nside'],
                      beams[ idx0 ][   'co'], 
                      beams[ idx0 ][   'cx'], 
                      beams[ idx1 ][   'co'], 
                      beams[ idx1 ][   'cx'],
                    
                      # use gpu zero, limit memory usage to 7 GB
                      gpu_dev=0, maxmem=7000 )
        
        else:
            data = numpy.zeros_like( det_ra[ idx0 ] )

        detector_data[ idx0 ] = data

# make isOff int32
isOff = numpy.asarray( isOff, dtype='int32' )

out_nside = 128

AtA, AtD = update_matrices(
             det_ra, det_dec, det_pa,
             receiver.pol_angles,
             detector_data,
             out_nside,
             det_mask=isOff )

# Save matrices
numpy.savez( output_file, AtA=AtA, AtD=AtD, nside=out_nside )
