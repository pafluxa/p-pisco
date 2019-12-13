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
from pisco.convolution.core import generate_TOD
from pisco.beam_analysis.mueller import ComplexMuellerMatrix as CM
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
print isOn

receiver = Receiver()
receiver.initialize( uids, azOff, elOff, detPol )
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Assemble beams across the focal plane
print 'assembling beamsors'
fwhm_x   = config['beams']['fwhmX']
fwhm_y   = config['beams']['fwhmY']
rotation = config['beams']['theta']

for x,y,r in zip(fwhm_x, fwhm_y, rotation): 
    print x, y, r

# TODO: put this in the config file
# Assemble all beamsors using "dumb" beams
# This is a very simple case where it is assumed that whatever mismatch is observed in the beams is not
# related to polarizing properties of the receiver. The consequence is that every detector has a beamsor
# that is "diagonal" by its own, i.e. all leakage componets are zero.
beam_nside = config['beamcfg']['nside']
beamsors   = []
for idx in range( receiver.ndets ):

    # Note fwhm_x and fwhm_y are inverted. This is because they come from fitting the beam
    # in az-alt coordinates, where x points East-West and Y North South.
    # This function uses HEALPix convention, where x points North-South and y East-West
    #print fwhm_y[idx], fwhm_x[idx]
    E_co    = make_gaussian_elliptical_beam( beam_nside, fwhm_y[idx], fwhm_x[idx], phi_0=-rotation[idx] )
    E_cx    = numpy.zeros_like( E_co )
    beamsor = CM.make_optical_mueller_matrix( beam_nside, E_co, E_cx, E_co, E_cx, numpy.radians(5) )

    #beamsor.plot( numpy.abs ) 
    beamsors.append( beamsor )
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Load maps from input
maps = numpy.load( map_path )
map_nside = maps['nside'][()]
I_map     = maps['I'] 
Q_map     = maps['Q'] 
U_map     = maps['U']
# Set this guy to zero for now.
#Q_map     = numpy.zeros_like( maps['I'] )
#U_map     = numpy.zeros_like( maps['I'] )
V_map     = numpy.zeros_like( maps['I'] )
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
for idx in numpy.arange( receiver.ndets ):
    
    if isOn[ idx ] == 1:
        
        print( idx, receiver.ndets )

        data = generate_TOD( 
                  # detector pointing
                  det_ra [ idx ], 
                  det_dec[ idx ], 
                  det_pa [ idx ],
                  receiver.pol_angles[ idx ],
                  # input sky
                  (I_map,Q_map,U_map,V_map),
                  # beamsor
                  beam_nside, beamsors[ idx ],
                  # use gpu zero, limit memory usage to 8 GB
                  gpu_dev=2, maxmem=7000 ) 

        detector_data[ idx ] = data

# make isOff int32
isOff = numpy.asarray( isOff, dtype='int32' )
print isOff
print receiver.pol_angles

out_nside = 128
AtA, AtD = update_matrices(
             det_ra, det_dec, det_pa,
             receiver.pol_angles,
             detector_data,
             out_nside,
             det_mask=isOff )

# Save matrices
numpy.savez( output_file, AtA=AtA, AtD=AtD, nside=out_nside )
