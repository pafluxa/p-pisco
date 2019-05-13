#!/usr/bin/env python
# coding: utf-8
import sys

import pandas
import time
import numpy
import healpy
import pylab

from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.mapping.core import *
from pisco.pointing import *

# Load maps from input
maps = numpy.load( sys.argv[1] )
map_nside = maps['nside'][()]
I_map     = maps['I'] 
Q_map     = maps['Q'] 
U_map     = maps['U'] 
V_map     = maps['V']


#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( './data/array_data/piscotel.csv' )
#----------------------------------------------------------------------------------------------------------#
receiver = Receiver()
receiver.initialize(  focalPlane['uid'],
                     -focalPlane['az_off'],
                      focalPlane['el_off'],
                      numpy.radians( focalPlane['rot'] ) )

#----------------------------------------------------------------------------------------------------------#
# Read beam from GRASP simulation
# This loads a single beam simulation!!
#----------------------------------------------------------------------------------------------------------#
beam_data = numpy.load( './data/beams/piscotel_beam.npz' )
nx        = beam_data['mdata_x'][1][0]['nx'] 
limits    = beam_data['mdata_x'][1][0]['limits']
grid_side = numpy.radians( limits[1] - limits[0] )
beam_co   = beam_data['Ex_co']
beam_cx   = beam_data['Ex_cx'] 
# Assign beam to a dictionary. This allows to iterate over many beams if required
beam = { 'co': beam_co , 'cx': beam_cx, 'nx': nx, 'grid_side':grid_side }

# Set projection matrices to zero
AtA = 0.0
AtD = 0.0

# Setup mock scan. Visit each pixel once...
tht, phi = healpy.pix2ang( map_nside, numpy.arange( healpy.nside2npix( map_nside ) ) )
ra  = phi
dec = numpy.pi/2.0 - tht
pa  = numpy.zeros_like( ra )

# ...but at different orientations!
for boresight_rotation in numpy.radians( numpy.linspace(-180,180,4) ):

    # Make some noise to known the program is alive
    print numpy.degrees( boresight_rotation )

    # Get beam from the dictionary
    grid_nx   = beam['nx']
    grid_side = beam['grid_side']
    beam1_co,beam1_cx = numpy.copy(beam['co']), numpy.copy(beam['cx'])
    beam2_co,beam2_cx = numpy.copy(beam['co']), numpy.copy(beam['cx'])
    
    # Synthetize TOD
    tod  = deproject_sky_for_feedhorn(
        ra, dec, pa + boresight_rotation,
        receiver.pol_angles[0],
        (I_map,Q_map,U_map,V_map),
        grid_side, grid_nx,beam1_co,beam1_cx, beam2_co, beam2_cx,
        gpu_dev=0 )
    
    # Add TOD to map space in matrix form
    ata, atd = update_matrices(
                     ra.reshape((1,-1)),
                     dec.reshape((1,-1)),
                     pa.reshape((1,-1)) + boresight_rotation,
                     receiver.pol_angles,
                     tod.reshape((1,-1)),
                     map_nside )
    AtA += ata
    AtD += atd

    # Save matrices
    numpy.savez( './runs/matrices_piscotel_input_lcdmr0d10_id_0000.npz', AtA=AtA, AtD=AtD, nside=map_nside )

