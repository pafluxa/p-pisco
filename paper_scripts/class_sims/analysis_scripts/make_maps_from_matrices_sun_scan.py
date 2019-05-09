# coding: utf-8
import sys
import os

import numpy
import numpy as np

import pylab
import matplotlib.pyplot as plt

import healpy
from   healpy.projector import MollweideProj, CartesianProj
from   healpy.pixelfunc import vec2pix

from pisco.mapping.core import matrices_to_maps

from cmb_analysis.powerspectrum.pyspice import spice

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

# Load data from user input
data = numpy.load( sys.argv[1] )
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
# temporary fix
NSIDE = 256
I,Q,U,W = matrices_to_maps( NSIDE, AtA, AtD )

# Create a window
window = numpy.zeros_like( W , dtype='bool' )

# Define spherical projection axis
vec2pix_func = lambda x,y,z: vec2pix( NSIDE,x,y,z,nest=False)

I[ W == 0 ] = healpy.UNSEEN
I[ W >  0 ] = I[ W > 0 ]/W[ W > 0 ]
logI = numpy.copy( I )
logI[ W > 0 ] = 10*numpy.log10( logI[ W > 0] )

sun_image = healpy.gnomview( I, rot=(0,45), min=-40, max=20, reso=10, xsize=600, ysize=600, return_projected_map=True )

dist = 10/60.0 * numpy.arange(-300,300)
pylab.figure()
pylab.plot( dist, sun_image[ 600/2 ] )
#healpy.graticule( 10, 10, local=True )
pylab.show()



