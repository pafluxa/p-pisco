# coding: utf-8
import pisco
from pisco.mapping.core import *

import pylab
import time
import numpy
import healpy

# Setup raster scan
nsamples = 500
ndets    =   1
ra   = numpy.linspace( -0.1,0.1, nsamples )
dec  = numpy.linspace( -0.1,0.1, nsamples )

ra   = numpy.repeat( ra   , dec.size )
dec  = numpy.tile  ( dec  , dec.size )
pa   = numpy.random.random( ra.size )

phi    = ra.reshape( (ndets,-1) )
theta  = numpy.pi/2.0 - dec.reshape( (ndets,-1) )
psi    = pa.reshape( (ndets,-1) )

# Setup map matrices
map_nside = 256

# Setup data
data = numpy.random.random( (ndets,ra.size) )
data = data.astype( 'float32' )

AtA, AtD = update_matrices( phi, theta, psi, data, map_nside )
I,Q,U,W  = matrices_to_maps( map_nside, AtA, AtD )

I[ W==0 ] = healpy.UNSEEN
healpy.gnomview( I , xsize=600, ysize=600, min=-1, max=1 )
pylab.show()

