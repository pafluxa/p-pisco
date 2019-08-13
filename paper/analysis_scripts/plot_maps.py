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

# Load data from user input
data = numpy.load( sys.argv[1] )

AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]

pI,pQ,pU,pW = matrices_to_maps( NSIDE, AtA, AtD )

# load original maps
oI = numpy.load( sys.argv[2] )['I']
oQ = numpy.load( sys.argv[2] )['Q']
oU = numpy.load( sys.argv[2] )['U']

# transform to uK 
pI *= 1e6
# to nK
pQ *= 1e9
pU *= 1e9

oI *= 1e6
# to nK
oQ *= 1e9
oU *= 1e9

w  = numpy.ones_like( pW )
w[ numpy.where( pW < 10 ) ] = 0

# Load beam FWHM from user input, in degrees
beam_fwhm = (float)( sys.argv[3] )

# Smooth original maps
sI,sQ,sU = healpy.smoothing( (oI,oQ,oU), fwhm=numpy.deg2rad( beam_fwhm ), pol=True )

# Define spherical projection axis
vec2pix_func = lambda x,y,z: vec2pix( NSIDE,x,y,z,nest=False)

fig = pylab.figure()

# Setup projection axes
ax_I = fig.add_subplot(131)
ax_Q = fig.add_subplot(132)
ax_U = fig.add_subplot(133)

proj = CartesianProj()

# Perform projection
I_proj = proj.projmap( (pI - sI) * w, vec2pix_func )
Q_proj = proj.projmap( (pQ - sQ) * w, vec2pix_func )
U_proj = proj.projmap( (pU - sU) * w, vec2pix_func )

maxrI,minrI = 0.5, -0.5
rIticks     = numpy.linspace( minrI, maxrI, 5 )
rmsRI       = numpy.std( I_proj )
# plot I residual map
ax_I.set_title( 'residual I has %2.2f uK of RMS noise.' % (rmsRI) )
imageI = ax_I.imshow( I_proj,
               vmin=minrI, vmax=maxrI,
               cmap=plt.get_cmap('bwr'),
               extent=( (-180,180,-90,90) ),
               origin='lower')
pylab.colorbar( imageI, ax=ax_I, orientation='horizontal', 
                ticks=rIticks, format='%+1.1f' ) 

maxrQ,minrQ = 5, -5
rQticks     = numpy.linspace( minrQ, maxrQ, 5 )
rmsRQ       = numpy.std( Q_proj )

ax_Q.set_title( 'residual Q has %2.2f nK of RMS noise.' % (rmsRQ) )
imageQ = ax_Q.imshow( Q_proj,
               vmin=minrQ, vmax=maxrQ,
               cmap=plt.get_cmap('bwr'),
               extent=( (-180,180,-90,90) ),
               origin='lower')
pylab.colorbar( imageQ, ax=ax_Q, orientation='horizontal', 
                ticks=rQticks, format='%+1.1f' ) 

maxrU,minrU = 5, -5
rUticks     = numpy.linspace( minrU, maxrU, 5 )
rmsRU       = numpy.std( U_proj )

ax_U.set_title( 'residual U has %2.2f nK of RMS noise.' % (rmsRU) )
imageU = ax_U.imshow( U_proj,
               vmin=minrU, vmax=maxrU,
               cmap=plt.get_cmap('bwr'),
               extent=( (-180,180,-90,90) ),
               origin='lower')
pylab.colorbar( imageU, ax=ax_U, orientation='horizontal', 
                ticks=rUticks, format='%+1.1f' ) 

pylab.show()

