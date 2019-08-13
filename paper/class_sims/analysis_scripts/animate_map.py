# coding: utf-8
import sys
import os
import healpy, numpy, pisco
from pisco.mapping.core import matrices_to_maps
import pylab

import pylab
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import healpy
from   healpy.projector import MollweideProj
from   healpy.pixelfunc import vec2pix



# Load data for the first time
data = numpy.load( sys.argv[1] )
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
I,Q,U,W = matrices_to_maps( NSIDE, AtA, AtD )

# Define spherical projection axis
vec2pix_func = lambda x,y,z: vec2pix( NSIDE,x,y,z,nest=False)
fig  = pylab.figure()
ax_I = fig.add_subplot(111)
#ax_W = fig.add_subplot(212)

proj = MollweideProj()
Q_proj = proj.projmap( I , vec2pix_func )

#proj = MollweideProj()
#W_proj = proj.projmap( W , vec2pix_func )

imageI = ax_I.imshow( Q_proj,
                   vmin=-Q[W>10].std()/2, vmax=Q[W>10].std()/2,
                   cmap=plt.get_cmap('gray'),
                   extent=( (0,360,-90,90) ),
                   origin='lower')
'''
imageW = ax_W.imshow( W_proj,
                   vmin=0.0, vmax=W.max(),
                   cmap=plt.get_cmap('gray'),
                   extent=( (0,360,-90,90) ),
                   origin='lower')
'''

def update(i):

    global I,W

    try:
        data = numpy.load( sys.argv[1] )
        AtA,AtD,nside = data['AtA'], data['AtD'], data['nside'][()]
        i,q,u,w = matrices_to_maps( nside, AtA, AtD )

        Q = numpy.copy(q)
        #W = numpy.copy(w)

        Q_proj = proj.projmap( I, vec2pix_func )
        #W_proj = proj.projmap( W, vec2pix_func )

        imageI.set_array( Q_proj )
        #image_W.set_array( W_proj )

        return imageI, #imageW,

    except:
        return imageI, #imageW,

ani = animation.FuncAnimation( fig, update, interval=2000, blit=True )
plt.show()


