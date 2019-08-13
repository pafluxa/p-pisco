# coding: utf-8
import sys
import os
import healpy, numpy, pisco
from pisco.mapping.core import matrices_to_maps
import pylab

data = numpy.load( sys.argv[1] )
AtA,AtD,nside = data['AtA'], data['AtD'], data['nside'][()]

AtA = numpy.reshape( AtA, (-1,3,3) )

cond_number_map = numpy.zeros( healpy.nside2npix( nside ) )
for p, ata in enumerate(AtA):
   
    if ata[0][0] < 3:    
        continue

    cond_number_map[ p ] = ata[0][0] #numpy.linalg.cond( ata )

healpy.mollview( cond_number_map, min=0, max=1000.0, title='Hit map' )
pylab.show()
