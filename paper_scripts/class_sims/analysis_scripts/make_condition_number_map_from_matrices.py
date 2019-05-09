# coding: utf-8
import sys
import os
import healpy, numpy, pisco
from pisco.mapping.core import matrices_to_maps
import pylab

data = numpy.load( sys.argv[1] )
AtA,AtD,nside = data['AtA'], data['AtD'], data['nside'][()]

AtA = numpy.reshape( AtA, (-1,3,3) )

hit_map         = numpy.zeros( healpy.nside2npix( nside ) )
cond_number_map = numpy.zeros( healpy.nside2npix( nside ) )

for p, ata in enumerate(AtA):
   
    if ata[0][0] < 3:    
        continue

    hit_map[ p ]         = ata[0][0] 
    cond_number_map[ p ] = numpy.linalg.cond( ata )

healpy.mollview( cond_number_map, min=0, max=5 , title='' )
healpy.mollview( hit_map                       , title='' )
pylab.show()
