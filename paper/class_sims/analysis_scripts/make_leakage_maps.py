# coding: utf-8
import sys
import os
import healpy, numpy, pisco
from pisco.mapping.core import matrices_to_maps
import pylab

data = numpy.load( sys.argv[1] )
AtA,AtD,nside = data['AtA'], data['AtD'], data['nside'][()]

AtA = numpy.reshape( AtA, (-1,3,3) )

leak_maps = numpy.zeros( (3,3,healpy.nside2npix( nside )) )
for p, ata in enumerate(AtA):
   
    if ata[0][0] < 3:    
        continue
    
    inv_ata = numpy.linalg.inv( ata )
    leak_maps[:,:,p] = inv_ata
    '''
    for row in range(0,3):
        for col in range(0,3):
            leak_maps[row][col][p] = inv_ata[row][col]
    '''
print leak_maps.shape

for row in range( 0,3 ):
    for col in range(0,3):
        
        i = row*3 + col
        healpy.mollview( leak_maps[row][col]/leak_maps[0][0], min=-0.1, max=0.1, sub=(3,3,i+1) )

pylab.show()
