# coding: utf-8
import sys

import numpy
import healpy
from healpy.rotator import euler_matrix_new, Rotator

npsb = 1
ndets = 2*npsb

nside = int( sys.argv[1] )

pixels = numpy.arange( healpy.nside2npix( nside ) )

tht,phi = healpy.pix2ang( nside, pixels )

ra  = phi
dec = numpy.pi/2.0 - tht
# generate position angles per pixel (0,45 and 90 degrees)
# one sweep obsevers at a single pa, but we have 6 sweeps because
# of the offsets
pa0   = numpy.zeros_like( ra )
pa45  = pa0 + numpy.pi/4.0
pa90  = pa0 + numpy.pi/2.0

# to make single detector pointing
ra1  = numpy.tile(  ra, 3 )
dec1 = numpy.tile( dec, 3 )
pa1  = numpy.concatenate( (pa0,pa45,pa90) ) 

ra  = numpy.tile(  ra1, (ndets,1) ) 
dec = numpy.tile( dec1, (ndets,1) )
pa  = numpy.tile(  pa1, (ndets,1) )

basename  = 'pixelCenters' 
file_name = basename + '_ndays_{}_nscans_1_sps_1Hz.npz'.format( nside )

# print shape for sanity check
print ra.shape, dec.shape, pa.shape
print 3*pixels.size
print( "writing to disk..." )
numpy.savez( file_name , ra=ra,dec=dec,pa=pa )
print( "done writing" )
