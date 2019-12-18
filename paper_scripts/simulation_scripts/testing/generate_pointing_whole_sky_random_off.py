# coding: utf-8
import sys

import numpy
import healpy
from healpy.rotator import euler_matrix_new, Rotator

nside = int( sys.argv[1] )
symm_offsets = bool( int(sys.argv[2]) )
print( symm_offsets )

pixels = numpy.arange( healpy.nside2npix( nside ) )

tht,phi = healpy.pix2ang( nside, pixels )

ra  = phi
dec = numpy.pi/2.0 - tht

# add some random errors to the pointing
factor = 0.1
pixSize = healpy.nside2resol( nside )
x = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, ra.size  )
y = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, dec.size  )

# with offsets added
dec1 = dec + x
ra1  = ra  + y / numpy.cos(dec)       

# tile
ra1  = numpy.tile(  ra1, 3 )
dec1 = numpy.tile( dec1, 3 )

if symm_offsets == False:
    
    print( "generating pointing with random offsets" )

    # if symm_offsets is False, then we generate a new set of offsets and subtract
    # those from the original pointing.
    x = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, ra.size  )
    y = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, dec.size  )
else:
    # if symm_offsets is true, do nothing and subtract the offsets on the 
    # next whole sky sweep
    print( "generating pointing with symmetric random offsets" )
    pass

dec2 = dec - x
ra2  = ra  - y / numpy.cos(dec)
# tile
ra2  = numpy.tile(  ra2, 3 )
dec2 = numpy.tile( dec2, 3 )

# generate position angles per pixel (0,45 and 90 degrees)
# one sweep obsevers at a single pa, but we have 6 sweeps because
# of the offsets
pa0   = numpy.zeros_like( ra )
pa45  = pa0 + numpy.pi/4.0
pa90  = pa0 + numpy.pi/2.0
pa    = numpy.concatenate( (pa0,pa45,pa90,pa0,pa45,pa90) )

# and finally concatenate to emulate having two detectors instead of one
ra  = numpy.concatenate( ( ra1, ra2) )
dec = numpy.concatenate( (dec1,dec2) )

ra  = numpy.concatenate( (ra , ra) )
dec = numpy.concatenate( (dec,dec) )
pa  = numpy.concatenate( (pa,  pa) )

basename = ""
if symm_offsets == True:
    basename = 'withSymmetricRandomOffsets' 
else:
    basename = 'withRandomOffsets'

file_name = basename + '_ndays_{}_nscans_1_sps_1Hz.npz'.format( nside )

# print shape for sanity check
#print ra.shape, dec.shape, pa.shape
#print 2*3*2*pixels.size
print( "writing to disk..." )
numpy.savez( file_name , ra=ra,dec=dec,pa=pa )
print( "done writing" )
