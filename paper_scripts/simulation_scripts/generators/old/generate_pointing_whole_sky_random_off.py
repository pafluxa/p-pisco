# coding: utf-8
import numpy
import healpy
from healpy.rotator import euler_matrix_new, Rotator

nside = 512
pixels = numpy.arange( healpy.nside2npix( nside ) )

tht,phi = healpy.pix2ang( nside, pixels )

tht_rot,phi_rot = Rotator(rot=(0.0,0.0,0.0))(tht, phi)

ra  = phi_rot
dec = numpy.pi/2.0 - tht_rot

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

# generate position angles: 1 per pass.
pa0 = numpy.zeros_like( ra )
pa1 = pa0 + numpy.pi/4.0
pa2 = pa0 + numpy.pi/2.0
pa  = numpy.concatenate( (pa0,pa1,pa2) )

'''
# with offsets subtracted
dec2 = dec - x
ra2  = ra  - y / numpy.cos(dec)

# tile
ra2  = numpy.tile(  ra2, 3 )
dec2 = numpy.tile( dec2, 3 )

# concatenate
ra  = numpy.concatenate( ( ra1, ra2) )
dec = numpy.concatenate( (dec1,dec2) )
pa  = numpy.concatenate( (  pa,  pa) )
'''
ra  = numpy.concatenate( (ra1,ra1) )
dec = numpy.concatenate( (dec1,dec1) )
pa  = numpy.concatenate( (pa,pa) )

# Make sure declation doesn't go outside limits
dec[ dec >  numpy.pi/2.0 ] =  numpy.pi/2.0
dec[ dec < -numpy.pi/2.0 ] = -numpy.pi/2.0

# reshape to fool PISCO
ra  =  ra.reshape( (2,-1) )
dec = dec.reshape( (2,-1) )
pa  =  pa.reshape( (2,-1) )

print ra.shape, dec.shape, pa.shape

numpy.savez( '../../data/pointing/wholeSkyPointingRandomOffsets_ndays_512_nscans_1_sps_1Hz.npz' , ra=ra,dec=dec,pa=pa )
