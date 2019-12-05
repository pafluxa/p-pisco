# coding: utf-8
import numpy
import healpy
from healpy.rotator import euler_matrix_new, Rotator

nside = 128
pixels = numpy.arange( healpy.nside2npix( nside ) )

tht,phi = healpy.pix2ang( nside, pixels )

ra  = phi
dec = numpy.pi/2.0 - tht

pa0 = numpy.zeros_like( ra )
pa1 = pa0 + numpy.pi/4.0
pa2 = pa0 + numpy.pi/2.0
pa  = numpy.concatenate( (pa0,pa1,pa2) )

ra  = numpy.tile( ra, 3 )
dec = numpy.tile( dec, 3 )

ra  =  ra.reshape( (1,-1) )
dec = dec.reshape( (1,-1) )
pa  =  pa.reshape( (1,-1) )

numpy.savez( '../../data/pointing/wholeSkyPointing_ndays_128_nscans_1_sps_1Hz.npz' , ra=ra,dec=dec,pa=pa )
