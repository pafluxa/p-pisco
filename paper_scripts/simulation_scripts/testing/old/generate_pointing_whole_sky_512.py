# coding: utf-8
import numpy
import healpy

# Not needed anymore
#from healpy.rotator import euler_matrix_new, Rotator

# TODO: this should come from a config file!
nside = 512
pixels = numpy.arange( healpy.nside2npix( nside ) )

tht,phi = healpy.pix2ang( nside, pixels )
# this is not needed anymore, but can be used to slighly shift the pointing
# from pixel centers 
#tht_rot,phi_rot = Rotator(rot=(0.0,0.0,0.0))(tht, phi)


# TODO: this should come from a configuration file
# number of PSB
n_psb = 1 
n_det = 2 * n_psb

# array with position angles of beam center during scanning.
observing_angles = numpy.asarray( (0.0, 45.0, 90.0) )
# to radians!!
observing_angles = numpy.deg2rad( observing_angles )

# number of passes is the number of observing angles
n_passes = len( observing_angles )

# the number of samples in every pass is just the length of `tht`
# basically the number of pixels in the input map
n_samples = tht.size

# setup the buffers
ra  = numpy.zeros( (n_det, n_passes , n_samples) )
dec = numpy.zeros( (n_det, n_passes , n_samples) )
pa  = numpy.zeros( (n_det, n_passes , n_samples) )

# generate pointing for each detector
for det in range(n_det):
    
    for p,obs_angle in enumerate( observing_angles ):

        _ra  = phi
        _dec = numpy.pi/2.0 - tht
        _pa  = numpy.zeros_like( _ra ) + obs_angle

        ra [ det, p ] = phi
        dec[ det, p ] = numpy.pi/2.0 - tht
        pa [ det, p ] = numpy.zeros_like( phi ) + obs_angle

# clever re-shape
ra  = numpy.reshape( ra,  (n_det, n_passes * n_samples) )
dec = numpy.reshape( dec, (n_det, n_passes * n_samples) )
pa  = numpy.reshape( pa,  (n_det, n_passes * n_samples) )

# Make sure declation doesn't go outside limits
dec[ dec >  numpy.pi/2.0 ] =  numpy.pi/2.0
dec[ dec < -numpy.pi/2.0 ] = -numpy.pi/2.0

print ra.shape, dec.shape, pa.shape

numpy.savez( '../../data/pointing/wholeSkyPointing_ndays_%d_nscans_1_sps_1Hz.npz' % (nside) , ra=ra,dec=dec,pa=pa )
