#!/usr/bin/env python
# coding: utf-8
import pisco
from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.mapping.core import *

import sys
import time
import numpy
from numpy import pi
import healpy
import pylab

# Load LCDM model with r = 0
map_data  = numpy.load( sys.argv[1] )
map_nside = map_data['nside'][()]

I_map   = map_data['I']
Q_map   = map_data['Q']
U_map   = map_data['U']
V_map   = map_data['V']

map_nside = 256
I_map,Q_map,U_map,V_map = healpy.ud_grade( (I_map,Q_map,U_map,V_map), map_nside )

ra_s   = 0
dec_s  = 0 
v_c    = healpy.ang2vec( numpy.pi/2.0 - dec_s, ra_s )

# Setup a simple Gaussian beam of 10 arcmin
beam_npix   = 101
beam_fwhm   = 1.5
beam_extn   = numpy.radians(7.0) # 7 degrees

# Beam for feedhorn 1
beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_npix, amplitude=1.00 )
beam0_cx  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_npix, amplitude=0.00)

beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_npix, amplitude=1.00 )
beam90_cx = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_npix, amplitude=0.00 )

# Get alm space smoothed input maps
I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(beam_fwhm), pol=True )
#I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_smooth,Q_smooth,U_smooth), sigma=numpy.radians(1.5/3.0), pol=False )

# Setup ring scan
#scan_pixels = healpy.query_disc( map_nside, v_c, numpy.radians(10.0) )
scan_pixels = numpy.arange( 0, healpy.nside2npix( map_nside ) )
print len(scan_pixels)
theta, phi  = healpy.pix2ang( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta
pa   = numpy.zeros_like( ra )

# perform deprojection aliged with the source
AtA,AtD = 0.0, 0.0
scan_angles = numpy.linspace( 0, pi/2.0, 3 )
#scan_angles = [0.0,pi/4.0,pi/2.0]
det_angle   = 0.0
#for i,det_angle in enumerate(det_angles):
for i, scan_angle in enumerate(scan_angles):

    print i+1
    tod = deproject_sky_for_feedhorn(
        ra, dec, pa + scan_angle,
        det_angle,
        (I_map,Q_map,U_map,V_map),
        beam_extn, beam_npix, beam0_co,beam0_cx, beam90_co, beam90_cx , gpu_dev=1)
    
    ata, atd = update_matrices(  ra.reshape( (1,-1)), 
                                dec.reshape( (1,-1)), 
                                 pa.reshape( (1,-1)) + scan_angle, 
                                 numpy.array( [det_angle] ), 
                                 tod.reshape( (1,-1) ), map_nside ) 
    AtA += ata
    AtD += atd

# Build PISCO convolved maps
I,U,Q,W = matrices_to_maps( map_nside, AtA, AtD )

Q = -2*Q
U =  2*U

Imin = I_smooth.min()
Imax = I_smooth.max()
Qmin = Q_smooth.min()
Qmax = Q_smooth.max()
Umin = U_smooth.min()
Umax = U_smooth.max()

pylab.figure()
mollparams = { 'rot':(ra_s, dec_s) }
#gnomparams = { 'reso':10.0, 'xsize':100, 'ysize':100, 'rot':(ra_s, dec_s) }
healpy.mollview( I            , sub=(3,3,1), title='Ip'  , min=Imin, max=Imax, **mollparams )
healpy.mollview( I_smooth     , sub=(3,3,2), title='Is'  , min=Imin, max=Imax, **mollparams )
healpy.mollview( I_smooth - I , sub=(3,3,3), title='res' , min=Imin/10, max=Imax/10, **mollparams )

healpy.mollview( Q            , sub=(3,3,4), title='Qp'  , min=Qmin, max=Qmax, **mollparams )
healpy.mollview( Q_smooth     , sub=(3,3,5), title='Qs'  , min=Qmin, max=Qmax, **mollparams )
healpy.mollview( Q_smooth - Q , sub=(3,3,6), title='res' , min=Qmin/10, max=Qmax/10, **mollparams )

healpy.mollview( U            , sub=(3,3,7), title='Up'  , min=Umin, max=Umax, **mollparams  )
healpy.mollview( U_smooth     , sub=(3,3,8), title='Us'  , min=Umin, max=Umax, **mollparams )
healpy.mollview( U_smooth - U , sub=(3,3,9), title='res' , min=Umin/10, max=Umax/10, **mollparams )

# Get PS from PISCO maps
TT, EE, BB, TE, EB, TB = healpy.anafast( (I,Q,U), pol=True )

# Get PS from input maps
TT_alm, EE_alm, BB_alm, TE_alm, EB_alm, TB_alm = healpy.anafast(
    (I_map,
     Q_map,
     U_map), pol=True )


# Make some noise
ls = numpy.arange( TT.size )
fig = pylab.figure( figsize=(10,4) )
ax0 = fig.add_subplot( 131 )
ax0.plot( ls, ls*(ls+1)*TT     , label='TT pisco')
ax0.plot( ls, ls*(ls+1)*TT_alm , label='TT alm')
ax0.set_yscale( 'log' )
ax0.set_xlim( 2, 180.0/(beam_fwhm) )
pylab.legend()

ax1 = fig.add_subplot( 132, sharey=ax0 )
ax1.plot( ls, ls*(ls+1)*EE         , label='EE pisco')
ax1.plot( ls, ls*(ls+1)*EE_alm     , label='EE alm')
ax1.set_xlim( 2, 180.0/(beam_fwhm) )
pylab.legend()

ax2 = fig.add_subplot( 133, sharey=ax0 )
ax2.plot( ls, ls*(ls+1)*BB    , label='BB pisco')
ax2.plot( ls, ls*(ls+1)*BB_alm, label='BB_alm')
ax2.set_xlim( 2, 180.0/(beam_fwhm) )

pylab.legend()
pylab.show()

