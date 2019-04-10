#!/usr/bin/env python
# coding: utf-8
import pisco
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.beam_analysis.utils import *
from pisco.mapping.core import *

import sys
import time
import numpy
from numpy import pi
import healpy
import pylab

map_nside = 256
npix      = healpy.nside2npix( map_nside )
I_map     = numpy.zeros( npix )
Q_map     = numpy.zeros( npix )
U_map     = numpy.zeros( npix )
V_map     = numpy.zeros( npix )

# Setup a pint source
ra_s  =  0.0
dec_s = 45.0

tht_s   = numpy.radians( 90 - dec_s )
phi_s   = numpy.radians( ra_s )
v       = healpy.ang2vec( tht_s, phi_s )
spixels = healpy.vec2pix( map_nside, v[0], v[1], v[2] )

I_map[ spixels ] = 1.0
Q_map[ spixels ] = 1.0
U_map[ spixels ] = 0.0

# Setup a simple Gaussian beam of 1.5 degrees FWHM
beam_nside  =  512
beam_fwhm   =  1.0

# Beam for feedhorn 1
beam0_co  = make_gaussian_elliptical_beam( beam_nside, beam_fwhm, beam_fwhm, 0 )
beam0_cx  = numpy.zeros_like( beam0_co )

# Beam for feedhorn 2
beam90_co = make_gaussian_elliptical_beam( beam_nside, beam_fwhm, beam_fwhm, 0 )
beam90_cx = numpy.zeros_like( beam90_co )

# Get alm space smoothed input maps
I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(beam_fwhm), pol=True )

# Setup scan
nscans      = 3
scan_pixels = healpy.query_disc( map_nside, v, numpy.radians(5.0) )
theta, phi  = healpy.pix2ang   ( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta
pa   = numpy.zeros_like( ra )

# perform deprojection aliged with the source
AtA,AtD   = 0.0,0.0
pol_angle = 0.0

scan_angles = numpy.linspace( -45,45, nscans )
for i,scan_angle in enumerate(scan_angles):
    
    print i+1
    
    scan_angle = numpy.deg2rad( scan_angle )

    tod = deproject_sky_for_feedhorn(
        ra, dec, pa + scan_angle,
        pol_angle,
        (I_map,Q_map,U_map,V_map),
        beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx, gpu_dev=2 )
    
    arr_pa   = numpy.asarray( [ pa]   ) + scan_angle
    arr_ra   = numpy.asarray( [ ra]   )
    arr_dec  = numpy.asarray( [dec]   )

    arr_pol_angles = numpy.asarray( pol_angle )
    arr_pisco_tod  = numpy.asarray( [tod] )
    
    ata, atd = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside ) 
    
    AtA += ata
    AtD += atd

# Build PISCO convolved maps
I,Q,U,W = matrices_to_maps( map_nside, AtA_pa, AtD_pa )

# Visualize using Gnomview. Center map at the source
dec_s = 90.0 - numpy.degrees( tht_s )
ra_s  =        numpy.degrees( phi_s )

fig =pylab.figure()
Ip  = healpy.gnomview( I,     sub=(2,3,1), rot=(ra_s, dec_s), return_projected_map=True , title='PISCO rec. I', notext=True )
Qp  = healpy.gnomview( Q,     sub=(2,3,2), rot=(ra_s, dec_s),return_projected_map=True , title='PISCO rec. Q', notext=True  )
Up  = healpy.gnomview( U,     sub=(2,3,3), rot=(ra_s, dec_s),return_projected_map=True , title='PISCO rec. U', notext=True )
Ih  = healpy.gnomview( I_smooth, sub=(2,3,4), rot=(ra_s, dec_s), return_projected_map=True, title='True convolution I', notext=True  )
Qh  = healpy.gnomview( Q_smooth, sub=(2,3,5), rot=(ra_s, dec_s),return_projected_map=True, title='True convolution Q', notext=True  )
Uh  = healpy.gnomview( U_smooth, sub=(2,3,6), rot=(ra_s, dec_s),return_projected_map=True , title='True convolution U', notext=True )
pylab.show()

'''
This code doesn't work well. 

import matplotlib as mpl
fig, axes = pylab.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
plots  = [Ip,Qp,Up]
titles = ['I_p','Q_p', 'U_p']
map_extn = 5.0
for m,ax,plot,title in zip(['I','Q','U'],axes.flat[0:3],plots,titles):
    
    if m == 'I':
        mmin = numpy.min( Ih )
        mmax = numpy.max( Ih )
    if m == 'Q':
        mmin = numpy.min( Qh )
        mmax = numpy.max( Qh )
    if m == 'U':
        mmin = numpy.min( Uh )
        mmax = numpy.max( Uh )

    ax.set_title( title )
    im = ax.imshow( plot, vmin=mmin, vmax=mmax, extent=(-map_extn/2,map_extn/2,-map_extn/2,map_extn/2)  )
    ax.set_xlabel( 'degrees' )
    ax.set_ylabel( 'degrees' )

cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat[0:3]])
pylab.colorbar(im, cax=cax, **kw)

plots  = [(Ip-Ih)/numpy.max(Ih),(Qp-Qh)/numpy.max(Qh),(Up-Uh)/numpy.max(Uh) ]
titles = ['I_p - I_s','Q_p - Q_s','U_p - U_s']
for m,ax,plot,title in zip(['I','Q','U'],axes.flat[3:6],plots,titles):
    
    if m == 'I':
        mmin = numpy.min( Ih )
        mmax = numpy.max( Ih )
    if m == 'Q':
        mmin = numpy.min( Qh )
        mmax = numpy.max( Qh )
    if m == 'U':
        mmin = numpy.min( Uh )
        mmax = numpy.max( Uh )
    
    ax.set_title( title )
    im = ax.imshow( plot, vmin=-0.01, vmax=0.01, extent=(-map_extn/2,map_extn/2,-map_extn/2,map_extn/2) )
    ax.set_xlabel( 'degrees' )
    ax.set_ylabel( 'degrees' )

cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat[3:6]])
pylab.colorbar(im, cax=cax, **kw)
pylab.show()
'''
