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
map_nside = 256
npix      = healpy.nside2npix( map_nside )
I_map     = numpy.zeros( npix )
Q_map     = numpy.zeros( npix )
U_map     = numpy.zeros( npix )
V_map     = numpy.zeros( npix )

# Setup a pint source
ra_s  = 0.0
dec_s = 0.0
tht_s = numpy.radians( 90 - dec_s )
phi_s = numpy.radians( ra_s )
v     = healpy.ang2vec( tht_s, phi_s )
source_pixels = healpy.vec2pix( map_nside, v[0], v[1], v[2] )
I_map[ source_pixels ] = 1.0

# Setup a simple Gaussian beam of 10 arcmin

nscan = [256]
resos = [11,101,201,301,401,501]
beam_fwhm   = 1.5
beam_extn   = numpy.radians(7.0) # Extension of 5 degrees
# Get alm space smoothed input maps
I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(beam_fwhm), pol=True )

for nscans in nscan:
    for beam_nside in resos:

        # Beam for feedhorn 1
        beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )
        beam0_cx  = numpy.zeros_like( beam0_co )

        beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )
        beam90_cx = numpy.zeros_like( beam90_co )

        # Setup ring scan
        scan_pixels = healpy.query_disc( map_nside, v, numpy.radians(3.0) )
        theta, phi  = healpy.pix2ang   ( map_nside, scan_pixels )
        ra   = phi
        dec  = numpy.pi/2.0 - theta

        # perform deprojection aliged with the source
        AtA_pa,AtD_pa = 0.0,0.0
        TOD_pa = []
        scan_angles = numpy.linspace( -pi,pi, nscans )
        #scan_angles = numpy.radians( [45,90,-45,-90] )
        det_angle   = 0.0
        for i,scan_angle in enumerate(scan_angles):
            
            pa   = numpy.zeros_like( ra ) + scan_angle
            tod = deproject_sky_for_feedhorn(
                ra, dec, pa,
                det_angle,
                (I_map,Q_map,U_map,V_map),
                beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx, gpu_dev=1 )
            TOD_pa.append( tod )
            arr_pa   = numpy.asarray( [ pa]   )
            arr_ra   = numpy.asarray( [ ra]   )
            arr_dec  = numpy.asarray( [dec]   )
            arr_pol_angles = numpy.asarray( det_angle )
            arr_pisco_tod = numpy.asarray( [tod] )
            ata, atd = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside ) 
            AtA_pa += ata
            AtD_pa += atd

        # Build PISCO convolved maps
        I_pa,Q_pa,U_pa,W_pa = matrices_to_maps( map_nside, AtA_pa, AtD_pa )
        print nscans, beam_nside, numpy.sum( numpy.abs(I_pa - I_smooth) + numpy.abs(Q_pa - Q_smooth) + numpy.abs(U_pa - U_smooth) )/numpy.sum( I_smooth )

    print ''
'''
dec_s = 90.0 - numpy.degrees( tht_s )
ra_s  = numpy.degrees( phi_s )
fig =pylab.figure()
Ip  = healpy.gnomview( I_pa, sub=(3,2,1), rot=(ra_s, dec_s), return_projected_map=True  )
Ih  = healpy.gnomview( I_smooth , sub=(3,2,2), rot=(ra_s, dec_s), return_projected_map=True  )

Qp  = healpy.gnomview( Q_pa, sub=(3,2,3), rot=(ra_s, dec_s),return_projected_map=True   )
Qh  = healpy.gnomview( Q_smooth , sub=(3,2,4), rot=(ra_s, dec_s),return_projected_map=True  )

Up  = healpy.gnomview( U_pa , sub=(3,2,5), rot=(ra_s, dec_s),return_projected_map=True  )
Uh  = healpy.gnomview( U_smooth , sub=(3,2,6), rot=(ra_s, dec_s),return_projected_map=True  )

mmin = Ih.min()
mmax = Ih.max()

import matplotlib as mpl
fig, axes = pylab.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

plots  = [Ip,Qp,Up]
titles = ['PISCO Stokes I','PISCO Stokes Q', 'PISCO Stokes U']
for ax,plot,title in zip(axes.flat[0:3],plots,titles):
    ax.set_title( title )
    im = ax.imshow( plot, vmin=mmin, vmax=mmax, extent=(-beam_extn/2,beam_extn/2,-beam_extn/2,beam_extn/2)  )
    ax.set_xlabel( 'radians' )
    ax.set_ylabel( 'radians' )

cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat[0:3]])
pylab.colorbar(im, cax=cax, **kw)

plots  = [Ip-Ih,Qp-Qh,Up-Uh]
titles = ['PISCO - l-space (I)','PISCO - l-space (Q)','PISCO - l-space (U)']
for ax,plot,title in zip(axes.flat[3:6],plots,titles):
    ax.set_title( title )
    im = ax.imshow( plot, vmin=-mmax/100.0, vmax=mmax/100.0, extent=(-beam_extn/2,beam_extn/2,-beam_extn/2,beam_extn/2) )
    ax.set_xlabel( 'radians' )
    ax.set_ylabel( 'radians' )

cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat[3:6]])
pylab.colorbar(im, cax=cax, **kw)

pylab.show()
'''
