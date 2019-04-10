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
map_data  = numpy.load( 'IQUV_maps_0000.npz' )
map_nside = map_data['nside'][()]
I_map   = map_data['I']
Q_map   = map_data['Q'] 
U_map   = map_data['U']
V_map   = map_data['V']


# Setup a simple Gaussian beam 
beam_nside  = 1024
beam_fwhm   =  1.5
beam_extn   = 10.0
# Beam for feedhorn 1
beam1_co = 1.0*numpy.sqrt( make_gaussian_beam( beam_nside, beam_fwhm, max_theta=beam_extn ) )
beam1_cx = 0.0*make_gaussian_crosspolar_beam( beam_nside, beam_fwhm, max_theta=beam_extn )

# Beam for feedhorn 2
beam2_co = 1.0*numpy.sqrt( make_gaussian_beam( beam_nside, beam_fwhm, max_theta=beam_extn ) )
beam2_cx = 0.0*make_gaussian_crosspolar_beam( beam_nside, beam_fwhm, max_theta=beam_extn )

beam1_co = beam1_co/(beam1_co.sum()) 
beam2_co = beam2_co/(beam2_co.sum()) 

# Smooth input map using alm-space convolution
I_smooth,Q_smooth,U_smooth = healpy.smoothing( (I_map,Q_map,U_map), pol=True, fwhm=numpy.radians( beam_fwhm ) )

# Setup ring scan
scan_pixels = numpy.arange( healpy.nside2npix( map_nside ) )
theta, phi  = healpy.pix2ang( map_nside, scan_pixels )
ra   = phi
dec  = numpy.pi/2.0 - theta
pa   = numpy.zeros_like( ra )

# Parse coordinates and TOD for mapping
arr_ra   = numpy.asarray( [ ra]   )                                                                           
arr_dec  = numpy.asarray( [dec]   )                                                                           
arr_pa   = numpy.asarray( [ pa]   )

# Run signle sample deprojection
det_pol_angles = numpy.asarray( [0.0] )
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, (beam1_co,beam1_cx), (beam2_co, beam2_cx),
    gpu_dev=0 )

arr_pol_angles = numpy.asarray( [ det_pol_angles ] )
arr_pisco_tod = numpy.asarray( [pisco_tod] )
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside )

det_pol_angles = numpy.asarray( [pi/4.0] )
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, (beam1_co,beam1_cx), (beam2_co, beam2_cx),
    gpu_dev=0 )

arr_pol_angles = numpy.asarray( [ det_pol_angles ] )
arr_pisco_tod = numpy.asarray( [pisco_tod] )
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside, AtA=AtA, AtD=AtD )

det_pol_angles = numpy.asarray( [-pi/4.0] )
pisco_tod = deproject_sky_for_feedhorn(
    ra, dec, pa,
    det_pol_angles[0],
    (I_map,Q_map,U_map,V_map),
    beam_nside, (beam1_co,beam1_cx), (beam2_co, beam2_cx),
    gpu_dev=0 )

arr_pol_angles = numpy.asarray( [ det_pol_angles ] )
arr_pisco_tod = numpy.asarray( [pisco_tod] )
AtA,AtD = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside, AtA=AtA, AtD=AtD )

# Build PISCO convolved maps
I,Q,U,W = matrices_to_maps( map_nside, AtA, AtD )  

c1 = numpy.dot( I, I_smooth )/numpy.dot( I, I )
c2 = numpy.dot( Q, Q_smooth )/numpy.dot( Q, Q )
c3 = numpy.dot( U, U_smooth )/numpy.dot( U, U )

# Get PS from PISCO maps
TT, EE, BB, TE, EB, TB = healpy.anafast( (I*c1,Q*c2,U*c3), pol=True )

# Get PS from input maps
TT_diff, EE_diff, BB_diff, TE_diff, EB_diff, TB_diff = healpy.anafast( 
    (I_smooth,
     Q_smooth,
     U_smooth), pol=True )

# Make some noise
ls = numpy.arange( TT.size )
fig = pylab.figure( figsize=(10,4) )
ax0 = fig.add_subplot( 131 )
ax0.plot( ls, TT , label='TT pisco')
ax0.plot( ls, TT_diff, label='TT_diff' )
ax0.set_yscale( 'log' )

ax1 = fig.add_subplot( 132, sharey=ax0 )
ax1.plot( ls, EE , label='EE pisco')
ax1.plot( ls, EE_diff , label='EE diff')

ax2 = fig.add_subplot( 133, sharey=ax0 )
ax2.plot( ls, BB , label='BB pisco')
ax2.plot( ls, BB_diff , label='BB diff')

pylab.legend()

pylab.show()
'''
# Sample the same coordinates as PISCO
alm_tod = 0.5*healpy.get_interp_val( I_smooth, theta, phi )

pylab.scatter( pisco_tod, alm_tod, s=0.1, alpha=0.4 )
pylab.show()
'''



