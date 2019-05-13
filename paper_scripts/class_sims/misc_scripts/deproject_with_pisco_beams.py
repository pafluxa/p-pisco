#!/usr/bin/env python
# coding: utf-8
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import sys

import pisco
from pisco.beam_analysis.utils import *
from pisco.convolution.core import deproject_sky_for_feedhorn
from pisco.mapping.core import *
from pisco.pointing import *
from pisco.pointing.core import *
from pisco.tod import *

import astropy.units as u
from astropy.coordinates import EarthLocation

from cmb_analysis.powerspectrum import pyspice

import pandas
import time
import numpy
import healpy
import pylab

# Load maps from input
maps = numpy.load( sys.argv[1] )
map_nside = maps['nside'][()]
I_map     = maps['I'] 
Q_map     = maps['Q'] 
U_map     = maps['U'] 
V_map     = maps['V']*0



#----------------------------------------------------------------------------------------------------------#
# Read in focal plane
#----------------------------------------------------------------------------------------------------------#
focalPlane = pandas.read_csv( './data/array_data/piscotel.csv' )
#----------------------------------------------------------------------------------------------------------#
receiver = Receiver()
receiver.initialize(  focalPlane['uid'],
                     -focalPlane['az_off'],
                      focalPlane['el_off'],
                      numpy.radians( focalPlane['rot'] ) )
#----------------------------------------------------------------------------------------------------------#
# Read beam parameter file
uid, theta, fwhm_x, fwhm_y = numpy.loadtxt( './data/array_data/piscotel_beam_params.txt', unpack=True )

grid_side = numpy.radians( fwhm_x * 5 )
nx = 1001

beam_co   = make_gaussian_elliptical_beam( fwhm_x, fwhm_y, grid_side, nx, theta=theta, amplitude=1.0 )
beam_cx   = numpy.zeros_like( beam_co )
beam = { 'co': beam_co , 'cx': beam_cx*0.0, 'nx': nx, 'grid_side':grid_side }

# Setup map matrices
AtA = 0.0
AtD = 0.0

# Setup mock scan
tht, phi = healpy.pix2ang( map_nside, numpy.arange( healpy.nside2npix( map_nside ) ) )
ra  = phi
dec = numpy.pi/2.0 - tht
pa  = numpy.zeros_like( ra )

# Get beam
grid_nx   = beam['nx']
grid_side = beam['grid_side']
beam1_co,beam1_cx = numpy.copy(beam['co']), numpy.copy(beam['cx'])
beam2_co,beam2_cx = numpy.copy(beam['co']), numpy.copy(beam['cx'])

output_map_nside = map_nside
for boresight_rotation in numpy.radians( numpy.linspace(-180,180,4) ):
    
    print numpy.degrees( boresight_rotation )

    tod  = deproject_sky_for_feedhorn(
        ra, dec, pa + boresight_rotation,
        0.0,
        (I_map,Q_map,U_map,V_map),
        grid_side, grid_nx,beam1_co,beam1_cx, beam2_co, beam2_cx,
        gpu_dev=0 )

    ata, atd = update_matrices(
                     ra.reshape((1,-1)),
                     dec.reshape((1,-1)),
                     pa.reshape((1,-1)) + boresight_rotation,
                     receiver.pol_angles,
                     tod.reshape((1,-1)),
                     output_map_nside )
    AtA += ata
    AtD += atd
    
    numpy.savez( './runs/matrices_fwhmx_1d0_fwhmy_1d0_input_lcdmr0d01_id_0000.npz', AtA=AtA, AtD=AtD, nside=output_map_nside )
'''
I,Q,U,W = matrices_to_maps( map_nside, AtA, AtD )

# Smooth input maps by the given resolution
I_s,Q_s,U_s = healpy.smoothing( (I_map,Q_map,U_map), fwhm=numpy.radians(fwhm_x), pol=True )
healpy.mollview( I - I_s , min=I.min()/1, max=I.max()/1 )
healpy.mollview( Q - Q_s , min=Q.min()/1, max=Q.max()/1)
healpy.mollview( U - U_s , min=U.min()/1, max=U.max()/1)
pylab.show()
# Calibrate to relative I
cal = numpy.dot( I, I_s )/numpy.dot( I,I )
#I = cal * I
#Q = cal * Q
#U = cal * U
print cal

# Set lmax=150
lmax = 350
# Get PS from PISCO maps
TT, EE, BB, TE, EB, TB = healpy.anafast( (I,Q,U), pol=True, lmax=lmax )

# Get PS from input maps
TT_alm, EE_alm, BB_alm, TE_alm, EB_alm, TB_alm = healpy.anafast(
    (I_map,
     Q_map,
     U_map), pol=True , lmax=lmax )

# Obtain window function as the transform of the beam
tp,pp = healpy.pix2ang( map_nside, numpy.arange( healpy.nside2npix( map_nside ) ) )
sigma = numpy.radians( fwhm_x/2.3486 )
beam  = numpy.exp (-0.5*tp**2/sigma**2 )
beam_cl = ( healpy.anafast( beam, pol=False, lmax=lmax ) )

# Get l numbers
ls = numpy.arange( TT.size )

# Normalize window function
beam_cl /= beam_cl.max()

# Create Gaussian window to deconvolve it from PISCO
gwin_TT, gwin_EE, gwin_BB, gwin_TE = healpy.gauss_beam( numpy.radians(fwhm_x),lmax=lmax, pol=True ).T
gwin_TT = gwin_TT**2
gwin_EE = gwin_EE**2
gwin_BB = gwin_BB**2

pylab.plot( beam_cl )
pylab.plot( gwin_TT )
pylab.show()
# Get pixel window function of the map
bl_temp, bl_pol = healpy.pixwin( map_nside, pol=True )
bl_temp = ( bl_temp**2)[0:lmax+1]
bl_pol  = ( bl_pol **2)[0:lmax+1]

# Read input CLs
ls_in,cl_TT_in,cl_EE_in,cl_BB_in,cl_TE_in = numpy.loadtxt( sys.argv[2] )

cl_TT = cl_TT_in[0:lmax+1]
cl_EE = cl_EE_in[0:lmax+1]
cl_BB = cl_BB_in[0:lmax+1]

# Make some noise
fig = pylab.figure( figsize=(6,6) )
ax0 = fig.add_subplot( 111 )
ax0.plot( ls, ls*(ls+1)*TT/(gwin_TT*bl_temp), label='TT pisco')
#ax0.plot( ls, ls*(ls+1)*TT/(beam_cl), label='TT pisco')
ax0.plot( ls, ls*(ls+1)*TT_alm, label='TT')

ax0.plot( ls, ls*(ls+1)*EE/(gwin_EE*bl_pol) , label='EE pisco')
#ax0.plot( ls, ls*(ls+1)*EE/(beam_cl) , label='EE pisco')
ax0.plot( ls, ls*(ls+1)*EE_alm , label='EE')

ax0.plot( ls, ls*(ls+1)*BB/(gwin_BB*bl_pol), label='BB pisco')
#ax0.plot( ls, ls*(ls+1)*BB/(beam_cl), label='BB pisco')
ax0.plot( ls, ls*(ls+1)*BB_alm , label='BB')

ax0.plot( ls, cl_TT, label='TT input')
ax0.plot( ls, cl_EE, label='EE input')
ax0.plot( ls, cl_BB, label='BB input')

pylab.legend()
pylab.yscale( 'log' )
pylab.xscale( 'log' )

pylab.xlabel( r'$l$' )
pylab.ylabel( r'K$^2$' )

pylab.show()
#pylab.savefig( 'maps.pdf' )
'''
