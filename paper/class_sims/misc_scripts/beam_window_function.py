#!/usr/bin/env python
# coding: utf-8
import array_split
from array_split import array_split, shape_split

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
from pisco.beam_analysis.mueller import ComplexMuellerMatrix as CM

import astropy.units as u
from astropy.coordinates import EarthLocation

import os

import pandas
import time
import numpy
from numpy import pi
import healpy
import pylab

# Do this shit right with argparser
import argparse
parser = argparse.ArgumentParser(description='Simulate CLASS maps using provided pointing.')
parser.add_argument( '-tag' , action='store', type=str, dest='tag',
                      help='Name of the output. Output will be matrices_tag.npz' )
parser.add_argument( '-beam_par_file',  action='store', type=str, dest='beam_par_file',
                      help='File with specifications of the beam parameters for each detector.')
parser.add_argument( '-beams',  action='store', type=str, dest='beams_path',
                      help='Path to detector beams in HEAPIX format, packed as npz files.' )
args = parser.parse_args()

# Setup simulation tag
tag = args.tag

#----------------------------------------------------------------------------------------------------------#
# Read beam parameter file
print 'reading beam parameters'
beam_data = pandas.read_csv( args.beam_par_file )
uids      = beam_data[ 'uid']
feeds     = beam_data[ 'feed']
fwhm_x    = beam_data[ 'fwhm_x']
fwhm_y    = beam_data[ 'fwhm_y']
rotation  = beam_data[ 'rot']
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Assemble beams in the focal plane
print 'assembling focal plane beams'
mueller_beams      = [None] * 88
for feed in feeds:
    
    print feed

    # Find the uids corresponding to the feed
    uid1, uid2 = beam_data['uid'][ beam_data['feed'] == feed ]
    print uid1, uid2

    print 'loading', os.path.join( args.beams_path ,'detector%d.npz' % (uid1) )
    data = numpy.load( os.path.join( args.beams_path ,'detector%d.npz' % (uid1) ) )
    beam_nside = data['nside'][()]
    Ex_co, Ex_cx = data['E_co'], data['E_cx']
    
    print 'loading', os.path.join( args.beams_path ,'detector%d.npz' % (uid2) )
    data = numpy.load( os.path.join( args.beams_path ,'detector%d.npz' % (uid2) ) )
    beam_nside = data['nside'][()]
    Ey_co, Ey_cx = data['E_co'], data['E_cx']
    
    M = CM.make_optical_mueller_matrix( beam_nside, Ex_co, Ex_cx, Ey_co, Ey_cx )
    mueller_beams[ feed ] = { 'I': M.M_TT , 'Q': M.M_QQ, 'U' : M.M_UU, 'nside': beam_nside }
    
    healpy.orthview( M.M_TT.real, half_sky=True, rot=(0,90), sub=(3,3,1), title='M$_{II}$' ) 
    healpy.orthview( M.M_TQ.real, half_sky=True, rot=(0,90), sub=(3,3,2), title='M$_{IQ}$' ) 
    healpy.orthview( M.M_TU.real, half_sky=True, rot=(0,90), sub=(3,3,3), title='M$_{IU}$' ) 
    
    healpy.orthview( M.M_QT.real, half_sky=True, rot=(0,90), sub=(3,3,4), title='M$_{QT}$' ) 
    healpy.orthview( M.M_QQ.real, half_sky=True, rot=(0,90), sub=(3,3,5), title='M$_{QQ}$' ) 
    healpy.orthview( M.M_QU.real, half_sky=True, rot=(0,90), sub=(3,3,6), title='M$_{QU}$' ) 
    
    healpy.orthview( M.M_UT.real, half_sky=True, rot=(0,90), sub=(3,3,7), title='M$_{UT}$' ) 
    healpy.orthview( M.M_UQ.real, half_sky=True, rot=(0,90), sub=(3,3,8), title='M$_{UQ}$' ) 
    healpy.orthview( M.M_UU.real, half_sky=True, rot=(0,90), sub=(3,3,9), title='M$_{UU}$' ) 
    
    pylab.show()

#----------------------------------------------------------------------------------------------------------#

# Stack beams
stacked_beam = 0.0
for uid in feed:
    
    stacked_beam += ( beams[ uid ]['co']/beams[ uid ]['co'].max() )**2

healpy.orthview( 20*numpy.log10(stacked_beam+0.00001),
    rot=(0,90),
    title='Stacked beam',
    sub=(1,1,1),
    min=-40, max=40,
    half_sky=True,
    unit='dBi' )
pylab.show()

# Compute power spectra of the beam
lmax     = 250
#stacked_beam = make_gaussian_beam( 512, 1.5 )**2
TT_beam   = healpy.anafast( stacked_beam, lmax=lmax )
TT_gauss1 = healpy.gauss_beam( numpy.radians(1.5), lmax=lmax )
TT_gauss2 = healpy.gauss_beam( numpy.radians(1.6), lmax=lmax )
TT_gauss3 = healpy.gauss_beam( numpy.radians(1.7), lmax=lmax )

pylab.plot( (TT_beam/TT_beam[0]) )
pylab.plot(  TT_gauss1**2 )
pylab.plot(  TT_gauss2**2 )
pylab.plot(  TT_gauss3**2 )
pylab.show()

#numpy.savez( 'stacked_beams.npz', beam=stacked_beam, nside=512, window_function=(TT_beam/TT_beam[0]) )
