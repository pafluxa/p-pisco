# coding: utf-8
import sys
import os

import numpy
from   numpy import pi
import numpy as np

import pandas

import pylab
import matplotlib.pyplot as plt

import healpy
import healpy as hp

from pisco.mapping.core import matrices_to_maps

from composite_elliptical_beam_window_function import make_composite_elliptical_beam_window_function

from config_parser import parse_config_file

import os

# PolSpice wrapper from Duncan Watts (2019)
#
def spice_wrap2(m1, m2, weights=None, lmax=300):

    clfile = 'temp_cls.txt'
    mapfile1 = 'temp_m1.fits'
    mapfile2 = 'temp_m2.fits'
    weightfile = 'temp_mask.fits'
    
    if weights is None:
        weights = numpy.ones_like( m1[0] ) 
        weights /= numpy.sum(weights)

    hp.write_map(mapfile1, m1, overwrite=True)
    hp.write_map(mapfile2, m2, overwrite=True)
    hp.write_map(weightfile, weights, overwrite=True)
    
    print "running polspice..."

    command = [
        '/home/pafluxa/software/PolSpice_v03-05-01/build/spice',
        '-mapfile', mapfile1,
        '-mapfile2', mapfile2,
        '-verbosity', '0',
        '-weightfile', weightfile,
        '-nlmax', str(lmax),
        '-overwrite', 'NO',
        '-polarization', 'YES',
        '-decouple', 'YES',
        '-symmetric_cl', 'YES', # averages T_map1*E_map2 and T_map2*E_map1, etc.
        '-clfile', clfile,
        '-tolerance', '1e-7'
        ]

    os.system(' '.join(command))
    ell, TT, EE, BB, TE, TB, EB = np.loadtxt(clfile,unpack=True)
    os.system('rm temp*')

    print "done."

    return ell,TT,EE,BB,TE,TB,EB

# Across the code, there is o<variable name>, u<variable name> and p<variable name>.
# The meaning is
#
# o : original, no smoothing
# p : PISCO
#

# read in configuration file
config = parse_config_file( sys.argv[1] )

# read beam parameters
fwhm_x   = config['beams']['fwhmX']                                                                           
fwhm_y   = config['beams']['fwhmY']                                                                           

# read output file path
output_path = config['outputFile']

# build maps from PISCO generated TOD
data = numpy.load( output_path )
print "mapping..."
AtA,AtD,NSIDE = data['AtA'], data['AtD'], data['nside'][()]
pI,pQ,pU,pW = matrices_to_maps( NSIDE, AtA, AtD )
print "done."

# Read original maps from disk
maps      = numpy.load( config['inputMapPath'] )
oI        = maps['I']
oQ        = maps['Q']
oU        = maps['U']
map_nside = maps['nside'][()]

lmax = 250

# scale maps to uK
pI *= 1e6
pQ *= 1e6
pU *= 1e6

oI *= 1e6
oQ *= 1e6
oU *= 1e6

# Define l numbers                                                                                            
ell   = numpy.arange( 1, lmax + 1 )                                                                                
#dl2cl = 1./( (2*numpy.pi)**2 /( ell*(ell+1) ) )

# Get window function considering all beams in the focal plane                                                
#print fwhm_x, fwhm_y
#beam_fwhm = 1.5
'''
w, beam_fwhm = make_composite_elliptical_beam_window_function( fwhm_x, fwhm_y, lmax, pol=True )               
glTT_mkb, glEE_mkb, glBB_mkb, glTE_mkb = w                                                                    
wl_TT_mkb = (glTT_mkb**2)                                                                    
wl_EE_mkb = (glEE_mkb**2)                                                                    
wl_BB_mkb = (glBB_mkb**2)                                                                    
'''

beam_fwhm = 1.5
# Get window function of equivalent Gaussian beam
glTT, glEE, glBB, glTE = healpy.sphtfunc.gauss_beam( numpy.radians(beam_fwhm), pol=True, lmax=lmax ).T
wl_TT = (glTT**2)[1:lmax+1]
wl_EE = (glEE**2)[1:lmax+1]
wl_BB = (glBB**2)[1:lmax+1]

#print( "smoothing..." )
# Convolve original maps using smoothing
#sI, sQ, sU = healpy.smoothing( (oI, oQ, oU), fwhm=numpy.radians(beam_fwhm), pol=True )
#print( "done.")
w = healpy.read_map( './data/masks/CLASS_coverage_mask.fits' )
wUp = healpy.ud_grade( w, 256 )
#_,sTT,sEE,sBB,sTE,sTB,sEB = spice_wrap2( (sI,sQ,sU), (sI,sQ,sU), weights=w, lmax=lmax )
_,pTT, pEE, pBB, _,_,_ = spice_wrap2( (pI,pQ,pU), (pI,pQ,pU), weights=None, lmax=lmax ) 
_,oTT, oEE, oBB, _,_,_ = spice_wrap2( (oI,oQ,oU), (oI,oQ,oU), weights=None, lmax=lmax ) 
#print "running anafast"
#oTT, oEE, oBB, _, _, _ = healpy.anafast( (oI,oQ,oU), pol=True, alm=False, iter=3 )          
#pTT, pEE, pBB, _, _, _ = healpy.anafast( (pI,pQ,pU), pol=True, alm=False, iter=3 )          
#_,pTT,pEE,pBB,pTE,pTB,pEB = spice_wrap2( (pI,pQ,pU), (pI,pQ,pU), weights=w, lmax=lmax )


# Adjust to the lmax parameter, because Spice doesn't have that option
oTT = oTT[1:lmax+1]
oEE = oEE[1:lmax+1]
oBB = oBB[1:lmax+1]

pTT = pTT[1:lmax+1] / wl_TT
pEE = pEE[1:lmax+1] / wl_EE
pBB = pBB[1:lmax+1] / wl_BB

fig  = pylab.figure( figsize=(20,5) )                                                                         
                                                                                                              
ax_TT = fig.add_subplot( 131 )                                                                                
ax_TT.set_title( r'TT Power Spectra' )                                                                        
ax_TT.set_xlabel( r'$\ell$' )                                                                                 
ax_TT.set_ylabel( r'uK$^2$' )                                                                                  
ax_TT.set_ylim( (100,3000) )                                                                              
ax_TT.set_xlim( (2,lmax) )                                                                                    
                                                                                                              
ax_EE = fig.add_subplot( 132 )                                                                                
ax_EE.set_title( r'EE Power Spectra' )                                                                        
ax_EE.set_xlabel( r'$\ell$' )                                                                                 
ax_EE.set_xlim( (2,lmax) )                                                                                    
ax_EE.set_ylim( (-1e-3,2) )                                                                              
ax_EE.set_yscale('symlog', linthreshy=5e-1)

ax_BB = fig.add_subplot( 133 )                                                                                
ax_BB.set_title( r'BB Power Spectra' )                                                                        
ax_BB.set_xlabel( r'$\ell$' )                                                                                 
ax_BB.set_xlim( (2,lmax) )                                                                                    
ax_BB.set_ylim( (-1e-3,2) )                                                                              
ax_BB.set_yscale('symlog', linthreshy=1e-2)                                                                  
                                                                                                              
ell2 = ell*(ell+1)/(2*numpy.pi)
# Plot C_\ell of the original maps                                                                          
ax_TT.plot( ell2*pTT, c='red', alpha=0.6,                                                        
            label='pisco' )                              
ax_TT.plot( ell2*oTT, c='black', alpha=0.8, linestyle='--',
            label='input' )                              

ax_EE.plot( ell2*pEE, c='red', alpha=0.6,                                                        
            label='pisco' )                              
ax_EE.plot( ell2*oEE, c='black', alpha=0.8, linestyle='--',
            label='input' )                              

ax_BB.plot( ell2*pBB, c='red', alpha=0.6,                                                        
            label='pisco' )                              
ax_BB.plot( ell2*oBB, c='black', alpha=0.8, linestyle='--',
            label='input' )                              

ax_TT.legend()                                                                                                
ax_EE.legend()                                                                                                
ax_BB.legend()                                                                                                
                                                                                                              
pylab.show()     
