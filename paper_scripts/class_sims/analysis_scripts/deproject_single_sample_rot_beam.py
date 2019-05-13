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
                                                                                                              
I_map   = map_data['Q']                                                                                       
Q_map   = map_data['Q']                                                                                       
U_map   = map_data['U']                                                                                       
V_map   = map_data['V']

# Setup a pure Q source
source_pixels = 400000 

# Setup a simple Gaussian beam of 10 arcmin                                                                   
beam_nside  = 31                                                                                              
beam_fwhm   = 2.0                                                                                             
beam_extn   = numpy.radians(10.0) # Extension of 5 degrees                                                    
                                                                                                              
# Beam for feedhorn 1                                                                                         
beam0_co  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )                 
beam0_cx  = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.00)                  
                                                                                                              
beam90_co = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=1.00 )                 
beam90_cx = make_gaussian_beam( beam_fwhm, beam_fwhm, beam_extn, beam_nside, amplitude=0.00 )  

# Setup ring scan
theta, phi  = healpy.pix2ang( map_nside, source_pixels) 
ra   = numpy.array( (phi,phi) )
dec  = numpy.array( (pi/2.0 - theta, pi/2.0-theta) )

AtA_pa,AtD_pa = 0.0,0.0                                                                                       
TOD_pa = []                                                                                                   
scan_angles = [0,pi/4.0, pi/2.0]                                                                              
det_angles  = [0,0,0]                                                                                         
for scan_angle, det_angle in zip( scan_angles, det_angles ):                                                  
                                                                                                                  
    pa   = numpy.zeros_like( ra ) + scan_angle                                                                
    
    tod = deproject_sky_for_feedhorn(                                                                         
        ra, dec, pa,                                                                                          
        det_angle,                                                                                            
        (I_map,Q_map,U_map,V_map),                                                                            
        beam_extn, beam_nside, beam0_co,beam0_cx, beam90_co, beam90_cx )                                      
    
    TOD_pa.append( tod )                                                                                      
    arr_pa   = numpy.asarray( [ pa]   )                                                                       
    arr_ra   = numpy.asarray( [ ra]   )                                                                       
    arr_dec  = numpy.asarray( [dec]   )                                                                       
    arr_pol_angles = numpy.asarray( [ det_angle ] )                                                       
    arr_pisco_tod = numpy.asarray( [tod] )                                                                    
    ata, atd = update_matrices( arr_ra, arr_dec, arr_pa, arr_pol_angles, arr_pisco_tod, map_nside )           
    AtA_pa += ata                                                                                             
    AtD_pa += atd
