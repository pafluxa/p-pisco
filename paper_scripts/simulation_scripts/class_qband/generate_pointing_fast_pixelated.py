#!/usr/bin/env python
# coding: utf-8
__author__="Michael K. Brewer, May 2019."

import numpy
import argparse

import os
from os.path import join as join_paths

import time
from datetime import datetime

from config_parser import parse_config_file

import healpy

# for pointing calculations
from pypoint.core import focalplane_to_equatorial

# Create a "menu" for command line interfacing
#-----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Simulate CLASS Q-band pointing for a featureless sky.\
The scan is composed of constant elevation (45 degrees) azimuth scans." )

# number of days
parser.add_argument(
    "-days", action="store", type=int, dest="days", default=1,
    help="The simulated pointing will encompass the specified number of days. Defaults to 1." )

# sampling frequency
parser.add_argument(
    "-sps", action="store", type=int, dest="sps", default=10,
    help="Generated pointing will be generated at `sps` samples per second. Defaults to 10." )

# sampling frequency
parser.add_argument(
    "-nside", action="store", type=int, dest="nside",
    help="Generated pointing will be pixelated to an `nside` healpix grid." )

parser.add_argument(
    "-latitude", action="store", type=float, dest="latitude",
    default=-22.959748,
    help="Latitude of the experiment in degrees and decimal format. Defaults to CLASS latitude." )

parser.add_argument(
    "-config", action="store", type=str, dest="configFile",
    help="Path to configuration file." )


# TODO: add a parameter to let the ser choose between generating pointing in ICRS coordinates or not.
# Parse arguments
args = parser.parse_args()
#-----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------#
# Setup simulation tag                                                                                        
#----------------------------------------------------------------------------------------------------------#
config = parse_config_file( args.configFile )  
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Read output File
output_file = config['pointingFile']                                                                            
print output_file  
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Read beam parameter file
#----------------------------------------------------------------------------------------------------------#
uids   = config['focalplane']['uids']                                                                         
feeds  = config['focalplane']['feeds']                                                                        
azOff  = config['focalplane']['azOff']                                                                        
elOff  = config['focalplane']['elOff']                                                                        
detPol = config['focalplane']['detPol']                                                                       
isOn   = config['focalplane']['on']                                                                           
isOff  = (isOn + 1) % 2   
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# Setup experiment location
lat = numpy.radians( args.latitude )
#----------------------------------------------------------------------------------------------------------#

ts  = 0
te  = ts + args.days * 86400.0
sps = args.sps
T   = 720

# Compute the number of scans as the duration of a scan
nscans = int( numpy.ceil( (te - ts)/T ) )
ndays  = args.days

# Define the TOD
ctime     = numpy.arange( ts, te, 1./sps )
# HARD CODED: constant elevation scan at 45 degrees.
elevation = numpy.zeros_like( ctime ) + numpy.pi/4.0
azimuth   = numpy.zeros_like( ctime )
rotation  = numpy.zeros_like( ctime )

for scan in range( nscans ):

    if scan%2 == 0:
        azimuth[ scan*T*sps: (scan+1)*T*sps ] = numpy.radians( numpy.linspace( -180, 540, (T*sps) ) )
    else:
        azimuth[ scan*T*sps: (scan+1)*T*sps ] = numpy.radians( numpy.linspace(  540,-180, (T*sps) ) )

boresights = [-45,-30,-15, 0, 15, 30, 45]
for day in range( ndays ):

    bday = day % 7
    rotation[ day * 86400 * sps: (day+1) * 86400 * sps ] += numpy.radians( boresights[ bday ] )

# MKB magic
sel = numpy.sin(elevation)
cel = numpy.sqrt(1.0 - sel * sel)
saz = numpy.sin(azimuth)
caz = numpy.cos(azimuth)
pa_bc = numpy.arctan2(-saz, cel * numpy.tan(lat) - sel * caz)
sdec = sel * numpy.sin(lat) + cel * numpy.cos(lat) * caz
cdec = numpy.sqrt(1.0 - sdec * sdec)
sha = cel * saz / cdec
ha_bc = numpy.arcsin(sha)
dec_bc = numpy.arcsin(sdec)

lst = ((1.00273790935 * ctime) % 86400.0) * 2.0 * numpy.pi / 86400.0

ra_bc = lst - ha_bc

# dummy polAng array to zero.
detPolDummy = numpy.zeros_like( detPol )

ra, dec, pa = focalplane_to_equatorial(ra_bc, dec_bc, pa_bc + rotation, azOff, elOff, delPolDummy )
tht = numpy.pi/2.0 - dec

# pixelate pointing
pixels = healpy.ang2pix( args.nside, ra, tht )
tht_pix, ra_pix = healpy.pix2ang( args.nside, pixels )
dec_pix = numpy.pi/2.0 - tht_pix
# Is this correct?
pa_pix  = pa

# Save file
numpy.savez( output_file, ra=ra_pix, dec=dec_pix, pa=pa_pix )

