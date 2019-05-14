#!/usr/bin/env python
# coding: utf-8
__author__="Michael K. Brewer, May 2019."

import numpy
import argparse

import os
from os.path import join as join_paths

import time
from datetime import datetime

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

# starting date
parser.add_argument(
    "-start_date", action="store", type=str, dest="startDate", default="15-12-1989T07:30:00",
    help="Start date used to compute the pointing stream. Must be in format DD-mm-YYYYTHH:MM:SS.\
    Defaults to P. Fluxa birthday ;-)" )
# tag
parser.add_argument(
    '-tag' , action='store', type=str, dest='tag', default="",
    help="Stream will be saved into specified output path with the tag appended" )

# output folder
parser.add_argument(
    "-output_path", action="store", type=str, dest="outputPath",
    help="Pointing stream will be saved into specified path. If the path exists, an exception is raised." )

parser.add_argument(
    '-beam_par_file',  action='store', type=str, dest='beamParamsFile',
    help="File with specifications of the beam parameters for each detector in CSV format.\
    Must contain columns called Detector, AzOff, ElOff. AzOff and ElOff in degrees.")

parser.add_argument(
    "-latitude", action="store", type=float, dest="latitude",
    default=-22.959748,
    help="Latitude of the experiment in degrees and decimal format. Defaults to CLASS latitude." )

parser.add_argument(
    "-longitude", action="store", type=float, dest="longitude",
    default=-67.787260,
    help="Longitude of the experiment in degrees and decimal format. Defaults to CLASS longitude." )

parser.add_argument(
    "-override", action="store", type=str, dest="skipCheck",
    default="no",
    help="Set to `yes` if you are sure of what you are doing" )

# TODO: add a parameter to let the ser choose between generating pointing in ICRS coordinates or not.
args = parser.parse_args()
#-----------------------------------------------------------------------------------------------------------

# Setup experiment location
lat = numpy.radians( args.latitude )

# Read beam parameter file
#----------------------------------------------------------------------------------------------------------#
print 'reading beam parameters'
beam_data = numpy.genfromtxt( args.beamParamsFile,
                              delimiter=',', names=True, dtype=None, encoding='ascii' )
uids      = numpy.array( beam_data[ 'Detector'] )
azOff     = numpy.array( beam_data[    'AzOff'] )
elOff     = numpy.array( beam_data[    'ElOff'] )

azOff  = numpy.deg2rad(  azOff )
elOff  = numpy.deg2rad(  elOff )
#----------------------------------------------------------------------------------------------------------#

d   = datetime.strptime( args.startDate, "%d/%m/%YT%H:%M:%S" )
ts  = float( time.mktime( d.timetuple() ) )
te  = ts + args.days * 86400.0
sps = args.sps
T   = 720

# Compute the number of scans as the duration of a scan
nscans = int( numpy.ceil( (te - ts)/T ) )
ndays  = args.days

# path to save everything to
outputName = ""

if args.tag == "":
    outputName = "classQbandPointing_ndays_%d_nscans_%d_sps_%dHz.npz" % (
                  (int)(ndays), (int)(nscans), (int)(sps) )

else:
    outputName = "classQbandPointing_ndays_%d_nscans_%d_sps_%dHz_tag_%s.npz" % (
                  (int)(ndays), (int)(nscans), (int)(sps), args.tag )

outputPath = join_paths( args.outputPath, outputName )

# check if the same output file exists
if os.path.exists( outputPath ):
    raise RuntimeError( "%s exists." % (outputPath) )


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
    rotation[ day * 86400.0 * sps: (day+1) * 86400.0 * sps ] += numpy.radians( boresights[ bday ] )

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
#Keep RA in range.
if ra_bc > 2.0 * numpy.py: ra_bc -= 2.0 * numpy.pi
elif ra_bc < 0.0: ra_bc -= 2.0 * numpy.pi


# dummy polAng array to zero.
polAng = numpy.zeros_like( elOff )
print azOff.shape, elOff.shape, polAng.shape
ra, dec, pa = focalplane_to_equatorial(ra_bc, dec_bc, pa_bc + rotation, azOff, elOff, polAng)


numpy.savez( outputPath, ra=ra, dec=dec, pa=pa )

