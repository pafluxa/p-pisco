import pisco
from pisco.beam_analysis.utils import *

import os
import sys
import glob

import numpy
import pylab
import healpy
import pandas

from numpy import log10

beam_par_file = sys.argv[1]
#----------------------------------------------------------------------------------------------------------#
# Read beam parameter file
print 'reading beam parameters'
beam_data = pandas.read_csv( beam_par_file )
uids      = beam_data[ 'uid']
feeds     = beam_data[ 'feed']
fwhm_x    = beam_data[ 'fwhm_x']
fwhm_y    = beam_data[ 'fwhm_y']
rotation  = beam_data[ 'rot']
#----------------------------------------------------------------------------------------------------------#

# Read (from terminal) the path to the beams
beams_path     = sys.argv[2]
for uid in uids:

    # Load data
    print 'loading',   os.path.join( beams_path ,'detector%d.npz' % (uid) )
    data = numpy.load( os.path.join( beams_path ,'detector%d.npz' % (uid) ) )
    # Extract pixelization properties
    meta_data  = data['mdata']
    limits     = numpy.radians( meta_data[1][0]['limits'] )
    nx, ny     = meta_data[1][0]['nx'], meta_data[1][0]['ny']
    grid_size  = limits[1] - limits[0] # Square grid

    # Extract electric fields from the grid
    E_co, E_cx = data['E_co'], data['E_cx']

    # Convert to HEALPIX grid
    beam_nside = 512
    beam_co    = azel_grid_to_healpix_beam( numpy.abs(E_co), nx, ny, grid_size/2.0, beam_nside )
    beam_cx    = azel_grid_to_healpix_beam( numpy.abs(E_cx), nx, ny, grid_size/2.0, beam_nside )

    '''
    healpy.orthview( 20*log10(beam_co), rot=(0,90), title='GRASP AZEL to HPX CO', sub=(1,2,1) , half_sky=True )
    healpy.orthview( 20*log10(beam_cx), rot=(0,90), title='GRASP AZEL to HPX CX', sub=(1,2,2) , half_sky=True )
    pylab.show()
    '''
    numpy.savez( 'detector%d.npz' % (uid), E_co=beam_co, E_cx=beam_cx, nside=beam_nside )
