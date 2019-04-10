import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d]\n%(message)s",
    datefmt="%H:%M:%S")

ivoutlogger = logging.getLogger('class_telescope.ivout')
ivoutlogger.setLevel( logging.DEBUG )

import gzip
import re
import os
import array
import numpy
import numpy as np

from pisco.calibration import Calibrator
from pisco.experiments.class_telescope.input_output import raw_data_manager as RDM
from pisco.experiments.class_telescope.misc.mce import MCERunfile

class IVOut(object):
    '''
    Python Object to compute raw data to physical units such as TES
    current (A) and power (pW) on the detectors by using I-V data.
    ''' 
    
    # Nominal values for Q-band. Stolen from JA code 
    G_array = np.array([
               145, 128, 176,  -1, 187, 140,  -1, 151, 153, 183,  -1,
               166, 185, 178, 181, 193, 179, 187, 200, 200,  -1, 199,
               190, 204, 195, 195, 214, 200,  -1,  -1, 226,  81, 229,
                -1, 136, 137, 149, 147, 134, 133, 140, 139, 136, 143,
                -1, 211, 210, 195, 193, 184,  63, 170, 193, 199,  -1,
               125, 138, 153, 148, 139, 140, 164, 148, 140,  -1, 124,
               150, 136, 129, 147, 180, 203, 124,  -1,  -1,  -1, 129,
               167, 193, 186, 186, 209, 201, 189, 199, 182, 183,  -1 ]) #pW/K
	
    C_array= 2.5 * np.ones(88) #pJ/K

    tau_thermal = C_array/G_array

    polarity = -1
    dac_bits = 14
    M_ratio = 24.6
    Rfb = 5100.
    filtgain = 2048.
    decimation = 1.0/113.
    sample_freq = 50e6/200./11.

    # i-v name 
    name     = 'unknown'

    # Mce stuff 
    mce_name = None
    runfile  = None

    def __init__( self, iv_out_path, use_gzip=True ):
        '''
        @brief returns IVCalibration object directly from .out file
        @param use_gzip : open as GZIP if required.

        @note : this method is not intended to be called at the user level
                as it will return an incomplete representation of the I-V 
                calibration object.
        '''
        
        # I-V cannot be loaded without its corresponding runfile.
        runfile_path = iv_out_path.replace('out', 'run' )
        if not os.path.isfile( runfile_path ):
            err = os.path.dirname( iv_out_path ), 'does not contain a runfile that matches the i-v file.'
            raise RuntimeError, err
        
        # Associate runfile 
        self.runfile = MCERunfile( runfile_path )
        self.mce_name = self.runfile.Item('FRAMEACQ', 'HOSTNAME')[0]

        # Name is just the filename wihtout the extensions
        self.name = str( os.path.basename( iv_out_path ).split('.')[0] )
        
        # Read the whole damn thing
        lines = []
        if use_gzip:
            f = gzip.open( iv_out_path )
        else:
            f = open( iv_out_path )
        lines = f.readlines()
        f.close()
        
        self.parse_ivoutfile_to_props( lines )

        self.resp = numpy.asarray   ( self.resp )
        self.resp = numpy.nan_to_num( self.resp )

        self.resp_fit = numpy.asarray   ( self.resp_fit )
        self.resp_fit = numpy.nan_to_num( self.resp_fit )

        self.tau_factor_fit = numpy.asarray   ( self.tau_factor_fit )
        self.tau_factor_fit = numpy.nan_to_num( self.tau_factor_fit )

    def parse_ivoutfile_to_props( self, lines ):
        '''
        @brief parses the i-v out lines.
        @param lines : list of text lines.
        @raises : RuntimeError if something goes wrong.

        This function assumes each line has correct i-v out formatting. Based
        on this, it will dinamically add the i-v out properties to the Python
        object. 
        '''
        
        #Get cols and rows 
        num_cols = int( self.runfile.Item( 'HEADER', 'RB cc num_cols_reported' )[0] )
        num_rows = int( self.runfile.Item( 'HEADER', 'RB cc num_rows_reported' )[0] )
        for line in lines:
            # Extract property name and column (thanks LP!)
            name, col = re.search(r'^<(?P<name>.+)_C(?P<col>\d+)>', line).groups()
            col       = int( col )
            values    = line.split()[1::]
            # TODO: add forbidden character checking besides () statements!
            name = name.replace( '(', '' ).replace( ')', '' )
        
            # Add property if it hasn't been already added
            if not hasattr( self, name ):
                setattr( self, name, [ None ] * num_cols  )
            
            # Check that there are num_row values
            if len( values ) != num_rows:
                raise RuntimeError, 'something is wrong with the iv.out!'

            # Add the data
            exec( 'self.%s[%d] = ( map( float, values ) )' % (name,col) )

def calibrate_tod( tod, iv_out_path ):
    '''
    returns Calibrates a TOD to pW
    '''

    ivout = IVOut( iv_out_path, use_gzip=True )

    C             =  Calibrator()
    C.name        =  str( ivout.name ) 
    C.description = 'Calibration from I-V curves' 
    C.calType     = 'ivout'

    resp_fit = ivout.resp_fit
    
    dI_dDAC  = 1./2**ivout.dac_bits / ivout.M_ratio / ivout.Rfb / ivout.filtgain   	
    
    C.coeffs = numpy.zeros( ( 1+tod.rows.max(),1+tod.cols.max() ), dtype=float )

    for c,r in zip( tod.cols, tod.rows):
        C.coeffs[r][c] = resp_fit[c][r] * dI_dDAC * ivout.polarity * 1000
    
    for r,c in zip( tod.rows, tod.cols):
        # Transform row/col to det_uid
        uid = np.where( (tod.rows == r) * (tod.cols == c) )[0][0] 
        tod.data[ uid ] *= C.coeffs[r][c]
