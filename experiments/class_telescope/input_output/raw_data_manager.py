import json

import os
import shutil
import re
import glob
import time
import datetime
import warnings

import pygetdata
from pygetdata import dirfile
from pygetdata import entry

import numpy
from numpy import fmod, interp, where , floor, linspace

import fileinput


import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S")

rdmlogger = logging.getLogger('class_telescope.rdm')
rdmlogger.setLevel( logging.INFO )

'''
Utilitary functions
'''
# Stolen from http://stackoverflow.com/questions/4494404/ \
# find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = numpy.diff( condition.astype('int32') ).astype( 'bool' )
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = numpy.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = numpy.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

class RawDataManagerError(Exception):
    pass

class RawDataManager(object):
    '''
    @brief Contains methods and parameters to handle CLASS packaged data format.
    '''
    #Static dictionary containing directory/file structure of CLASS packaged data.
    _datainfo = {'q_band': {'fields':{}, 'streams':{}}}

    '''Shorthands to frequently required fields'''
    _datainfo['q_band']['fields']['timing'] = {
            'unixtime': 'acu_sync_ut1',
            'syncnumber': 'acu_sync_serial',
            'syncdelta': 'acu_sync_delta',}

    _datainfo['q_band']['fields']['pointing'] = {
            'az': 'acu_azimuth',
            'alt': 'acu_elevation',
            'rot': 'acu_bore_encoder',}

    _datainfo['q_band']['fields']['mask'] = {
        'acu': 'acu_data_valid',
        'mce': 'mceq_data_valid',}

    '''Map from human readable names to dirfile locations'''
    _datainfo['q_band']['streams']['hk'] = {
        'weather': 'site/weather',
        'diodes': 'mount1/async/q_band/diodes',
        'temps': 'mount1/async/temperature',
        'ln2_level': 'mount1/async/q_band/cold_trap_level',
        'slow rtds': 'mount1/async/q_band/rtds/slow',
        'fast rtds': 'mount1/async/q_band/rtds/fast',
        'pressure': 'mount1/async/q_band/pressure',
        'flow': 'mount1/async/q_band/flow',
        'vpm': 'mount1/async/q_band/vpm',
        'abob': 'mount1/async/abob',
        'blufors': 'mount1/async/q_band/bluefors_status',}

    _datainfo['q_band']['streams']['sync'] = {'sync': 'mount1/sync',}
    _datainfo['q_band']['other'] = {
            'mce': 'mount1/async/q_band/mce',
            'vpm': 'mount1/async/q_band/vpm', }
    
    dfpaths  = {}
    mcedata  = None
    vpmdata  = None
    abobdata = None

    def __init__(self, path, band='q_band', verbose=False):
        '''
        @brief initializes a RawDataManager pointing to path.

        @param path : system path with CLASS packaged data.
        @param band : observing band. Defaults to 'q_band'.
        @param verbose : set to True to print a lot of output.
        '''

        self.basepath = path

        
        if band not in ['q_band',]:
            rdmlogger.error( '\nPlease specify a valid observing band.' )
        self.band = band

        #loads metadata info
        try:
            f = open( os.path.join( self.basepath, 'metadata.json' ) )
            self._metadata = json.load( f )
            f.close()
        except Exception,e:
            rdmlogger.critical( '\nCannot load metadata.json in %s ' % self.basepath )
            raise RawDataManagerError
        
        # Setup dirfile paths
        for S in self._datainfo[self.band]['streams']:
            for s in  self._datainfo[self.band]['streams'][S]:
                relpath = self._datainfo[self.band]['streams'][S].get( s )

                if relpath in self._metadata['dirfiles']:
                    abspath = str( os.path.join( self.basepath, relpath ) )
                    self.dfpaths[s] = abspath
                    debug_msg =  '%s::%s'%(S,s),'->',abspath
                    #rdmlogger.debug(debug_msg)

        # Look for MCE related data
        relpath = self._datainfo[self.band]['other'].get( 'mce' )
        abspath = str( os.path.join( self.basepath, relpath ) )
        if os.path.exists( abspath ):
            if os.listdir( abspath ):
                self.mcedata = {}
                self.mcedata[ 'runfiles' ] = {}
                # List all .run.gz files in the mce folder
                # This is an ugly hack, I know
                year = int( self._metadata['time'].split('T')[0].split('-')[0] )
                runfiles = glob.glob( os.path.join( abspath, str(year)+'*.run.gz' ) )
                for r in runfiles:
                    runfile_time = os.path.basename( r ).strip('.run.gz') + '-0-0-0'
                    runfile_time = map( int, runfile_time.split( '-' ) )
                    runfile_time = str( int( time.mktime(runfile_time) ) )
                    self.mcedata['runfiles'][ runfile_time ] = r

                self.mcedata[ 'ivs' ] = {}
                iv_files = glob.glob( os.path.join( abspath, 'iv_*.gz' ) )
                for iv in iv_files:
                    iv_type = ''
                    if '.bias' in iv:
                        iv_type = 'bias'
                    if '.run' in iv:
                        iv_type = 'run'
                    if '.out' in iv:
                        iv_type = 'out'
                    
                    iv_time = os.path.basename( iv ).strip( 'iv_' ).strip(iv_type+'.run.gz')
                    iv_time = map( int, (iv_time + '-0-0-0').split( '-' ) )
                    iv_time = str( int( time.mktime(iv_time) ) )
                    
                    if not iv_time in self.mcedata['ivs'].keys():
                        self.mcedata['ivs'][ iv_time ] = {}
                    
                    self.mcedata['ivs'][ iv_time ][ iv_type ] = iv               

        # Look for VPM related data        
        relpath = self._datainfo[self.band]['other'].get( 'vpm' )
        abspath = str( os.path.join( self.basepath, relpath ) )
        if os.path.exists( abspath ):
            self.vpmdata = {}
            self.vpmdata[ 'control dirfile' ] = abspath
    
    def has_sync_data( self ):

        return ( self.dfpaths.get('sync',False) and True )


    def has_detector_data( self ):
        '''
        @brief Returns True if path contains detector data. 
        '''
        
        det_df_path =  self.dfpaths.get('sync', None)
        if det_df_path is None:
            return False

        det_df = dirfile( det_df_path )
        try:
            det_df.validate( 
                self._datainfo[self.band]['fields']['mask']['mce'])
            det_df.close()
        except:
            det_df.close()
            return False

        return True
   
    def has_acu_data( self ):
        '''
        @brief Returns True if path contains ACU data. 
        '''
        
        det_df_path =  self.dfpaths.get('sync', None)
        if det_df_path is None:
            return False

        det_df = dirfile( det_df_path )
        try:
            det_df.validate( 
                self._datainfo[self.band]['fields']['mask']['acu'] )
            det_df.close()
        except:
            det_df.close()
            return False

        return True
    
    def has_mce_data( self ):
        '''
        @brief Returns True if path contains MCE data.'
        '''

        return bool( self.mcedata )

    def has_vpm_data( self ):
        '''
        @brief Returns True if data package contains VPM control file.
        '''
        return bool( self.vpmdata )

    def has_mce_runfiles( self ):
        '''
        @brief Returns true if path has at least 1 MCE runfile.'
        '''
        if not self.has_mce_data():
            return False

        return bool( self.mcedata['runfiles'] )
            

    def has_iv_curves( self ):
        '''
        @brief Returns true if path has at least IV curves.'
        '''
        if not self.has_mce_data():
            return False
        
        return bool( self.mcedata['ivs'] )

    def available_hk( self ):
        '''
        @brief returns a list with available housekeeping data streams.
        '''
        f = self.dfpaths.keys()
        try:
            f.remove('sync')
        except:
            pass
        return f
        
    def get_basename( self ):
        '''
        @brief returns the basepath of the raw data location.
               used to build TODInfo object.
        '''
        return os.path.dirname( self.basepath.rstrip('/') )

    def get_filename( self ):
        '''
        @brief  returns the filename of the raw data.
                usef to build TODInfo object.
        '''
        return os.path.basename( self.basepath.rstrip('/') )

    def get_name( self ):
        '''
        @brief  return the name of the raw data. defaults
                to 'unknown'
        '''
        return self._metadata.get( 'name', 'unknown' )

    def get_id( self ):
        '''
        @brief  return the id of the metadata. Defaults to None.
        '''
        return self._metadata.get( '_id', 'unknown' )

    def get_runfile( self ):
        '''
        @brief returns a path the runfile in the TOD.
        '''

        if not self.has_mce_data():
            return None
        
        # Try regular runfiles

        rf_ctimes = self.mcedata['runfiles'].keys()
        # Sort backwards so the first we find is the newest one   
        #rf_ctimes = rf_ctimes[::-1]
        for ctime in rf_ctimes:
            if self.mcedata['runfiles'].get( ctime, False ):
                return  self.mcedata['runfiles'].get( ctime )

        # Try iv runfiles 
        iv_ctimes = sorted( self.mcedata['ivs'].keys() )
        # Sort backwards so the first we find is the newest one   
        iv_ctimes = iv_ctimes[::-1]
        for ctime in iv_ctimes:
            if self.mcedata['ivs'][ ctime ].get( 'run', False ):
                return  self.mcedata['ivs'][ ctime ].get( 'run' )

        # Give up

        return None

    def get_all_runfiles( self ):
        '''
        @brief returns a dictionary with the ctime:path pair for runfiles present in the Data Package.
        '''

        r = {}
        if not self.has_mce_data():
            rdmlogger.debug( "No runfiles present in this data package." )
            return r

        # Try regular runfiles.
        if self.mcedata['runfiles']:
            r.update( self.mcedata.get('runfiles') )

        if self.mcedata['ivs']:
            # Runfiles might also come from I-V procedures.
            ctimes = self.mcedata['ivs'].keys()
            for t in ctimes:
                r.update( {t : self.mcedata['ivs'][t]['run']} )

        return r


    def get_iv_out( self ):
        '''
        @brief returns a path the iv.out file the TOD.
        '''

        if not self.has_iv_curves():
            return None
        
        # Try with I-V curve generated runfiles
        iv_ctimes = self.mcedata['ivs'].keys() 
        # Sort backwards so the first we find is the newest one   
        #iv_ctimes = iv_ctimes[::-1]
        for ctime in iv_ctimes:
            if self.mcedata['ivs'][ ctime ].get( 'out', False ):
                return  self.mcedata['ivs'][ ctime ].get( 'out' )
        
        return None

    def get_all_iv_out( self ):
        '''
        @brief returns a dictionary with ctime:path pairs for all iv.out files in the Data Package.
        '''
        r = {}
        if not self.has_iv_curves():
            rdmlogger.debug( "No runfiles present in this data package." )
            return r

        # .out files from I-V procedures.
        ctimes = self.mcedata['ivs'].keys()
        for t in ctimes:
            if 'out' in self.mcedata['ivs'][t]:
                r.update( {t : self.mcedata['ivs'][t]['out']} )
        return r


    def _get_start_end_from_sync_field( self, field_code ):
        '''
        @brief returns start and end values of a field in the synchronous stream.
        '''
        if not self.has_acu_data():
            raise RuntimeError, 'No available synchronous data.'

        df     = pygetdata.dirfile( self.dfpaths.get('sync', None ) )
        # First valid frame
        acu_mask = df.getdata( 'acu_data_valid' )
        first_valid_frame = numpy.argmax( acu_mask )
        acu_mask = acu_mask[::-1]
        last_valid_frame = len( acu_mask ) - numpy.argmax( acu_mask ) - 1
        
        spf    = df.spf( field_code )
        
        start_frame = df.getdata( 
                field_code, 
                first_sample=0, num_samples=spf, 
                first_frame=first_valid_frame )

        end_frame = df.getdata( field_code, 
                first_sample=0, num_samples=spf,
                first_frame=last_valid_frame )
        
        if start_frame.size < 1 or end_frame.size < 1:
            #rdmlogger.error( 'synchronous dirfile at %s seems corrupt.' % (self.dfpaths['sync']) )
            s = -1
            e = -1
            return s,e 

        if field_code == self._datainfo[self.band]['fields']['timing']['unixtime']:
            try:
                s = numpy.min( start_frame[ start_frame > 0 ] )
                e = numpy.max( end_frame[ end_frame > 0 ] )
            except:
                #rdmlogger.error( 'synchronous dirfile at %s seems corrupt.' % (self.dfpaths['sync']) )
                s = -1
                e = -1
        else:
            s = numpy.min( start_frame )
            e = numpy.max( end_frame   )
         
        #print s,e

        df.close()
            
        return s,e

    def get_start_end_ctime( self ):
        '''
        @brief returns start and end ctimes.
        '''
        return self._get_start_end_from_sync_field( self._datainfo[self.band]['fields']['timing']['unixtime'] )

    def get_start_end_azimuth( self ):
        '''
        @brief returns start and end ctimes.
        '''
        return self._get_start_end_from_sync_field( self._datainfo[self.band]['fields']['pointing']['az'] )
    
    def get_start_end_altitude( self ):
        '''
        @brief returns start and end ctimes.
        '''
        return self._get_start_end_from_sync_field( self._datainfo[self.band]['fields']['pointing']['alt'] )

    def get_start_end_bs( self ):
        '''
        @brief returns start and end ctimes.
        '''
        return self._get_start_end_from_sync_field( self._datainfo[self.band]['fields']['pointing']['rot'] )
   
    def was_vpm_running( self , min_running_time=10 ):
        '''
        @brief checs if VPM was on more than min_running_time seconds 

        returns 100 if if was running the whole time
                 10 if it was running more than min_running_time
                  0 if it was not running
               -100 if we can't tell because there appears to be no vpm data available.
        '''

        if self.has_vpm_data():

            vpm_df_path = self.vpmdata['control dirfile']
            vpm_df = dirfile( vpm_df_path )
            rec  = vpm_df.getdata( 'recording' )
            
            if not numpy.all( rec ) or rec.size < 1:
                rdmlogger.warning( 'VPM status unknown because recording field is not 1.' )
                return -1

            time = vpm_df.getdata( 'time' )
            tun  = vpm_df.getdata( 'tuning' )

            # This means it was running the whole time
            if numpy.all( tun.astype('bool') ):
                return 100

            # Magic
            vpm_on_chunks = contiguous_regions( tun.astype( 'bool' ) )
            for startend in vpm_on_chunks:

                start = time[ startend[0]    ]
                end   = time[ startend[1] - 1]

                if end - start > min_running_time:

                    return 10

            return 0
        
        return -100

    def get_scan_type( self ):
        '''
        @brief Looks at the acu_obs_state BIT and returns accordingly.

        @returns: 
            'susp' if BIT field is not consistent along the data package.
            
            The rest is taken from the original format file specs.
            # Observing State: 0 = Idle       1 = Five Point    2 = Sky Dip       3 = Az scan 
            #                  4 = Dec Scan   5 = Drift Scan    6 = Profile Scan 15 = Aborted

            Returned names don't have white spaces nor caps lock.
        '''

        state = 'unknown'
   
        try:
            df  = dirfile( self.dfpaths['sync'] ) 
            obs_state = df.getdata( 'acu_obs_state' )
            df.close()
        except:
            rdmlogger.error( 'Cannot read acu_obs_state field on this TOD. Flagging as scan_type as unknown.')
            state = 'unknown'
            return state
        
        if not numpy.all( obs_state ):
            rdmlogger.warning( 'Observing status changed during data adquisition. Flagging as multiple.' )
            state = 'multiple'
        
        elif numpy.allclose( obs_state , 0 ):
            state = 'idle'

        elif numpy.allclose( obs_state , 1 ):
            state = 'fivepoint'
        
        elif numpy.allclose( obs_state , 2 ):
            state = 'skydip'
        
        elif numpy.allclose( obs_state , 3 ):
            state = 'azscan'
        
        elif numpy.allclose( obs_state , 4 ):
            state = 'decscan'
        
        elif numpy.allclose( obs_state , 5 ):
            state = 'driftscan'
        
        elif numpy.allclose( obs_state , 6 ):
            state = 'profscan'
        
        elif numpy.allclose( obs_state ,15 ):
            state = 'aborted'

        else:
            state = 'unknown'

        return state


    @staticmethod
    def synchronize_field( data, slaveclock, masterclock ):
        '''
        @brief Wrapper to numpy.interpolate, with some caveats to handle
               non-uniform coverage of the data and mantain original spf.

        @param data        : numpy array with data to be synchronized
        @param slaveclock  : timebase for data
        @param masterclock : new timebase
        
        Note: this routine will not add more data if the new timebase is more
        extended than the old one. Instead, it will the rightmos and leftmost values.
        '''
        
        #Obtain average sampling rate of slave clock'''
        slv_sps = numpy.average( numpy.diff( slaveclock ) )

        #Handle timebase coverage'''
        min_mtime = numpy.min( masterclock )
        max_mtime = numpy.max( masterclock )
        masterclock = numpy.arange( min_mtime-slv_sps, max_mtime+slv_sps, slv_sps )

        min_stime = numpy.min( slaveclock )
        max_stime = numpy.max( slaveclock )

        mclock = numpy.asarray( masterclock )
        sclock = numpy.asarray( slaveclock )

        left  = 0
        right = 0
        if min_mtime < min_stime:
            mclock =  mclock[ masterclock >= min_stime ]
            left = masterclock[ masterclock < min_stime ].size
        
        if max_mtime > max_stime:
            mclock =  mclock[ mclock < max_stime + slv_sps ]
            right = masterclock[ masterclock > max_stime ].size
        
        new_data = interp( mclock, sclock, data )
    
        return new_data

