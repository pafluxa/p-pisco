import os

import numpy
import pandas

from ConfigParser import SafeConfigParser

def parse_config_file( path ):
    
    # debug: set seed to numpy.random
    numpy.random.seed( 42 )

    opts = {}

    parser = SafeConfigParser()
    parser.read( path )

    # --------------------------------------------------------------------------------------------------#
    # Parse output path
    # --------------------------------------------------------------------------------------------------#
    outPath = parser.get( 'Output', 'output folder' )
    outName = parser.get( 'Output', 'file name' )

    out = os.path.join( outPath, outName )
    opts[ 'outputFile' ] = out
    # done
    # --------------------------------------------------------------------------------------------------#
    
    # --------------------------------------------------------------------------------------------------#
    # Parse detector pointing info
    # --------------------------------------------------------------------------------------------------#
    basepath = parser.get( 'Detector pointing', 'basepath' )
    name     = parser.get( 'Detector pointing', 'name' )
    days     = parser.get( 'Detector pointing', 'days' )
    scans    = parser.get( 'Detector pointing', 'scans' )
    sps      = parser.get( 'Detector pointing', 'sps' )

    # build the filename
    # TODO: this is ugly, ugly, UGLY
    detPointingPath = os.path.join(
        basepath, name + "_" + \
                  "ndays"  + "_" +  days + "_"\
                  "nscans" + "_" + scans + "_"\
                     "sps" + "_" +   sps + "Hz" + ".npz" )

    opts[ 'pointingFile' ] = detPointingPath
    # done
    # --------------------------------------------------------------------------------------------------#
    
    # --------------------------------------------------------------------------------------------------#
    # Parse input map name
    # --------------------------------------------------------------------------------------------------#
    basepath = parser.get( 'Input map', 'basepath' )
    mtype    = parser.get( 'Input map', 'type' )
    n        = parser.get( 'Input map', 'n' )
    r        = parser.get( 'Input map', 'r' )
    nside    = parser.get( 'Input map', 'nside' ) 
    # build the filename
    # TODO: this is ugly, ugly, UGLY
    _name = "n" + "_" +  n + "_" + \
            "nside" + "_" + nside + ".npz"
    
    inputMapPath = os.path.join( basepath, mtype, "r" + r, _name )

    opts[ 'inputMapPath' ] = inputMapPath
    # done
    # --------------------------------------------------------------------------------------------------#


    # --------------------------------------------------------------------------------------------------#
    # Parse, load and process pointing offsets
    # --------------------------------------------------------------------------------------------------#
    cols = ['Detector','Feed','AzOff','ElOff','Det_pol', 'On']
    
    offsetsFile = parser.get( 'Pointing offsets', 'file' )
    inDegrees = parser.getboolean( 'Pointing offsets', 'angles in degrees' )
    averagePairOffsets = parser.getboolean( 'Pointing offsets', 
        'average offsets for pairs' )
    
    # load data
    recvData = pandas.read_csv( offsetsFile, usecols=cols )
    
    uids   = recvData[ 'Detector' ].values
    feeds  = recvData['Feed'].values
    azOff  = recvData['AzOff'].values
    elOff  = recvData['ElOff'].values
    detPol = recvData['Det_pol'].values
    state  = recvData['On'].values
    
    if inDegrees:
        azOff  = numpy.deg2rad(  azOff )
        elOff  = numpy.deg2rad(  elOff )
        detPol = numpy.deg2rad( detPol )

    if averagePairOffsets:
        # iterate over feeds
        for feed in feeds:
            
            # find uids in the feed
            pairUids = uids[ numpy.where( feeds == feed ) ]
            
            if( len(pairUids) != 2 ):
                raise RuntimeError( "feed %d has only one detector." % (feed) )

            # array indexes, which might not be actual uids
            idx0 = numpy.argwhere( uids == pairUids[0] )
            idx1 = numpy.argwhere( uids == pairUids[1] )

            _azOff = 0.5 * ( azOff[ idx0 ] + azOff[ idx1 ] )
            _elOff = 0.5 * ( elOff[ idx0 ] + elOff[ idx1 ] )

            azOff[ idx0 ] = _azOff
            azOff[ idx1 ] = _azOff

            elOff[ idx0 ] = _elOff
            elOff[ idx1 ] = _elOff

    opts['focalplane'] = {'uids'   :   uids,
                          'feeds'  :  feeds,
                          'azOff'  :  azOff,
                          'elOff'  :  elOff,
                          'detPol' : detPol,
                              'on' :  state }
    # done 
    # --------------------------------------------------------------------------------------------------#
    
    # --------------------------------------------------------------------------------------------------#
    # Parse, load and process beam parameters
    # --------------------------------------------------------------------------------------------------#
    cols = ['Detector','Feed','FWHM_x','FWHM_y', 'Theta']
    
    beamParamsFile = parser.get( 'Beams', 'file' )
    inDegrees = parser.getboolean( 'Beams', 'angles in degrees' )
    useNominal   = parser.getboolean( 'Beams', 'use nominal' )
    
    averagePairFWHM  = parser.getboolean( 'Beams', 'match beam fwhm for pairs' )
    averagePairTheta = parser.getboolean( 'Beams', 'match beam orientation for pairs' )
    
    # try to get parameter to enlarge beams
    amplifyMismatch = False
    try:
        amplifyMismatch = parser.getboolean( 'Beams', 'amplify mismatch' )
    except ValueError:
        pass
    
    # in degrees by default
    mismatch = 0.0
    if amplifyMismatch:
        mismatch = parser.getfloat( 'Beams', 'mismatch' )

    # load data
    beamData = pandas.read_csv( beamParamsFile, usecols=cols )
    
    uids  = beamData['Detector' ].values
    feeds = beamData[     'Feed'].values
    fwhmX = beamData[   'FWHM_x'].values
    fwhmY = beamData[   'FWHM_y'].values
    theta = beamData[    'Theta'].values
    
    if not inDegrees:
        fwhmX  = numpy.rad2deg( fwhmX )
        fwhmY  = numpy.rad2deg( fwhmY )
        theta  = numpy.rad2deg( theta )
    
    if useNominal:
        fwhmX  = 1.5 * numpy.ones_like ( fwhmX ) 
        fwhmY  = 1.5 * numpy.ones_like ( fwhmY ) 
        theta  =       numpy.zeros_like( theta )
    
    # if mismatch is on, we need to rotate one of the beams by 90 degrees
    beamRotDelta = 0.0
    if amplifyMismatch:
        beamRotDelta = 90
    
    # iterate over feeds
    for feed in feeds:
        
        # find uids in the feed
        pairUids = uids[ numpy.where( feeds == feed ) ]
        
        if( len(pairUids) != 2 ):
            raise RuntimeError( "feed %d has only one detector." % (feed) )
        
        # array indexes, which might not be actual uids
        idx0 = numpy.argwhere( uids == pairUids[0] )
        idx1 = numpy.argwhere( uids == pairUids[1] )
        
        if averagePairFWHM:
            
            _fwhmX = 0.5 * ( fwhmX[ idx0 ] + fwhmX[ idx1 ] )
            _fwhmY = 0.5 * ( fwhmY[ idx0 ] + fwhmY[ idx1 ] )
            
            fwhmX[ idx0 ] = _fwhmX + mismatch 
            fwhmY[ idx0 ] = _fwhmY  
            
            fwhmX[ idx1 ] = _fwhmX + mismatch
            fwhmY[ idx1 ] = _fwhmY
        
        if averagePairTheta:
            _theta = 0.5 * ( theta[ idx0 ] + theta[ idx1 ] )
            theta[ idx0 ] = _theta
            theta[ idx1 ] = _theta + beamRotDelta
        
    opts[ 'beams' ] = { 'uids'  : uids,
                        'fwhmX' : fwhmX,
                        'fwhmY' : fwhmY,
                        'theta' : theta }
                        
    # done 
    # --------------------------------------------------------------------------------------------------#
    
    return opts
