import os

import numpy
import pandas

from ConfigParser import SafeConfigParser

def parse_config_file( path ):
    
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
    cols = ['Detector','Feed','AzOff','ElOff','Det_pol']
    
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
                          'detPol' : detPol }
    # done 
    # --------------------------------------------------------------------------------------------------#
    
    # --------------------------------------------------------------------------------------------------#
    # Parse, load and process beam parameters
    # --------------------------------------------------------------------------------------------------#
    cols = ['Detector','Feed','FWHM_x','FWHM_y', 'Theta']
    
    beamParamsFile = parser.get( 'Beams', 'file' )
    inDegrees = parser.getboolean( 'Beams', 'angles in degrees' )
    averagePairBeams = parser.getboolean( 'Beams', 'average beams for pairs' )
    
    # load data
    beamData = pandas.read_csv( beamParamsFile, usecols=cols )
    
    uids  = beamData['Detector' ].values
    feeds = beamData['Feed'].values
    fwhmX = beamData['FWHM_x'].values
    fwhmY = beamData['FWHM_y'].values
    theta  = beamData['Theta'].values
    
    if not inDegrees:
        fwhmX  = numpy.rad2deg( fwhmX )
        fwhmY  = numpy.rad2deg( fwhmY )
        theta  = numpy.rad2deg( theta )

    if averagePairBeams:
        # iterate over feeds
        for feed in feeds:
            
            # find uids in the feed
            pairUids = uids[ numpy.where( feeds == feed ) ]
            
            if( len(pairUids) != 2 ):
                raise RuntimeError( "feed %d has only one detector." % (feed) )

            # array indexes, which might not be actual uids
            idx0 = numpy.argwhere( uids == pairUids[0] )
            idx1 = numpy.argwhere( uids == pairUids[1] )

            _fwhmX = 0.5 * ( fwhmX[ idx0 ] + fwhmX[ idx1 ] )
            _fwhmY = 0.5 * ( fwhmY[ idx0 ] + fwhmY[ idx1 ] )
            _theta = 0.5 * ( theta[ idx0 ] + theta[ idx1 ] )

            fwhmX[ idx0 ] = _fwhmX
            fwhmX[ idx1 ] = _fwhmX
            
            fwhmY[ idx0 ] = _fwhmY
            fwhmY[ idx1 ] = _fwhmY

            theta[ idx0 ] = _theta
            theta[ idx1 ] = _theta
    
    opts[ 'beams' ] = { 'uids' : uids,
                        'fwhmX' : fwhmX,
                        'fwhmY' : fwhmY,
                        'theta' : theta }
                        
    # done 
    # --------------------------------------------------------------------------------------------------#
    
    return opts
