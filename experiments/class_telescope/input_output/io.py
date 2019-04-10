import re
import os
import pandas
import numpy
import pygetdata

from pisco.tod import TOD
from raw_data_manager import RawDataManager as RDM

@staticmethod
def todNameToTodPath( todName, basepath ):
    '''
    '''

    # Extract year, month, day and time
    yy,mm,dd,HH,MM,SS = todName.split( '-' )

    path = "%s-%s/%s-%s-%s/%s-%s-%s-%s-%s-%s/" % (yy,mm,yy,mm,dd,yy,mm,dd,HH,MM,SS)

    print path

    return os.path.join( basepath,path )

def dataPkgToTOD( systemPath, detStreamFields='all', readDetData=True, mce='mceq' ):
    '''
    Generates a DataFrame from CLASS packaged data.

    input:
        systempath : absolute system path where the package lives
        
        detStreamField : receiverData will have a field that specifies the name of
                         the files detector data is being written to. In the case of
                         CLASS, this is located in the `channel` field of receiverData.

                         By default, `dataPkgToTOD` will read all detector channels available.

        include_async : if set to True, asynchronous data is synchronized with
                        the master clock (acu_sync_ut1) and oversampled to 200Hz
                        so as to create a single giant DataFrame with ALL the data.
                        Use with care as it will produce A LOT more data.
    output:

        dataFrame with CLASS data.

    ''' 
    #systemPath = todNameToTodPath( todname, basepath )
    rdm     = RDM( systemPath )
    dirfile = pygetdata.dirfile( rdm.dfpaths['sync'] )
    data    = TOD()

    # Read in azimuth, altitude, boresight rotation and ctime. These are
    # all synchronous data streams. 
    ctime  = dirfile.getdata('acu_sync_ut1').astype('float64')
    az     = dirfile.getdata('acu_azimuth').astype('float64')
    alt    = dirfile.getdata('acu_elevation').astype('float64')
    vpm    = dirfile.getdata('acu_vpm4' ).astype( 'float64' ) 
    
    # Treat boresight separately because it is sampled at a lower sampling rate
    rot    = dirfile.getdata('acu_bore_encoder').astype('float64')
     
    # Oversample bs
    # Get frame rate for ctime
    spsCtime = dirfile.spf( 'acu_sync_ut1' )
    spsBore  = dirfile.spf( 'acu_bore_encoder' )
    factor   = spsCtime/spsBore
    rot = numpy.interp( ctime, ctime[::factor], rot )
    
    # Find bad samples in az,alt and rot and interpolate
    #bad_az_samples  = numpy.where( numpy.abs(  az - 1e10 ) > 1e10 )
    #bad_alt_samples = numpy.where( numpy.abs( alt - 1e10 ) > 1e10 )
    #bad_rot_samples = numpy.where( numpy.abs( rot - 1e10 ) > 1e10 )
    
    data.initialize( ctime, az, alt, rot )
    
    # Add in the VPM
    data.vpm = vpm

    # Read in pointing correction fields
    # acu_para_tilt_avg, acu_perp_tilt_avg, acu_el_para_tilt, acu_el_perp_tilt
    data.paraTilt   = dirfile.getdata('acu_para_tilt'   ).astype('float64')
    data.perpTilt   = dirfile.getdata('acu_perp_tilt'   ).astype('float64')
    data.elParaTilt = dirfile.getdata('acu_el_para_tilt').astype('float64')
    data.elParaTilt = dirfile.getdata('acu_el_perp_tilt').astype('float64')

    # Transform angular coordinates to radians
    data.az  /= numpy.degrees(1)
    data.alt /= numpy.degrees(1)
    data.rot /= numpy.degrees(1)

    data.paraTilt   /= numpy.degrees(1)
    data.perpTilt   /= numpy.degrees(1)
    data.elParaTilt /= numpy.degrees(1)
    data.elParaTilt /= numpy.degrees(1)

    # Load vpm encoder data
    data.vpm         = 3.1663 - dirfile.getdata('acu_vpm4').astype('float64')


    if not os.path.exists( os.path.join( rdm.dfpaths['sync'], mce ) ):
        raise RuntimeError, 'Path %s does not exists' % os.path.join( pygetdata.dirfile( rdm.dfpaths['sync'] ), mce )
    
    print 'reading : ', os.path.join( rdm.dfpaths['sync'], mce )
    det_df = pygetdata.dirfile( os.path.join( rdm.dfpaths['sync'], mce ) )
    
    # List all available detector channels
    det_fields = []
    if detStreamFields == 'all':
        
        fields_to_load = sorted( det_df.field_list() )
        
        for f in fields_to_load:
    
            if 'filt' in f and detStreamFields == 'all' and 'INTER' not in f:
        
                det_fields.append( f )
    
    det_fields = sorted( det_fields )
    
    data.detdata = numpy.empty( (len(det_fields), data.ctime.size ), dtype='float64' )

    if readDetData:
        for c in det_fields:
            
            row,col = re.findall(r'\d+', c )
            row = int( row )
            col = int( col )
            uid = 11*col + row

            detdata = det_df.getdata( c ).astype( 'float64' )
            # check sizes
            if detdata.size != data.ctime.shape:
                # Do zero padding
                offset = detdata.size - data.ctime.size
                # negative offset means detector data is shorter
                if offset < 0:
                    detdata = numpy.concatenate( (
                        detdata, numpy.repeat( (0), numpy.abs(offset) ) ) )
                    # Flag these samples as bad
                    data.ctime[ -offset:-1 ] = 0

            data.detdata[ uid ] = detdata

    data.detdata = data.detdata.astype( 'float64' )

    return data
