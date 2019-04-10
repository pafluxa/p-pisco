import numpy
import pandas

def downsample_filter_simple( data_in, n_iter=1, offset=0):                                                    
    """                                                                                                       
    Do nearest-neighbor remixing to downsample data_in (..., nsamps)                                          
    by a power of 2.                                                                                          
    """                                                                                                       
    if n_iter <= 0:                                                                                           
        return None                                                                                           
    ns_in = data_in.shape[-1]                                                                                 
    ns_out = ns_in // 2                                                                                       
    dims = data_in.shape[:-1] + (ns_out,)                                                                     
    data_out = numpy.empty(dims, dtype=data_in.dtype)                                                            
    # Central sample                                                                                          
    data_out[...,:] = data_in[...,offset:ns_out*2+offset:2] * 2                                               
    # To the left (all output samples except maybe the first)                                                 
    l_start = 1-offset                                                                                        
    l_count = ns_out - l_start                                                                                
    data_out[...,l_start:] += data_in[...,(1-offset):2*l_count:2]                                             
    # To the right (all output samples except maybe the last)                                                 
    # depending on 2*ns_out+offset <= ns_in                                                                   
    r_count = (ns_in - offset) // 2                                                                           
    data_out[...,:r_count] += data_in[...,offset+1::2]                                                        
    # Normalization...                                                                                        
    data_out[...,:] /= 4                                                                                      
    if l_start > 0:                                                                                           
        data_out[...,0] *= 4./3                                                                               
    if r_count < ns_out:                                                                                      
        data_out[...,-1] *= 4./3                                                                              
    if n_iter <= 1:                                                                                           
        return data_out                                                                                       
    # Destroy intermediate storage, and iterate                                                               
    data_in = data_out                                                                                        
    return downsample_filter_simple(data_in, n_iter-1, offset)

def concatenate_tod( tod, _tod ):

    tod.name = _tod.name
    tod.instrument = _tod.instrument
    tod.receiver = _tod.receiver

    tod.detdata = numpy.append( tod.detdata, _tod.detdata, axis=1 )

    tod.ctime = numpy.append( tod.ctime, _tod.ctime )
    tod.az    = numpy.append( tod.az   , _tod.az    )
    tod.alt   = numpy.append( tod.alt   , _tod.alt  )
    tod.rot   = numpy.append( tod.rot   , _tod.rot  )

    for d in _tod.bad_dets:
        if d not in tod.bad_dets:
            tod.bad_dets.append( d )

    tod.pointing_mask = numpy.concatenate( (tod.pointing_mask, _tod.pointing_mask ) )
    
    tod.nsamples += _tod.nsamples
    
    tod.data_mask = numpy.concatenate( (tod.data_mask, _tod.data_mask), axis=1 )

    return tod

class TOD( object ):

    # misc. parameters
    name = ''
    instrument = ''
    receiver = ''

    # boresight pointing information
    ctime = None
    az    = None
    alt   = None
    rot   = None

    # Pointing mask, to ignore bad samples in pointing streams
    pointing_mask = None

    # detector data
    detdata = None
    # Bad, bad detectors!
    bad_dets = [] 
    # Data mask, to mask single samples of single detectors
    data_mask = None

	# metadata for the TOD
    metadata = {}

    nsamples = 0
	# dummy initializer
    def __init__( self ):
		
        pass

    def append( self, tod ):
       
        return concatenate_tod( self, tod )

    def initialize( self, ctime, az, alt, rot, pointing_mask=None, data_mask=None ):

        self.ctime = ctime
        self.az = az
        self.alt = alt
        self.rot = rot

        if pointing_mask is None:
            self.pointing_mask = numpy.zeros_like( ctime, dtype='int32' )

        else:
            self.pointing_mask = pointing_mask
        
        self.nsamples = ctime.size
        
        self.data_mask = data_mask

    def downsample( self, ds ):
        '''
        Downsamples the TOD by 2^ds
        '''

        self.ctime = downsample_filter_simple( self.ctime, ds )
        self.az    = downsample_filter_simple( self.az   , ds )
        self.alt   = downsample_filter_simple( self.alt  , ds )
        self.rot   = downsample_filter_simple( self.rot  , ds )
        
        self.detdata = downsample_filter_simple( self.detdata , ds)
       
        self.pointing_mask = numpy.zeros_like( self.ctime , dtype='int32' )
        self.pointing_mask[ self.ctime < 1000 ] = 1

        self.nsamples = self.ctime.size
        self.ndets    = self.detdata.shape[0]

