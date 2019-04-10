'''
Core methods for convolution: deproject_sky
'''
import pylab
import numpy

import healpy

import copy

from pisco.convolution._convolution import *
from pisco.beam_analysis.mueller import *

from multiprocessing import Process, Array, Lock

def deproject_sky_for_feedhorn(
        feed_ra, feed_dec, feed_pa, 
        det_pol_angle, 
        sky,
        beam_nside, (beam1_co, beam1_cx), (beam2_co, beam2_cx) ):
    '''
    '''
    '''Get number of available GPUs'''

    N_gpu = 3
    
    # Normalize beams
    beam1_pwr = ( numpy.conj(beam1_co)*beam1_co + numpy.conj(beam1_cx)*beam1_cx ).max()
    beam2_pwr = ( numpy.conj(beam2_co)*beam2_co + numpy.conj(beam2_cx)*beam2_cx ).max()

    # Set to ignore warnings
    numpy.warnings.filterwarnings('ignore')
    
    # Compute Mueller Matrices from the beams
    M_II = compute_TT( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_QI = compute_TQ( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_UI = compute_TU( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_VI = compute_TV( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    
    M_IQ = compute_QT( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_QQ = compute_QQ( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_UQ = compute_QU( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_VQ = compute_QV( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    
    M_IU = compute_UT( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_QU = compute_UQ( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_UU = compute_UU( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )
    M_VU = compute_UV( (beam1_co, beam1_cx), (beam2_co, beam2_cx) ).astype( 'float32' )

    # Set to ignore warnings                                                                                  
    numpy.warnings.filterwarnings('default')
    
    # Store Mueller Matrix as a single numpy array
    M    = numpy.asarray( [ 
                [M_II,M_QI,M_UI,M_VI],
                [M_IQ,M_QQ,M_UQ,M_VQ],
                [M_IU,M_QU,M_UU,M_VU]] )

    _I,_Q,_U,_V   = sky
    _I = _I.astype( 'float32' ) 
    _Q = _Q.astype( 'float32' ) 
    _U = _U.astype( 'float32' ) 
    _V = _V.astype( 'float32' )
    
    det_stream = numpy.zeros_like( feed_ra, dtype='float32' )

    # Create chunks of coordinates and data stream
    chunks_feed_ra   = numpy.array_split( feed_ra  , N_gpu )
    chunks_feed_dec  = numpy.array_split( feed_dec , N_gpu )
    chunks_feed_pa   = numpy.array_split( feed_pa  , N_gpu )
    det_pol_angle_list = [ det_pol_angle ] * N_gpu

    beam_nside_list    = [ beam_nside    ] * N_gpu
    
    M_list             = [ M             ] * N_gpu
    
    _I_list            = [ _I            ] * N_gpu
    _Q_list            = [ _Q            ] * N_gpu
    _U_list            = [ _U            ] * N_gpu
    _V_list            = [ _V            ] * N_gpu
    

    chunks_data_       = numpy.array_split( det_stream, N_gpu )
    
    lock = Lock()
    chunks_data        = []
    for s in chunks_data_:
        chunks_data.append( Array( 'f', s, lock=lock ) )
    
    processes = []
    for dev in range( N_gpu ):
        
        p = Process( target=deproject_detector, args=( 
            chunks_feed_ra[dev], chunks_feed_dec[dev], chunks_feed_pa[dev], det_pol_angle_list[dev],
            beam_nside_list[dev], M_list[dev],
            _I_list[dev], _Q_list[dev], _U_list[dev], _V_list[dev],
            dev,
            chunks_data[dev] ) )
        p.start()
        processes.append( p )

    for p in processes:
        p.join()
    
    det_stream = numpy.asarray( chunks_data ).ravel()
    
    print det_stream

    return det_stream
