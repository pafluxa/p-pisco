'''
Core methods for convolution: deproject_sky
'''
import pylab
import numpy
import healpy

from pisco.convolution._convolution import *
from pisco.beam_analysis.mueller import ComplexMuellerMatrix as CM

import time

import sys

def deproject_sky_for_feedhorn(
        feed_ra, feed_dec, feed_pa, 
        det_pol_angle, 
        sky,
        beam_nside, beam0_co, beam0_cx, beam90_co, beam90_cx, 
        gpu_dev=0, maxmem=4096,
        grid_size=numpy.deg2rad(5.0) ):
    '''
    '''
    
    print "convolving"

    # Extract maps from input and parse them to float32
    _I = sky[0].astype( 'float32' )
    _Q = sky[1].astype( 'float32' )
    _U = sky[2].astype( 'float32' )
    _V = sky[3].astype( 'float32' )

    # Create buffers for detector streams
    det_stream = numpy.zeros_like( feed_ra, dtype='float64' )
    
    # Build Complex Mueller Matrices of the optics
    tic = time.time()
    M = CM.make_optical_mueller_matrix( beam_nside, beam0_co, beam0_cx, beam90_co, beam90_cx , grid_size )
    maxPix = M.max_pix
    #print maxPix, numpy.degrees( healpy.pix2ang( beam_nside, maxPix ) )
    toc = time.time()
    
    # I like tic-toc
    elapsed = -(tic - toc)
    print "\tmake_optical_mueller_matrix:", elapsed*1000.0, 'ms'
   
    # Build buffer of pixels as a evaluation grid
    sky_nside   = healpy.npix2nside(_I.size ) 
    nsamples    = det_stream.size
    
    buffer_size = (int)( healpy.query_disc( sky_nside, (0,0,1), grid_size ).size * 1.5 )
    #buffer_size = (int)( healpy.query_disc( sky_nside, (0,0,1), grid_size*2 ).size )
    
    # Normalize by M_TT
    d_omega_sky   = healpy.nside2pixarea( sky_nside )
    d_omega_beam  = healpy.nside2pixarea( beam_nside )
    norm          = numpy.sum( M.M_TT ) * d_omega_beam/d_omega_sky
    
    # Get M_TT and normalize accordingly
    M_beams = M.get_M( complex = True )
    M_beams = M_beams / norm
    
    reqMem  = (nsamples * buffer_size + M_beams.size + _I.size * 4)
    reqMem *= 4
    
    #print reqMem/1e6, maxmem

    chunkit = False
    if reqMem / 1e6 > maxmem:
        chunkit = True    
        #print 'chunking'    
    
    if chunkit:
        
        tic = time.time()
        
        nchunks =  (int)( reqMem / 1e6 )/maxmem + 1
        cra  = numpy.array_split(  feed_ra, nchunks )
        cdec = numpy.array_split( feed_dec, nchunks )
        cpa  = numpy.array_split(  feed_pa, nchunks )
        
        n = 1
        det_stream = numpy.empty( 1, dtype='float64' )
        for _ra, _dec, _pa in zip(cra,cdec,cpa):
            
            nsamples = _ra.size
        
            num_pixels      = numpy.empty( (nsamples), dtype='int32')
            evalgrid_pixels = numpy.empty( (nsamples , buffer_size) , dtype='int32')
            
            vectorized_query_disc( _ra, _dec, sky_nside, grid_size, evalgrid_pixels, num_pixels )

            tod = numpy.empty( nsamples, dtype='float64' ) 
            
            deproject_detector(
                _ra, _dec, _pa,
                det_pol_angle,
                beam_nside,
                numpy.real(M_beams), numpy.imag(M_beams), maxPix,
                num_pixels, evalgrid_pixels,
                _I,_Q,_U,_V, 
                gpu_dev,
                tod )
            
            det_stream = numpy.concatenate( (det_stream,tod) ) 
            
            print 'chunk:', n, 'out of', len(cra)
            n = n + 1


        det_stream = det_stream[1::]
        
        toc = time.time()
        # I like tic-toc
        elapsed = -(tic - toc)
        print "convolve single detector:", elapsed*1000.0, 'ms'
    
        return det_stream 

    else:
        
        tic = time.time()

        nsamples        = feed_ra.size
        num_pixels      = numpy.empty( (nsamples), dtype='int32')
        evalgrid_pixels = numpy.zeros( (nsamples , buffer_size) , dtype='int32')
        vectorized_query_disc( feed_ra, feed_dec, sky_nside, grid_size, evalgrid_pixels, num_pixels )
        
        deproject_detector(
            
            feed_ra, feed_dec, feed_pa, 
            det_pol_angle,

            beam_nside,
            numpy.real(M_beams), numpy.imag(M_beams), maxPix,
            num_pixels, evalgrid_pixels,
            
            _I,_Q,_U,_V, 
            gpu_dev,
            det_stream ) 
        
        toc = time.time()
        # I like tic-toc
        elapsed = -(tic - toc)
        print "convolve single detector:", elapsed*1000.0, 'ms'

        return det_stream

def generate_TOD(
        feed_ra, feed_dec, feed_pa,  
        det_pol_angle, 
        sky,
        beam_nside, beamsor,
        gpu_dev=0, maxmem=4096,
        grid_size=numpy.deg2rad(5.0) ):
    '''
    '''
    
    print "convolving"

    # Extract maps from input and parse them to float32
    _I = sky[0].astype( 'float32' )
    _Q = sky[1].astype( 'float32' )
    _U = sky[2].astype( 'float32' )
    _V = sky[3].astype( 'float32' )

    # Create buffers for detector streams
    det_stream = numpy.zeros_like( feed_ra, dtype='float64' )
    
    # Build buffer of pixels as a evaluation grid
    sky_nside   = healpy.npix2nside(_I.size ) 
    nsamples    = det_stream.size
    
    buffer_size = (int)( healpy.query_disc( sky_nside, (0,0,1), grid_size ).size * 1.1 )
    
    # Normalize by M_TT
    d_omega_sky   = healpy.nside2pixarea( sky_nside )
    d_omega_beam  = healpy.nside2pixarea( beam_nside )
    
    # Get M_TT and normalize accordingly
    norm    = numpy.sum( beamsor.M_TT ) * d_omega_beam/d_omega_sky
    M_beams = beamsor.get_M( complex = True )
    M_beams = M_beams / norm
    
    print M_beams.dtype,  numpy.real( M_beams ).dtype

    maxPix  = beamsor.max_pix
    reqMem  = (nsamples * buffer_size + M_beams.size + _I.size * 4)
    reqMem *= 4
    
    print "buffer size in Gigabytes:", ( reqMem * 1e-9)
    print "chunks: ", (int)( reqMem / 1e6 )/maxmem + 1
    #sys.exit(1)
    
    chunkit = False
    if reqMem / 1e6 > maxmem:
        chunkit = True    
        #print 'chunking'    
    
    if chunkit:
        
        tic = time.time()
        
        nchunks =  (int)( reqMem / 1e6 )/maxmem + 1
        cra  = numpy.array_split(  feed_ra, nchunks )
        cdec = numpy.array_split( feed_dec, nchunks )
        cpa  = numpy.array_split(  feed_pa, nchunks )
        
        n = 1
        det_stream = numpy.empty( 1, dtype='float64' )
        for _ra, _dec, _pa in zip(cra,cdec,cpa):
            
            nsamples = _ra.size
        
            num_pixels      = numpy.empty( (nsamples), dtype='int32')
            evalgrid_pixels = numpy.empty( (nsamples , buffer_size) , dtype='int32')
            
            vectorized_query_disc( _ra, _dec, sky_nside, grid_size, evalgrid_pixels, num_pixels )

            tod = numpy.empty( nsamples, dtype='float64' ) 
            
            deproject_detector(
                _ra, _dec, _pa,
                det_pol_angle,
                beam_nside,
                numpy.real(M_beams), numpy.imag(M_beams), maxPix,
                num_pixels, evalgrid_pixels,
                _I,_Q,_U,_V, 
                gpu_dev,
                tod )
            
            det_stream = numpy.concatenate( (det_stream,tod) ) 

        det_stream = det_stream[1::]
        
        toc = time.time()
        # I like tic-toc
        elapsed = -(tic - toc)
        print "convolve single detector:", elapsed*1000.0, 'ms'
    
        return det_stream 

    else:
        
        tic = time.time()

        nsamples        = feed_ra.size
        num_pixels      = numpy.empty( (nsamples), dtype='int32')
        evalgrid_pixels = numpy.zeros( (nsamples , buffer_size) , dtype='int32')
        vectorized_query_disc( feed_ra, feed_dec, sky_nside, grid_size, evalgrid_pixels, num_pixels )
        
        deproject_detector(
            
            feed_ra, feed_dec, feed_pa, 
            det_pol_angle,

            beam_nside,
            numpy.real(M_beams), numpy.imag(M_beams), maxPix,
            num_pixels, evalgrid_pixels,
            
            _I,_Q,_U,_V, 
            gpu_dev,
            det_stream ) 
        
        toc = time.time()
        # I like tic-toc
        elapsed = -(tic - toc)
        print "convolve single detector:", elapsed*1000.0, 'ms'

        return det_stream
