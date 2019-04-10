import joblib
from joblib import Parallel, delayed
import multiprocessing

import healpy
import numpy                                                                                                  

from numpy import sin, cos                                                                                    
from numpy import conj                                                                                        
                                                                                                              
def compute_vpm_jones_matrix( d, k, theta, phi ):                                                                           
                                                                                                              
    G_TM_para = -1                                 # This is Gamma^TE_para in Chuss paper.                              
    G_TE_para = -1                                 # This is Gamma^TM_para in Chuss paper                               
    G_TM_perp = -numpy.exp( 2*1.j*d*k*cos(theta) ) # This is Gamma^TM_perp in Chuss paper                               
    G_TE_perp = -numpy.exp( 2*1.j*d*k*cos(theta) ) # This is Gamma^TE_perp in Chuss paper                               
    
    c2 = cos(phi)**2
    s2 = sin(phi)**2
    sc = cos(phi)*sin(phi)

    J = numpy.asarray(                                                                                        
    [                                                                                                         
        [ G_TM_para*c2 - G_TM_perp*s2, (G_TE_para - G_TE_perp)*sc ],
        [ (G_TM_para-G_TM_perp)*sc, G_TE_para*s2 - G_TE_perp*c2 ]
    ] )                                                                                                       
                                                                                                              
    return J                                                                                            
                                                                                                              
def E( i, J_vpm ):                                                                                 
                                                                                                              
    m,n = numpy.unravel_index( i-1, (2,2) )                                                                   
                                                                                                              
    return J_vpm[m][n] * numpy.conj( J_vpm[m][n] )                                 
                                                                                                              
def F( i, j, J_vpm ):                                                                              
                                                                                                              
    m1,n1 = numpy.unravel_index( i-1, (2,2) )                                                                 
    m2,n2 = numpy.unravel_index( j-1, (2,2) )                                                                 
                                                                                                              
    return numpy.real( J_vpm[m1,n1] * numpy.conj( J_vpm[m2][n2] ) )              
                                                                                                              
def G( i, j, J_vpm ):                                                                              
                                                                                                              
    m1,n1 = numpy.unravel_index( i-1, (2,2) )                                                                 
    m2,n2 = numpy.unravel_index( j-1, (2,2) )                                                                 
                                                                                                              
    return numpy.imag( J_vpm[m1][n1] * numpy.conj( J_vpm[m2][n2] ) )                
                                                                                                              
                                                                                                              
def M_vpm( d, k, theta, phi ):                                                                                

    J_vpm = compute_vpm_jones_matrix( d,k,theta,phi )

    M00 =  0.5*( E(1,J_vpm) + E(2,J_vpm) + E(3,J_vpm) + E(4,J_vpm) )          
    M01 =  0.5*( E(1,J_vpm) - E(2,J_vpm) - E(3,J_vpm) + E(4,J_vpm) )          
    M02 =  F(1,3,J_vpm) + F(4,2,J_vpm)                                                        
    M03 = -G(1,3,J_vpm) - G(4,2,J_vpm)                                                        
                                                                                                              
    M10 =  0.5*( E(1,J_vpm) - E(2,J_vpm) + E(3,J_vpm) - E(4,J_vpm) )          
    M11 =  0.5*( E(1,J_vpm) + E(2,J_vpm) - E(3,J_vpm) - E(4,J_vpm) )          
    M12 =  F(1,3,J_vpm) - F(4,2,J_vpm)                                                        
    M13 = -G(1,3,J_vpm) + G(4,2,J_vpm)                                                        
                                                                                                              
    M20 =  F(1,4,J_vpm) + F(3,2,J_vpm)                                                        
    M21 =  F(1,4,J_vpm) - F(3,2,J_vpm)                                                        
    M22 =  F(1,2,J_vpm) + F(3,4,J_vpm)                                                        
    M23 = -G(1,2,J_vpm) + G(3,4,J_vpm)                                                        
                                                                                                              
    M30 = G(1,4,J_vpm) + G(3,2,J_vpm)                                                         
    M31 = G(1,4,J_vpm) - G(3,2,J_vpm)                                                         
    M32 = G(1,2,J_vpm) + G(3,4,J_vpm)                                                         
    M33 = F(1,2,J_vpm) - F(3,4,J_vpm)                                                         
                                                                                                              
    M = numpy.asarray([[M00,M01,M02,M03],                                                                     
                       [M10,M11,M12,M13],                                                                     
                       [M20,M21,M22,M23],                                                                     
                       [M30,M31,M32,M33] ] )                                                                  
                                                                                                              
    return numpy.abs( M )


def M_vpm_to_healpix( d, k, max_theta, grid_rotation, beam_nside=256 ):
    
    max_pix  = healpy.ang2pix( beam_nside, max_theta, 2*numpy.pi )
    pixels   = numpy.arange( 0, max_pix, 1 )

    thts, phi = healpy.pix2ang( beam_nside, pixels )    
    
    M = numpy.zeros( (3,3,pixels.size) )
    
    # Use joblib to speed things up a little
    num_cores = multiprocessing.cpu_count()
    print 'using ', num_cores, ' threads'
    
    # Create arrays of distances, k's and grid rotations
    ds  = numpy.repeat( d, pixels.size )
    ks  = numpy.repeat( k, pixels.size )
    grs = numpy.repeat( grid_rotation, pixels.size )
    
    ms_vpm = Parallel( n_jobs=num_cores )( 
        delayed( M_vpm )( d, k, theta, gr ) for ds, ks, theta, gr in zip( ds, ks, thts, grs ) )
    
    ms_vpm = numpy.asarray( ms_vpm )
    
    ms_vpm = numpy.swapaxes( ms_vpm, 0, 2 )

    return ms_vpm


