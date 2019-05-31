# coding: utf-8                                                                                               
import numpy                                                                                                  
import healpy                                                                                                 
from healpy.rotator import euler_matrix_new, Rotator                                                          
                                                                                                              
nside = 128                                                                                                   
pixels = numpy.arange( healpy.nside2npix( nside ) )                                                           
                                                                                                              
tht,phi = healpy.pix2ang( nside, pixels )                                                                     
                                                                                                              
tht_rot,phi_rot = Rotator(rot=(0.0,0.0,0.0))(tht, phi)                                                        
                                                                                                              
ra  = phi_rot                                                                                                 
dec = numpy.pi/2.0 - tht_rot                                                                                  
                                                                                                              
ra  = numpy.tile( ra, 3 )                                                                                     
dec = numpy.tile( dec, 3 )                                                                                    
                                                                                                              
# add some random errors to the pointing                                                                      
pixSize = healpy.nside2resol( nside )                                                                         
factor = 0.07                                                                                                 
x = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, ra.size  )                              
factor = 0.07                                                                                                
y = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, dec.size  )                             
                                                                                                              
decNoise = dec + y                                                                                            
raNoise  = ra  + x / numpy.cos(dec)                                                                           
                                                                                                              
# Make sure declation doesn't blow up                                                                         
decNoise[ decNoise >  numpy.pi/2.0 ] =  numpy.pi/2.0                                                          
decNoise[ decNoise < -numpy.pi/2.0 ] = -numpy.pi/2.0                                                          
                                                                                                              
pa0 = numpy.zeros_like( tht_rot )                                                                             
pa1 = pa0 + numpy.pi/4.0                                                                                      
pa2 = pa0 + numpy.pi/2.0                                                                                      
pa  = numpy.concatenate( (pa0,pa1,pa2) )                                                                      
                                                                                                              
ra  = raNoise.reshape ( (1,-1) )                                                                              
dec = decNoise.reshape( (1,-1) )                                                                              
pa  = pa.reshape ( (1,-1) )                                                                                   
                                                                                                              
numpy.savez( '../../data/pointing/wholeSkyPointingRandomNoiseF0d07_ndays_128_nscans_1_sps_1Hz.npz' , ra=ra,dec=dec,pa=pa )
