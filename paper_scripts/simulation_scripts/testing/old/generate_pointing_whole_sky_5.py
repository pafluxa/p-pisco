# coding: utf-8                                                
import numpy                                                 
import healpy                                                 
from healpy.rotator import euler_matrix_new, Rotator                             
                                                       
nside = 256                                                  
pixels = numpy.arange( healpy.nside2npix( nside ) )                              
                                                       
tht,phi = healpy.pix2ang( nside, pixels )                                   
                                                       
tht_rot,phi_rot = Rotator(rot=(0.0,0.0,0.0))(tht, phi)                            
                                                       
ra = phi_rot                                                 
dec = numpy.pi/2.0 - tht_rot                                         
                                                       
ra = numpy.tile( ra, 98 )                                           
dec = numpy.tile( dec, 98 )                                          
                                                       
# add some random errors to the pointing                                   
pixSize = healpy.nside2resol( nside )                                     
factor = 0.4                                                
x = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, ra.size )               
factor = 0.4                                                
y = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, dec.size )               
                                                       
decNoise = dec + y                                              
raNoise = ra + x / numpy.cos(dec)                                      
                                                       
# Make sure declation doesn't blow up                                     
decNoise[ decNoise > numpy.pi/2.0 ] = numpy.pi/2.0                             
decNoise[ decNoise < -numpy.pi/2.0 ] = -numpy.pi/2.0                             
                                                       
pa0 = numpy.zeros_like( tht_rot )                                       
pa1 = pa0 + numpy.radians(-45.0)                                           
pa2 = pa0 + numpy.radians(-30.0)                                           
pa3 = pa0 + numpy.radians(-15.0)                                           
pa4 = pa0 + numpy.radians( 15.0)                                           
pa5 = pa0 + numpy.radians( 30.0)                                           
pa6 = pa0 + numpy.radians( 45.0)                                           
pa = numpy.concatenate( (pa0,pa1,pa2,pa3,pa4,pa5,pa6) ) 
pa = numpy.tile( pa, 14 )                                  
                                                       
ra = raNoise.reshape ( (1,-1) )                                       
dec = decNoise.reshape( (1,-1) )                                       
pa = pa.reshape ( (1,-1) )                                          
                                                       
#numpy.savez( '../../data/pointing/wholeSkyPointingRandomNoise_ndays_128_nscans_1_sps_1Hz.npz' , ra=ra,dec=dec,pa=pa )
numpy.savez( 'test_pointing.npz' , ra=ra,dec=dec,pa=pa )
