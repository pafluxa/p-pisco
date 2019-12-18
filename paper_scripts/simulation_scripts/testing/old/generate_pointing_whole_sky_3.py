# coding: utf-8                                                                                               
import numpy                                                                                                  
import healpy                                                                                                 
                                                                                                              
nside = 128                                                                                                   
pixels = numpy.arange( healpy.nside2npix( nside ) )                                                           
                                                                                                              
dec, ra = healpy.pix2ang( nside, pixels )                                                                                                                                                                              
dec = numpy.pi/2.0 - dec                                                                                  
                                                                                                              
# add some random errors to the pointing                                                                      
pixSize = healpy.nside2resol( nside )                                                                         
factor = 0.07 
y = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, dec.size  )                             
factor = 0.07
x = numpy.random.uniform(-factor*pixSize / 2.0, factor*pixSize / 2.0, ra.size  )                              
                                                                                                              
# with offsets added                                                                                          
dec1 = dec + y                                                                                                
ra1  = ra  + x / numpy.cos(dec)                                                                               
                                                                                                              
# tile                                                                                                        
ra1  = numpy.tile(  ra1, 3 )                                                                                  
dec1 = numpy.tile( dec1, 3 )                                                                                  
                                                                                                              
# generate position angles: 1 per pass.                                                                       
pa0 = numpy.zeros_like( ra )                                                                                  
pa1 = pa0 + numpy.pi/4.0                                                                                      
pa2 = pa0 + numpy.pi/2.0                                                                                      
pa  = numpy.concatenate( (pa0,pa1,pa2) )

ra = ra1
dec = dec1                                                                      
  
'''                                                                                                            
# with offsets subtracted                                                                                     
dec2 = dec - x                                                                                                
ra2  = ra  - y / numpy.cos(dec)                                                                               
                                                                                                              
# tile                                                                                                        
ra2  = numpy.tile(  ra2, 3 )                                                                                  
dec2 = numpy.tile( dec2, 3 )                                                                                  
                                                                                                              
# concatenate                                                                                                 
ra  = numpy.concatenate( ( ra1, ra2) )                                                                        
dec = numpy.concatenate( (dec1,dec2) )                                                                        
pa  = numpy.concatenate( (  pa,  pa) )                                                                        
'''
                                                                                                              
# Make sure declation doesn't go outside limits                                                               
dec[ dec >  numpy.pi/2.0 ] =  numpy.pi/2.0                                                                    
dec[ dec < -numpy.pi/2.0 ] = -numpy.pi/2.0                                                                    
                                                                                                              
# reshape to fool PISCO                                                                                       
ra  =  ra.reshape( (1,-1) )                                                                                   
dec = dec.reshape( (1,-1) )                                                                                   
pa  =  pa.reshape( (1,-1) )                                                                                   
                                                                                                              
print ra.shape, dec.shape, pa.shape                                                                           
numpy.savez( 'test_pointing.npz', ra=ra, dec=dec, pa=pa )
