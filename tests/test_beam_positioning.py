'''
double dx = s_beam_dx[ pix ];                                                                 
double dy = s_beam_dy[ pix ];                                                                 
double cdx = cos(dx);                                                                           
double sdx = sin(dx);                                                                           
double cdy = cos(dy);                                                                           
double sdy = sin(dy);                                                                           
                                                                                              
double cr  = cdx * cdy;                                                                                  
double r   = acos(cr);                                                                                   
double sr  = sin(r);                                                                                     
double alpha = atan2(sdy, sdx * cdy );                                                        
                                                                                          
double gamma = (M_PI_2 - alpha - pa_det);                                                     
double sg    = sin(gamma);                                                                        
double cg    = cos(gamma);                                                                        
                                                                                              
double sdecbc  = sin( dec_det );                                                                  
double cdecbc  = cos( dec_det );                                                                  
                                                                                              
double dec_pix = asin ( cr * sdecbc + sr * cdecbc * cg);                                        
double  ra_pix = ra_det +                                                                          
             atan2( sr * sg,                                                                  
                    cr * cdecbc - sr * sdecbc * cg);                                      
                                                                                          
double delta = atan2( cos(dec_det)*sin(pa_det), sin(dec_det) );                               
double     L =  acos( cos(dec_det)*cos(pa_det) ); 
p = atan2( sin(dx+delta), cos(dy)*1./tan(L) - sin(dy)*cos(dx+delta) );

double tandy = cos(pix_phi) * tan( pix_tht );                                                         
double  dy   = atan( tandy );                                                                         
double tandx = tan(pix_phi) * sin( dy );                                                              
double  dx   = atan( tandx );

'''

import healpy
import numpy


