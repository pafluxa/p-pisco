#ifndef _PTGRUOTINESH
#define _PTGRUOTINESH

void                                                                                                          
libpointing_get_receiver_ICRS_coords 
(                                                                                                             
// Focal plane                                                                                                
int ndets, double det_dx[], double det_dy[], double det_pol_angle[],                                          
                                                                                                              
// Boresight coordinates                                                                             
int nsamples, double utc_ctime[], double azimuth[], double altitude[], double rotation[], int ptg_mask[],          
                                                                                                              
// Site specifications (longitude, latitude, height )                                                         
double site_latitude, double site_longitude, double site_height,                                                            
                                                                                                              
// Polar motions and UTC - UT1 difference (from astropy!)                                                     
double xp, double yp, double dut1,                                                                                          
                                                                                                              
// Output: table of ndets x nsamples for detector right ascention, declination and parallactic angles.        
double det_ra[], double det_dec[], double det_pa[]                                                                          
);

void                                                                                                          
libpointing_get_receiver_horizontal_coords                                                                          
(                                                                                                             
// Focal plane                                                                                                
int ndets, double det_dx[], double det_dy[], double det_pol_angle[],                                          
                                                                                                              
// Boresight coordinates                                                                                      
int nsamples, double utc_ctime[], double azimuth[], double altitude[], double rotation[],                     
                                                                                                              
// Output: table of ndets x nsamples for detector azimuth, altitude and rotation      
double *det_az, double *det_alt, double *det_rot                                                              
);

void                                                                                                          
libpointing_get_receiver_source_centered_coords                                                                            
(                                                                                                             
    int nsamples, int ndets,                                                                                  
    // Detector coordinates, in some coordinate system                                                        
    double *array_phi, double *array_theta, double *array_psi,                                                
    // Source coordinates                                                                                     
    double *source_phi, double *source_theta,                                                                   
    // Output coordinates                                                                                     
    double *track_phi, double *track_theta                                                                    
);

void                                                                                                          
libpointing_transform_ICRS_to_horizontal_coords                                                                            
(                                                                                                             
// Source coordinates                                                                                         
int nsamples, double utc_ctime[], double source_ra, double source_dec,                                        
                                                                                                              
// Site specifications (longitude, latitude, height )                                                         
double site_latitude, double site_longitude, double site_height,                                              
                                                                                                              
// Polar motions and UTC - UT1 difference (from astropy!)                                                     
double xp, double yp, double dut1,                                                                            
                                                                                                              
double source_azimuth[], double source_altitude[]                                                             
);

#endif
